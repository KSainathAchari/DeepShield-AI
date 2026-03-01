        "service": "DeepShield AI",
        "status": "running",
        "model_loaded": app_state.model is not None,
    }


@app.get("/health", tags=["Health"])
async def health():
    if app_state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    return {"status": "healthy", "model": "deepfake_model.h5"}


# ─────────────────────────────────────────────
# POST /predict  — main endpoint
# ─────────────────────────────────────────────
ALLOWED_CONTENT_TYPES = {
    "video/mp4", "video/avi", "video/quicktime",
    "video/x-msvideo", "video/x-matroska", "video/webm",
}
MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Detect deepfake in uploaded video",
    response_description="Prediction label, confidence, and raw sigmoid score",
)
async def predict(
    file: UploadFile = File(..., description="Video file (mp4, avi, mov, mkv, webm)"),
):
    """
    Upload a video file → extract 20 evenly-spaced frames →
    run MobileNetV2-based deepfake classifier on each frame →
    return averaged prediction.

    **Response**
    ```json
    {
      "label": "REAL" | "FAKE",
      "confidence": 94.72,
      "raw_score": 0.0528,
      "frames_analysed": 20,
      "processing_time_seconds": 1.83,
      "filename": "sample.mp4"
    }
    ```
    """
    t_start = time.perf_counter()

    # ── 1. Validate content type ───────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Accepted types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # ── 2. Read file bytes & size guard ────────
    file_bytes = await file.read()
    file_size  = len(file_bytes)
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.",
        )

    logger.info("Received '%s' — %.2f MB", file.filename, file_size / 1_048_576)

    # ── 3. Save to temp, extract frames ────────
    tmp_path = None
    try:
        tmp_path = save_upload_to_temp(file_bytes, suffix=".mp4")
        frames   = extract_frames(tmp_path, max_frames=20, img_size=224)
    except VideoProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)

    if len(frames) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not extract any frames from the uploaded video.",
        )

    # ── 4. Run inference ───────────────────────
    if app_state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available. Please retry shortly.",
        )

    raw_score = predict_frames(app_state.model, frames)

    # ── 5. Derive label + confidence ───────────
    label      = "FAKE" if raw_score >= 0.5 else "REAL"
    confidence = round(
        raw_score * 100 if label == "FAKE" else (1 - raw_score) * 100, 2
    )

    elapsed = round(time.perf_counter() - t_start, 3)
    logger.info(
        "Result: label=%s  confidence=%.2f%%  frames=%d  time=%ss",
        label, confidence, len(frames), elapsed,
    )

    return {
        "label":                    label,
        "confidence":               confidence,
        "raw_score":                round(float(raw_score), 6),
        "frames_analysed":          len(frames),
        "processing_time_seconds":  elapsed,
        "filename":                 file.filename,
    }



