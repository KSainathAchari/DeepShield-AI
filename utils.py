n_sample = min(max_frames, total_frames)
    indices  = np.linspace(0, total_frames - 1, num=n_sample, dtype=int)

    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning("Frame %d could not be read — skipping.", idx)
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()
    logger.info("Extracted %d / %d requested frames.", len(frames), n_sample)

    if len(frames) == 0:
        raise VideoProcessingError(
            "All frame reads failed. The video may be corrupted or incompatible."
        )

    return frames


# ─────────────────────────────────────────────
# Batch Inference
# ─────────────────────────────────────────────
def predict_frames(
    model: tf.keras.Model,
    frames: list[np.ndarray],
    batch_size: int = 32,
) -> float:
    """
    Run the deepfake model on a list of frames and return the averaged score.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded DeepShield model (sigmoid output).
    frames : list of np.ndarray
        Pre-processed frames from `extract_frames`.
    batch_size : int
        Inference batch size (default 32).

    Returns
    -------
    float
        Mean sigmoid score across all frames (0 = REAL, 1 = FAKE).
    """
    frame_array = np.stack(frames, axis=0)          # shape: (N, 224, 224, 3)
    logger.info("Running inference on %d frames …", len(frames))

    scores = model.predict(frame_array, batch_size=batch_size, verbose=0)
    scores = scores.squeeze()                        # (N,) or scalar

    # Handle single-frame edge case
    if scores.ndim == 0:
        scores = np.array([float(scores)])

    mean_score = float(np.mean(scores))
    logger.info(
        "Frame scores — mean: %.4f  min: %.4f  max: %.4f",
        mean_score, float(np.min(scores)), float(np.max(scores)),
    )
    return mean_score
