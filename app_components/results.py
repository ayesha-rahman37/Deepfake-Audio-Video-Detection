from typing import Any


def format_prediction(prediction: Any) -> dict:
    if not isinstance(prediction, dict):
        raise ValueError("Prediction must be a dictionary.")

    label = str(prediction.get("label", "")).strip().lower()
    if label not in {"real", "fake"}:
        raise ValueError("Prediction label must be Real or Fake.")

    confidence = prediction.get("confidence")
    if not isinstance(confidence, (int, float)):
        raise ValueError("Confidence must be numeric.")

    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")

    evidence = prediction.get("window_evidence", [])
    if not isinstance(evidence, list):
        raise ValueError("window_evidence must be a list.")

    return {
        "label": label.capitalize(),
        "confidence": float(confidence),
        "window_evidence": evidence,
    }