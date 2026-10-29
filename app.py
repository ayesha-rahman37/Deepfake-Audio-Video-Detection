
"""Streamlit interface for Deepfake Audio-Video Detection."""

import tempfile
from pathlib import Path

import streamlit as st

from app_components.validation import validate_uploaded_file
from app_components.pipeline import run_pipeline


st.set_page_config(
    page_title="Deepfake Audio-Video Detection",
    page_icon="🎬",
    layout="wide",
)


SUPPORTED_TYPES = ["mp4", "mov", "avi", "mkv", "webm"]
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB


def display_prediction(result):
    """Display prediction, confidence, and window-wise evidence."""

    if not isinstance(result, dict):
        st.error("The inference pipeline returned an unexpected result.")
        st.write(result)
        return

    prediction = (
        result.get("prediction")
        or result.get("label")
        or result.get("class_name")
        or result.get("final_prediction")
    )

    confidence = (
        result.get("confidence")
        or result.get("final_confidence")
        or result.get("prediction_confidence")
    )

    evidence = (
        result.get("window_evidence")
        or result.get("window_results")
        or result.get("windows")
        or result.get("evidence")
    )

    if prediction is not None:
        label = str(prediction).strip()
        normalized_label = label.lower()

        if "fake" in normalized_label:
            st.error(f"Prediction: {label}")
        elif "real" in normalized_label:
            st.success(f"Prediction: {label}")
        else:
            st.info(f"Prediction: {label}")
    else:
        st.warning("The result does not contain a prediction label.")

    if confidence is not None:
        try:
            confidence_value = float(confidence)

            # Support both 0–1 and 0–100 confidence values.
            if 0 <= confidence_value <= 1:
                confidence_value *= 100

            st.metric("Confidence", f"{confidence_value:.2f}%")
            st.progress(max(0.0, min(confidence_value / 100, 1.0)))

        except (TypeError, ValueError):
            st.write("Confidence:", confidence)

    st.subheader("Window-wise Evidence")

    if evidence is not None:
        if isinstance(evidence, (list, tuple)):
            if evidence:
                st.dataframe(
                    [
                        item if isinstance(item, dict) else {"Evidence": item}
                        for item in evidence
                    ],
                    use_container_width=True,
                )
            else:
                st.info("No window-wise evidence was returned.")
        elif isinstance(evidence, dict):
            st.json(evidence)
        else:
            st.write(evidence)
    else:
        st.info("Window-wise evidence is not available in this result.")

    with st.expander("View complete inference result"):
        st.json(result)


def save_uploaded_video(uploaded_file):
    """Save an uploaded video temporarily for inference."""

    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return Path(temp_file.name)


def main():
    st.title("🎬 Deepfake Audio-Video Detection")
    st.write(
        "Upload a video to analyze it using the configured "
        "preprocessing and trained-model inference pipeline."
    )

    st.info(
        "Supported formats: MP4, MOV, AVI, MKV, WEBM. "
        "Maximum upload size: 200 MB."
    )

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=SUPPORTED_TYPES,
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.caption("Upload a video to begin.")
        return

    file_size = uploaded_file.size
    file_size_mb = file_size / (1024 * 1024)

    col1, col2 = st.columns(2)

    with col1:
        st.write("**File name:**", uploaded_file.name)

    with col2:
        st.write("**File size:**", f"{file_size_mb:.2f} MB")

    if file_size > MAX_FILE_SIZE:
        st.error("The uploaded file exceeds the 200 MB limit.")
        return

    # Validate the uploaded file before processing.
    is_valid, validation_message = validate_uploaded_file(uploaded_file)

    if not is_valid:
        st.error(validation_message)
        return

    st.success(validation_message)

    if not st.button(
        "Analyze Video",
        type="primary",
        use_container_width=True,
    ):
        return

    temp_video_path = None

    try:
        with st.status(
            "Processing video...",
            expanded=True,
        ) as status:
            st.write("Saving uploaded video...")
            temp_video_path = save_uploaded_video(uploaded_file)

            st.write("Running preprocessing and model inference...")
            raw_prediction = run_pipeline(str(temp_video_path))

            status.update(
                label="Video analysis completed.",
                state="complete",
                expanded=False,
            )

        st.subheader("Detection Results")
        display_prediction(raw_prediction)

    except FileNotFoundError as exc:
        st.error(
            "A required video file or model checkpoint was not found."
        )
        st.caption(str(exc))

    except ValueError as exc:
        st.error("The video could not be processed.")
        st.caption(str(exc))

    except Exception as exc:
        st.error("Video processing failed.")
        st.caption(f"Error details: {exc}")

    finally:
        if temp_video_path is not None:
            try:
                temp_video_path.unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    main()
