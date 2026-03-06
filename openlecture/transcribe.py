from functools import lru_cache
from pathlib import Path
from faster_whisper import WhisperModel

# Define which version of the AI model to use
MODEL_SIZE = "medium"

# This "decorator" saves the model in memory so we don't reload it every time
@lru_cache(maxsize=1)
def _get_model() -> WhisperModel:
    """Loads the Whisper model and keeps it cached."""
    
    print("Loading Whisper model...")
    
    return WhisperModel(
        MODEL_SIZE,
        device="auto",
        compute_type="auto"
    )

def transcribe_audio(audio_path: str) -> str:
    """Main function to turn an audio file into text."""
    
    # 1. PRE-FLIGHT CHECKS
    # Check if the path provided is empty or just spaces
    if not audio_path or not audio_path.strip():
        raise ValueError("audio_path must not be empty.")

    # Convert the string path into a special 'Path' object for easier handling
    audio_file = Path(audio_path)

    # Check if the file actually exists on the disk
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    # Check if the path points to a file and not a folder
    if not audio_file.is_file():
        raise ValueError(f"Audio path is not a file: {audio_file}")

    # 2. LOADING THE AI MODEL
    try:
        model = _get_model()
    except Exception as exc:
        # If something goes wrong during loading, tell the user why
        raise RuntimeError("Failed to load Whisper model.") from exc

    print("Transcribing audio...")

    # 3. TRANSCRIPTION PROCESS
    try:
        # Ask the model to listen. It returns segments of text
        segments, _ = model.transcribe(
            str(audio_file),
            beam_size=5,
            vad_filter=True
            )
    except Exception as exc:
        raise RuntimeError(f"Failed to transcribe audio file: {audio_file}") from exc

    # 4. CLEANING AND JOINING RESULTS
    # We take every piece of text, clean extra spaces, and join them with a space
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip())