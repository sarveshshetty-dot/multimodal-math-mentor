try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
import os

# Module-level singleton so it only loads once per Streamlit session
_model_instance = None

def _get_model():
    global _model_instance
    if WhisperModel is None:
        return None
    if _model_instance is None:
        # tiny model: only ~75MB, fast CPU inference
        _model_instance = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _model_instance

class SpeechProcessor:
    """Wrapper around faster-whisper to transcribe audio input."""

    def process_audio(self, audio_file_path: str) -> str:
        """
        Transcribes the given audio file path using the tiny Whisper model.
        Model is cached globally so it only loads once.
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        try:
            model = _get_model()
            # beam_size=1 for fastest CPU inference
            segments, info = model.transcribe(audio_file_path, beam_size=1)
            text = " ".join([segment.text for segment in segments])
            return text.strip()
        except Exception as e:
            print(f"Speech processing error: {e}")
            return ""
