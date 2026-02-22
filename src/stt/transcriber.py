"""Audio transcription using distilled Whisper for French."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.utils.config import DEFAULT_STT_MODEL_ID

log = logging.getLogger(__name__)

# Formats that soundfile / librosa can read natively without ffmpeg
_NATIVE_FORMATS = {".wav", ".flac", ".ogg"}

# Mapping from short model names to backend-specific IDs
_MLX_MODEL_MAP = {
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}
_HF_MODEL_MAP = {
    "distil-large-v3": "distil-whisper/distil-large-v3",
    "large-v3": "openai/whisper-large-v3",
    "medium": "openai/whisper-medium",
    "small": "openai/whisper-small",
}


@dataclass
class Segment:
    """A transcription segment with timing information."""

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""

    text: str
    language: str
    segments: List[Segment] = field(default_factory=list)
    duration_sec: float = 0.0


def _convert_to_wav(src: Path) -> Path:
    """Convert any audio file to 16 kHz mono WAV via ffmpeg.

    Returns a temp file path. Caller is responsible for cleanup.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is required to convert audio formats like webm/mp3. "
            "Install it with: brew install ffmpeg"
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(src),
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            tmp.name,
        ],
        capture_output=True,
        check=True,
    )
    return Path(tmp.name)


class Transcriber:
    """Transcribe audio files using distilled Whisper for French.

    Backend priority:
    1. mlx-whisper (Apple Silicon Metal GPU) — fastest on Mac
    2. faster-whisper (CTranslate2, CPU INT8 or CUDA) — best on Linux/CUDA
    3. HuggingFace transformers pipeline — universal fallback

    The model is loaded lazily on the first call to transcribe().
    """

    def __init__(self, model_id: str = DEFAULT_STT_MODEL_ID) -> None:
        self._model_id = model_id
        self._backend: Optional[str] = None  # "mlx", "faster_whisper", "hf"
        self._model = None

    def _load_model(self) -> None:
        """Load the Whisper model (lazy, first call only)."""
        t0 = time.time()

        # 1. Try mlx-whisper (Apple Silicon)
        try:
            import mlx_whisper  # noqa: F401

            mlx_repo = _MLX_MODEL_MAP.get(self._model_id, self._model_id)
            # Trigger model download/cache by importing — actual loading happens at transcribe time
            self._model = mlx_repo
            self._backend = "mlx"
            log.info("Using mlx-whisper with %r (%.1fs)", mlx_repo, time.time() - t0)
            return
        except ImportError:
            pass

        # 2. Try faster-whisper (CTranslate2)
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_id,
                device="cpu",
                compute_type="int8",
            )
            self._backend = "faster_whisper"
            log.info("Loaded faster-whisper model %r in %.1fs", self._model_id, time.time() - t0)
            return
        except ImportError:
            pass

        # 3. Fallback to HuggingFace pipeline
        import torch
        from transformers import pipeline

        hf_id = _HF_MODEL_MAP.get(self._model_id, self._model_id)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model = pipeline(
            "automatic-speech-recognition",
            model=hf_id,
            device=device,
            torch_dtype=torch.float32,
        )
        self._backend = "hf"
        log.info("Loaded HF pipeline model %r on %s in %.1fs", hf_id, device, time.time() - t0)

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, WEBM, OGG...).

        Returns:
            TranscriptionResult with transcribed text and metadata.

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self._model is None:
            self._load_model()

        t0 = time.time()

        if self._backend == "mlx":
            result = self._transcribe_mlx(audio_path)
        elif self._backend == "faster_whisper":
            result = self._transcribe_faster_whisper(audio_path)
        else:
            # HF pipeline needs PCM audio — convert non-native formats via ffmpeg
            wav_tmp: Path | None = None
            try:
                if audio_path.suffix.lower() not in _NATIVE_FORMATS:
                    wav_tmp = _convert_to_wav(audio_path)
                    audio_path = wav_tmp
                result = self._transcribe_hf_pipeline(audio_path)
            finally:
                if wav_tmp is not None:
                    wav_tmp.unlink(missing_ok=True)

        log.info("Transcribed in %.1fs (%s, audio=%.1fs)", time.time() - t0, self._backend, result.duration_sec)
        return result

    def _transcribe_mlx(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe using mlx-whisper (Apple Silicon Metal GPU)."""
        import mlx_whisper

        # mlx-whisper handles format conversion internally via ffmpeg
        result = mlx_whisper.transcribe(
            str(audio_path),
            language="fr",
            path_or_hf_repo=self._model,  # MLX repo ID
        )

        segments: List[Segment] = []
        for seg in result.get("segments", []):
            segments.append(Segment(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", "").strip(),
            ))

        # Estimate duration from last segment end
        duration = segments[-1].end if segments else 0.0

        return TranscriptionResult(
            text=result.get("text", "").strip(),
            language=result.get("language", "fr"),
            segments=segments,
            duration_sec=duration,
        )

    def _transcribe_faster_whisper(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe using faster-whisper (CTranslate2)."""
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language="fr",
            beam_size=1,
        )

        segments: List[Segment] = []
        text_parts: List[str] = []
        for seg in segments_iter:
            segments.append(Segment(start=seg.start, end=seg.end, text=seg.text.strip()))
            text_parts.append(seg.text.strip())

        return TranscriptionResult(
            text=" ".join(text_parts),
            language=info.language,
            segments=segments,
            duration_sec=info.duration,
        )

    def _transcribe_hf_pipeline(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe using HuggingFace transformers pipeline."""
        import soundfile as sf

        audio, sr = sf.read(str(audio_path), dtype="float32")

        # Resample to 16 kHz if needed (should already be from ffmpeg conversion)
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Convert stereo to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        duration = len(audio) / sr

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The input name `inputs` is deprecated")
            result = self._model(
                {"raw": audio, "sampling_rate": sr},
                generate_kwargs={"language": "fr"},
                return_timestamps=True,
            )

        segments: List[Segment] = []
        if "chunks" in result:
            for chunk in result["chunks"]:
                ts = chunk.get("timestamp", (0.0, 0.0))
                segments.append(
                    Segment(
                        start=ts[0] if ts[0] is not None else 0.0,
                        end=ts[1] if ts[1] is not None else 0.0,
                        text=chunk["text"].strip(),
                    )
                )

        return TranscriptionResult(
            text=result["text"].strip(),
            language="fr",
            segments=segments,
            duration_sec=duration,
        )
