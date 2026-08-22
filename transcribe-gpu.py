"""
Video -> transcript pipeline, GPU edition. Runs in WSL2 against the local RTX 3080.

Ported from the CPU version on Nexus CT 103 (/home/claude/work/transcribe.py).
Same CLI, same classes; the differences are device selection, compute type, and
optional batched inference — all of which only matter once there's a real GPU.

Usage:
    python transcribe.py <path-or-url> [--model large-v3] [--output-dir output]

Accepts a local file path or a URL (fetched with yt-dlp). Always writes a plain-text
transcript; pass --srt for timestamped subtitles.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from faster_whisper import WhisperModel

# Batched inference lands a large speedup on GPU but arrived in faster-whisper 1.1.
# Import defensively so the script still runs against older builds.
try:
    from faster_whisper import BatchedInferencePipeline
    HAVE_BATCHED = True
except ImportError:
    HAVE_BATCHED = False


class VideoSource:
    """Resolves a user-supplied path or URL down to a local video file."""

    def __init__(self, target: str, download_dir: Path):
        self.target = target
        self.download_dir = download_dir

    def is_url(self) -> bool:
        return urlparse(self.target).scheme in ("http", "https")

    def resolve(self) -> Path:
        if self.is_url():
            return self._download()
        path = Path(self.target)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        return path

    def _download(self) -> Path:
        self.download_dir.mkdir(parents=True, exist_ok=True)
        out_template = str(self.download_dir / "%(title)s.%(ext)s")
        subprocess.run(
            ["yt-dlp", "-f", "bestvideo+bestaudio/best", "-o", out_template, self.target],
            check=True,
        )
        # yt-dlp names the file after the video title; grab the newest file we just wrote
        newest = max(self.download_dir.glob("*"), key=lambda p: p.stat().st_mtime)
        return newest


class AudioExtractor:
    """Pulls a mono 16kHz WAV out of a video file via ffmpeg — the format Whisper wants."""

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir

    def extract(self, video_path: Path) -> Path:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        audio_path = self.work_dir / f"{video_path.stem}.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-ac", "1", "-ar", "16000",
                str(audio_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return audio_path


class Transcriber:
    """
    Wraps faster-whisper.

    device="auto" lets ctranslate2 pick CUDA when it's usable and fall back to CPU
    otherwise, so the same script works if you run it somewhere without a GPU.
    compute_type follows the device: float16 is the natural fit for a 3080, while
    int8 is what you want when this lands on a CPU.
    """

    def __init__(self, model_size: str = "large-v3", device: str = "auto",
                 compute_type: str | None = None, batch_size: int = 8):
        resolved_device = self._resolve_device(device)
        if compute_type is None:
            compute_type = "float16" if resolved_device == "cuda" else "int8"

        self.device = resolved_device
        self.compute_type = compute_type
        self.batch_size = batch_size

        print(f"Loading {model_size} on {resolved_device} ({compute_type}) ...", file=sys.stderr)
        self.model = WhisperModel(model_size, device=resolved_device, compute_type=compute_type)

        # Batching only helps on GPU; on CPU it mostly just adds memory pressure.
        self.batched = None
        if HAVE_BATCHED and resolved_device == "cuda":
            self.batched = BatchedInferencePipeline(model=self.model)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        # ctranslate2 reports how many CUDA devices it can actually use, which is a
        # better test than checking for a driver — WSL can see nvidia-smi but still
        # fail to load the CUDA runtime if the libs are missing.
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def transcribe(self, audio_path: Path):
        if self.batched is not None:
            segments, info = self.batched.transcribe(
                str(audio_path), beam_size=5, batch_size=self.batch_size
            )
        else:
            segments, info = self.model.transcribe(str(audio_path), beam_size=5)
        return list(segments), info


class TranscriptWriter:
    """Writes plain-text and optional SRT output."""

    @staticmethod
    def write_text(segments, out_path: Path) -> None:
        with out_path.open("w", encoding="utf-8") as f:
            for seg in segments:
                f.write(seg.text.strip() + " ")

    @staticmethod
    def write_srt(segments, out_path: Path) -> None:
        def fmt(t: float) -> str:
            h, rem = divmod(t, 3600)
            m, s = divmod(rem, 60)
            ms = int((s - int(s)) * 1000)
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"

        with out_path.open("w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, start=1):
                f.write(f"{i}\n{fmt(seg.start)} --> {fmt(seg.end)}\n{seg.text.strip()}\n\n")


class TranscriptionPipeline:
    """Ties source resolution -> audio extraction -> transcription -> output together."""

    def __init__(self, model_size: str, output_dir: Path, work_dir: Path,
                 device: str = "auto", compute_type: str | None = None, batch_size: int = 8):
        self.output_dir = output_dir
        self.work_dir = work_dir
        self.transcriber = Transcriber(
            model_size=model_size, device=device,
            compute_type=compute_type, batch_size=batch_size,
        )

    def run(self, target: str, want_srt: bool) -> Path:
        source = VideoSource(target, download_dir=self.work_dir / "downloads")
        video_path = source.resolve()

        extractor = AudioExtractor(work_dir=self.work_dir / "audio")
        audio_path = extractor.extract(video_path)

        print(f"Transcribing {video_path.name} ...", file=sys.stderr)
        started = time.monotonic()
        segments, info = self.transcriber.transcribe(audio_path)
        elapsed = time.monotonic() - started

        print(f"Detected language: {info.language} (p={info.language_probability:.2f})", file=sys.stderr)
        # Realtime factor is the number worth watching: how many seconds of audio
        # got transcribed per second of wall clock.
        if info.duration:
            print(f"Took {elapsed:.1f}s for {info.duration:.1f}s of audio "
                  f"({info.duration / elapsed:.1f}x realtime)", file=sys.stderr)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        text_path = self.output_dir / f"{video_path.stem}.txt"
        TranscriptWriter.write_text(segments, text_path)

        if want_srt:
            srt_path = self.output_dir / f"{video_path.stem}.srt"
            TranscriptWriter.write_srt(segments, srt_path)

        return text_path


def main():
    parser = argparse.ArgumentParser(description="Transcribe a local video or URL to text.")
    parser.add_argument("target", help="Local file path or video URL")
    parser.add_argument("--model", default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size. On a 3080 large-v3 is affordable — "
                             "that's the whole point of running here instead of the server "
                             "(default: large-v3)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="Where to run (default: auto-detect CUDA)")
    parser.add_argument("--compute-type", default=None,
                        help="Override precision, e.g. float16, int8_float16, int8. "
                             "Defaults to float16 on GPU and int8 on CPU.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batched inference size on GPU (default: 8). Lower it if VRAM is tight.")
    parser.add_argument("--output-dir", default="output", help="Where to write the transcript")
    parser.add_argument("--work-dir", default="tmp", help="Scratch space for downloads/audio")
    parser.add_argument("--srt", action="store_true", help="Also write a timestamped .srt file")
    args = parser.parse_args()

    pipeline = TranscriptionPipeline(
        model_size=args.model,
        output_dir=Path(args.output_dir),
        work_dir=Path(args.work_dir),
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
    )
    out_path = pipeline.run(args.target, want_srt=args.srt)
    print(f"Transcript written to: {out_path}")


if __name__ == "__main__":
    main()
