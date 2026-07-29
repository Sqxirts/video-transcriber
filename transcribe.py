"""
Video -> transcript pipeline. Runs inside the video-transcriber container on Nexus.

Usage:
    python transcribe.py <path-or-url> [--model small] [--output-dir /work/output]

Accepts a local file path (mounted under /work/input) or a YouTube/web URL
(downloaded via yt-dlp first). Always writes a plain-text transcript;
pass --srt to also get timestamped subtitles.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from faster_whisper import WhisperModel


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
    """Wraps faster-whisper. Model downloads once and is cached in the mounted volume."""

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path):
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

    def __init__(self, model_size: str, output_dir: Path, work_dir: Path):
        self.output_dir = output_dir
        self.work_dir = work_dir
        self.transcriber = Transcriber(model_size=model_size)

    def run(self, target: str, want_srt: bool) -> Path:
        source = VideoSource(target, download_dir=self.work_dir / "downloads")
        video_path = source.resolve()

        extractor = AudioExtractor(work_dir=self.work_dir / "audio")
        audio_path = extractor.extract(video_path)

        print(f"Transcribing {video_path.name} ...", file=sys.stderr)
        segments, info = self.transcriber.transcribe(audio_path)
        print(f"Detected language: {info.language} (p={info.language_probability:.2f})", file=sys.stderr)

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
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large-v3"],
                         help="Whisper model size — bigger = more accurate, slower on CPU (default: small)")
    parser.add_argument("--output-dir", default="output", help="Where to write the transcript")
    parser.add_argument("--work-dir", default="tmp", help="Scratch space for downloads/audio")
    parser.add_argument("--srt", action="store_true", help="Also write a timestamped .srt file")
    args = parser.parse_args()

    pipeline = TranscriptionPipeline(
        model_size=args.model,
        output_dir=Path(args.output_dir),
        work_dir=Path(args.work_dir),
    )
    out_path = pipeline.run(args.target, want_srt=args.srt)
    print(f"Transcript written to: {out_path}")


if __name__ == "__main__":
    main()
