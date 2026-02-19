import re
import logging
import tempfile
from datetime import datetime
from pathlib import Path

from rag.ingestion.base import BaseIngestor
from rag.models import Chunk, Document, Platform
from rag.processing.chunking import MediaChunker

logger = logging.getLogger(__name__)


class YouTubeIngestor(BaseIngestor):
    def __init__(self):
        self.media_chunker = MediaChunker()
        self._last_segments = None
        self._last_transcript_source = None

    def ingest(self, source: str) -> tuple[Document, list[Chunk]]:
        video_id = self._extract_video_id(source)
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # Get video metadata via yt-dlp (title, author, etc.)
        meta = self._get_metadata(video_url)
        title = meta.get("title", f"YouTube: {video_id}")
        author = meta.get("channel") or meta.get("uploader")

        # Try transcript API first (faster, no download needed)
        segments = self._fetch_transcript(video_id)
        transcript_source = "api"

        # Fallback to Whisper if no transcript available
        if not segments:
            logger.info(f"No subtitles for {video_id}, falling back to Whisper transcription")
            segments = self._transcribe_with_whisper(video_url)
            transcript_source = "whisper"

        if not segments:
            raise ValueError(f"Could not get transcript for {video_id} (no subtitles and Whisper failed)")

        # Save raw segments before chunking
        self._last_segments = [
            {"text": s["text"], "start": s["start"], "end": s.get("end")}
            for s in segments
        ]
        self._last_transcript_source = transcript_source

        # Group into ~60 second windows
        grouped = self._group_by_time(segments, window_seconds=60)

        # Parse published_at from upload_date
        published_at = self._parse_upload_date(meta.get("upload_date"))

        doc = Document(
            title=title,
            source_url=video_url,
            platform=Platform.YOUTUBE,
            author=author,
            created_at=published_at,
            metadata={
                "video_id": video_id,
                "duration": meta.get("duration"),
                "channel_id": meta.get("channel_id"),
                "upload_date": meta.get("upload_date"),
            },
        )

        chunks = []
        for idx, group in enumerate(grouped):
            text = " ".join(s["text"] for s in group)
            start_time = group[0]["start"]
            chunks.append(
                Chunk(
                    document_id=doc.id,
                    content=text,
                    chunk_index=idx,
                    token_count=len(text.split()),
                    metadata={
                        "platform": "youtube",
                        "video_id": video_id,
                        "title": title,
                        "start_time": start_time,
                        "source_url": f"{video_url}&t={int(start_time)}",
                    },
                )
            )

        return doc, chunks

    @staticmethod
    def _parse_upload_date(upload_date: str | None) -> datetime | None:
        if not upload_date or len(upload_date) != 8:
            return None
        try:
            return datetime.strptime(upload_date, "%Y%m%d")
        except ValueError:
            return None

    @staticmethod
    def _get_metadata(url: str) -> dict:
        """Get video metadata without downloading."""
        try:
            import yt_dlp
            opts = {"quiet": True, "no_warnings": True, "skip_download": True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    "title": info.get("title"),
                    "channel": info.get("channel"),
                    "uploader": info.get("uploader"),
                    "channel_id": info.get("channel_id"),
                    "duration": info.get("duration"),
                    "upload_date": info.get("upload_date"),
                }
        except Exception as e:
            logger.warning(f"Could not fetch metadata: {e}")
            return {}

    @staticmethod
    def _fetch_transcript(video_id: str) -> list[dict] | None:
        """Try to get transcript via youtube-transcript-api."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.fetch(video_id)
            return [
                {
                    "text": entry.text,
                    "start": entry.start,
                    "end": entry.start + entry.duration,
                    "chapter": "default",
                }
                for entry in transcript_list
            ]
        except Exception as e:
            logger.info(f"No transcript via API for {video_id}: {e}")
            return None

    @staticmethod
    def _transcribe_with_whisper(url: str) -> list[dict] | None:
        """Download audio and transcribe with Whisper.
        Tries remote Mac Studio server first (fast, large-v3-turbo),
        falls back to local faster-whisper on CPU.
        """
        try:
            import yt_dlp
        except ImportError:
            logger.error("yt-dlp not installed")
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = str(Path(tmpdir) / "audio.mp3")
            opts = {
                "quiet": True,
                "no_warnings": True,
                "format": "bestaudio[ext=m4a]/bestaudio/best",
                "outtmpl": audio_path,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "128",
                }],
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])

            # Find the actual output file
            audio_files = list(Path(tmpdir).glob("audio*"))
            if not audio_files:
                logger.error("No audio file after download")
                return None
            audio_file = str(audio_files[0])

            # Try remote Whisper server on Mac Studio first
            segments = YouTubeIngestor._transcribe_remote(audio_file)
            if segments:
                return segments

            # Fallback to local faster-whisper on CPU
            logger.info("Remote Whisper unavailable, falling back to local CPU transcription")
            return YouTubeIngestor._transcribe_local(audio_file)

    @staticmethod
    def _transcribe_remote(audio_file: str) -> list[dict] | None:
        """Send audio to Mac Studio Whisper server."""
        import httpx

        whisper_url = "http://192.168.178.8:8765/v1/audio/transcriptions"
        try:
            logger.info("Sending audio to Mac Studio Whisper server...")
            with open(audio_file, "rb") as f:
                resp = httpx.post(
                    whisper_url,
                    files={"file": ("audio.mp3", f, "audio/mpeg")},
                    data={"model": "mlx-community/whisper-large-v3-turbo", "response_format": "verbose_json"},
                    timeout=600.0,  # 10 min timeout for long videos
                )
            if resp.status_code != 200:
                logger.warning(f"Whisper server returned {resp.status_code}")
                return None

            data = resp.json()
            segments = []
            for seg in data.get("segments", []):
                segments.append({
                    "text": seg["text"].strip(),
                    "start": seg["start"],
                    "end": seg.get("end"),
                    "chapter": "default",
                })

            logger.info(f"Remote Whisper transcription complete: {len(segments)} segments, language: {data.get('language', '?')}")
            return segments if segments else None

        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            logger.info(f"Mac Studio Whisper server not reachable: {e}")
            return None
        except Exception as e:
            logger.warning(f"Remote transcription failed: {e}")
            return None

    @staticmethod
    def _transcribe_local(audio_file: str) -> list[dict] | None:
        """Fallback: transcribe locally with faster-whisper on CPU."""
        try:
            from faster_whisper import WhisperModel

            logger.info("Transcribing locally with faster-whisper (CPU, this may take a while)...")
            model = WhisperModel("small", device="cpu", compute_type="int8")
            segments_iter, info = model.transcribe(audio_file, beam_size=5)

            logger.info(f"Detected language: {info.language} (probability {info.language_probability:.2f})")

            segments = []
            for segment in segments_iter:
                segments.append({
                    "text": segment.text.strip(),
                    "start": segment.start,
                    "end": segment.end,
                    "chapter": "default",
                })

            logger.info(f"Local Whisper transcription complete: {len(segments)} segments")
            return segments if segments else None

        except ImportError:
            logger.error("faster-whisper not installed")
            return None
        except Exception as e:
            logger.error(f"Local Whisper transcription failed: {e}")
            return None

    @staticmethod
    def _extract_video_id(url: str) -> str:
        patterns = [
            r"(?:v=|\/v\/|youtu\.be\/)([a-zA-Z0-9_-]{11})",
            r"^([a-zA-Z0-9_-]{11})$",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        raise ValueError(f"Cannot extract video ID from: {url}")

    @staticmethod
    def _group_by_time(
        segments: list[dict], window_seconds: float = 60.0
    ) -> list[list[dict]]:
        if not segments:
            return []
        groups = []
        current_group = [segments[0]]
        group_start = segments[0]["start"]

        for seg in segments[1:]:
            if seg["start"] - group_start >= window_seconds:
                groups.append(current_group)
                current_group = [seg]
                group_start = seg["start"]
            else:
                current_group.append(seg)

        if current_group:
            groups.append(current_group)
        return groups
