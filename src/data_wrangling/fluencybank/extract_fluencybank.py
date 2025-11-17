#!/usr/bin/env python3
"""
Fluency Bank Data Extraction Script

This script processes Fluency Bank dataset files:
- Video files (.mp4, .mov, .avi, etc.) from data/fluencybank/raw/video/
- .cha transcript files from data/fluencybank/raw/chat/

It extracts audio segments corresponding to each utterance in the .cha files
and creates a parquet file with annotations and audio file paths.
"""

import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import polars as pl
import librosa
import soundfile as sf
from tqdm import tqdm
import subprocess
import argparse


@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    annotated_text: str
    unannotated_text: str
    speaker_id: str  # Actual speaker name/ID (e.g., from filename)
    speaker_role: str  # Original speaker code (e.g., "INV", "PAR")
    clip_id: str
    clip_audio_file: str


class FluencyBankCHATParser:
    """Enhanced CHAT parser specifically for Fluency Bank extraction"""

    def __init__(self):
        self.headers = {}
        self.participants = {}
        self.utterances = []
        self.media_name = None

    def parse_file(self, file_path: str):
        """Parse a CHAT format (.cha) file"""
        self._reset()

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        in_transcript = False
        current_utterance = ""
        current_speaker = ""

        for line in lines:
            line = line.strip()

            if line.startswith("@End"):
                break
            elif line.startswith("@Begin"):
                in_transcript = True
                continue
            elif line.startswith("@") and not in_transcript:
                self._parse_header(line)
            elif line.startswith("*") and in_transcript:
                # Save previous utterance if exists
                if current_utterance and current_speaker:
                    self._add_utterance(current_speaker, current_utterance)

                # Start new utterance
                speaker, text = self._parse_utterance_line(line)
                current_speaker = speaker
                current_utterance = text
            elif in_transcript and not line.startswith("@"):
                # Continuation of previous utterance
                current_utterance += " " + line

        # Don't forget the last utterance
        if current_utterance and current_speaker:
            self._add_utterance(current_speaker, current_utterance)

    def _reset(self):
        """Reset parser state"""
        self.headers = {}
        self.participants = {}
        self.utterances = []
        self.media_name = None

    def _parse_header(self, line: str):
        """Parse header lines starting with @"""
        if ":" in line:
            key, value = line[1:].split(":", 1)
            key = key.strip()
            value = value.strip()
            self.headers[key] = value

            # Special handling for participant info
            if key == "Participants":
                self._parse_participants(value)
            elif key == "Media":
                # Extract media name (first part before comma)
                self.media_name = value.split(",")[0].strip()

    def _parse_participants(self, participant_line: str):
        """Parse the @Participants line to extract speaker info"""
        # Example: INV Investigator, PAR Participant
        parts = participant_line.split(",")
        for part in parts:
            part = part.strip()
            if " " in part:
                code, description = part.split(" ", 1)
                self.participants[code] = description

    def _parse_utterance_line(self, line: str) -> Tuple[str, str]:
        """Parse lines starting with * (utterances)"""
        # Extract speaker code
        speaker_match = re.match(r"\*([^:]+):", line)
        if speaker_match:
            speaker = speaker_match.group(1)
            text = line[len(speaker_match.group(0)) :].strip()
            return speaker, text
        return "", line

    def _add_utterance(self, speaker: str, text: str):
        """Add an utterance to the list with additional processing"""
        # Extract timestamps if present
        timestamps = self._extract_timestamps(text)

        # Only add utterances that have timestamps
        if timestamps:
            # Clean text (remove timestamps)
            clean_text = self._clean_text(text)

            utterance = {
                "speaker": speaker,
                "speaker_description": self.participants.get(speaker, ""),
                "text": clean_text,
                "original_text": text,
                "timestamps": timestamps,
            }

            self.utterances.append(utterance)

    def _extract_timestamps(self, text: str) -> List[Tuple[int, int]]:
        """Extract timestamp ranges from text (in milliseconds)"""
        timestamp_pattern = r"(\d+)_(\d+)"
        matches = re.findall(timestamp_pattern, text)
        return [(int(start), int(end)) for start, end in matches]

    def _clean_text(self, text: str) -> str:
        """Remove timestamps and clean text while preserving annotations"""
        # Remove timestamps
        text = re.sub(r"\d+_\d+", "", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove Unicode control characters and other unwanted characters
        text = re.sub(r"[\u0000-\u001F\u007F-\u009F]", "", text)
        return text

    def _remove_annotations(self, text: str) -> str:
        """Remove all annotations to get clean text"""
        # Remove filled pauses (&-um, &-uh, etc.)
        text = re.sub(r"&-\w+", "", text)
        # Remove unintelligible speech markers
        text = re.sub(r"\bxxx\b", "", text)
        # Remove pause markers
        text = re.sub(r"\s+\.\s+", " ", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove Unicode control characters and other unwanted characters
        text = re.sub(r"[\u0000-\u001F\u007F-\u009F]", "", text)
        return text

    def get_audio_segments(self, base_clip_id: str) -> List[AudioSegment]:
        """Convert utterances to AudioSegment objects"""
        segments = []

        # Extract speaker name from filename (e.g., "participant_001" from "participant_001_session")
        # This assumes the filename contains the speaker/participant identifier
        speaker_name = self._extract_speaker_name_from_clip_id(base_clip_id)

        for i, utterance in enumerate(self.utterances):
            timestamps = utterance["timestamps"]

            # Handle multiple timestamp ranges in one utterance
            for j, (start_ms, end_ms) in enumerate(timestamps):
                # Convert milliseconds to seconds
                start_time = start_ms / 1000.0
                end_time = end_ms / 1000.0

                # Skip segments that are too short (less than 0.1 seconds)
                if end_time - start_time < 0.1:
                    continue

                # Create unique clip ID
                segment_id = f"{base_clip_id}_{i:03d}_{j:03d}"
                clip_audio_file = f"{segment_id}.wav"

                # Get annotated and unannotated text
                annotated_text = utterance["text"]
                unannotated_text = self._remove_annotations(annotated_text)

                # Get speaker code as the speaker role
                speaker_role = utterance["speaker"]

                segment = AudioSegment(
                    start_time=start_time,
                    end_time=end_time,
                    annotated_text=annotated_text,
                    unannotated_text=unannotated_text,
                    speaker_id=speaker_name,
                    speaker_role=speaker_role,
                    clip_id=segment_id,
                    clip_audio_file=clip_audio_file,
                )

                segments.append(segment)

        return segments

    def _extract_speaker_name_from_clip_id(self, clip_id: str) -> str:
        """Extract speaker name/identifier from the clip ID (filename)"""
        # This method extracts the participant/speaker identifier from the filename
        # You may need to adjust this logic based on your filename format
        # Common patterns might be:
        # - "participant_001_session" -> "participant_001"
        # - "john_doe_interview" -> "john_doe"
        # - "P001_recording" -> "P001"

        # Remove common suffixes and extract the main identifier
        clip_id_lower = clip_id.lower()

        # Remove common suffixes
        suffixes_to_remove = ["_session", "_recording", "_interview", "_audio", "_video", "_chat"]
        for suffix in suffixes_to_remove:
            if clip_id_lower.endswith(suffix):
                clip_id = clip_id[: len(clip_id) - len(suffix)]
                break

        # If the filename has multiple parts separated by underscores,
        # assume the speaker name is the first part(s) before session info
        parts = clip_id.split("_")
        if len(parts) >= 2:
            # Try to identify if the last part looks like a session number
            if parts[-1].isdigit() or parts[-1].startswith("session"):
                return "_".join(parts[:-1])

        # If no clear pattern, return the entire clip_id as the speaker name
        return clip_id


class FluencyBankExtractor:
    """Main class for extracting Fluency Bank dataset"""

    def __init__(self, video_dir: str, cha_dir: str, output_dir: str, skip_existing: bool = False):
        self.video_dir = Path(video_dir)
        self.cha_dir = Path(cha_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.wav_output_dir = self.output_dir / "wav_clips"
        self.wav_output_dir.mkdir(parents=True, exist_ok=True)

        self.parser = FluencyBankCHATParser()
        self.skip_existing = skip_existing

    def find_file_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching .cha and video file pairs"""
        pairs = []

        # Get all .cha files
        cha_files = list(self.cha_dir.glob("*.cha"))

        # Common video extensions
        video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm"}

        for cha_file in cha_files:
            base_name = cha_file.stem  # filename without extension

            # Look for matching video file
            video_file = None
            for ext in video_extensions:
                potential_video = self.video_dir / f"{base_name}{ext}"
                if potential_video.exists():
                    video_file = potential_video
                    break

            if video_file:
                pairs.append((cha_file, video_file))
            else:
                print(f"Warning: No matching video file found for {cha_file.name}")

        return pairs

    def extract_audio_segment(self, video_path: Path, start_time: float, end_time: float, output_path: Path):
        """Extract audio segment from video file using ffmpeg"""
        try:
            # Create temporary file for raw extraction
            temp_path = output_path.with_suffix(".temp.wav")

            # Use ffmpeg to extract audio segment
            cmd = [
                "ffmpeg",
                "-i",
                str(video_path),  # Input video
                "-ss",
                str(start_time),  # Start time in seconds
                "-t",
                str(end_time - start_time),  # Duration
                "-ac",
                "1",  # Mono audio
                "-ar",
                "44100",  # Sample rate 44.1kHz
                "-y",  # Overwrite output file
                str(temp_path),  # Output file
            ]

            # Run ffmpeg command with suppressed output
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Resample to 16kHz using librosa for consistency
            audio, sr = librosa.load(str(temp_path), sr=16000)

            # Save final audio file at 16kHz
            sf.write(str(output_path), audio, 16000)

            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error extracting audio segment {output_path}: {e.stderr}")
            return False
        except Exception as e:
            print(f"Error extracting audio segment {output_path}: {e}")
            return False

        return True

    def process_file_pair(self, cha_file: Path, video_file: Path) -> List[AudioSegment]:
        """Process a single .cha and video file pair"""
        print(f"Processing: {cha_file.name} + {video_file.name}")

        # Parse .cha file
        self.parser.parse_file(str(cha_file))

        # Get base clip ID from filename
        base_clip_id = cha_file.stem

        # Get audio segments
        segments = self.parser.get_audio_segments(base_clip_id)

        print(f"Found {len(segments)} audio segments")

        # Extract audio clips
        segments_to_remove = []
        skipped_count = 0
        extracted_count = 0

        for segment in tqdm(segments, desc="Processing audio", leave=False):
            output_path = self.wav_output_dir / segment.clip_audio_file

            if self.skip_existing and output_path.exists():
                # Skip extraction but keep the segment in the list
                skipped_count += 1
                continue

            success = self.extract_audio_segment(
                video_file, segment.start_time, segment.end_time, output_path
            )

            if success:
                extracted_count += 1
            else:
                # Mark failed segment for removal
                segments_to_remove.append(segment)

        # Remove failed segments
        for segment in segments_to_remove:
            segments.remove(segment)

        # Print summary for this file pair
        if self.skip_existing and skipped_count > 0:
            print(
                f"  Skipped {skipped_count} existing files, extracted {extracted_count} new files, {len(segments_to_remove)} failed"
            )
        else:
            print(f"  Extracted {extracted_count} files, {len(segments_to_remove)} failed")

        return segments

    def extract_all(self) -> str:
        """Extract all file pairs and create parquet file"""
        print("Finding .cha and video file pairs...")
        file_pairs = self.find_file_pairs()

        if not file_pairs:
            print("No matching file pairs found!")
            return None

        print(f"Found {len(file_pairs)} file pairs")

        all_segments = []

        # Process each file pair
        for cha_file, video_file in tqdm(file_pairs, desc="Processing files"):
            segments = self.process_file_pair(cha_file, video_file)
            all_segments.extend(segments)

        if not all_segments:
            print("No audio segments extracted!")
            return None

        print(f"Total segments extracted: {len(all_segments)}")

        # Create polars dataframe
        df_data = []
        for segment in all_segments:
            df_data.append(
                {
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "annotated_text": segment.annotated_text,
                    "unannotated_text": segment.unannotated_text,
                    "speaker_id": segment.speaker_id,
                    "speaker_role": segment.speaker_role,
                    "clip_id": segment.clip_id,
                    "clip_audio_file": str(self.wav_output_dir / segment.clip_audio_file),
                }
            )

        df = pl.DataFrame(df_data)

        # Save as parquet
        parquet_path = self.output_dir / "fluencybank_segments.parquet"
        df.write_parquet(parquet_path)

        print(f"Saved parquet file: {parquet_path}")
        print(f"Saved {len(all_segments)} wav files to: {self.wav_output_dir}")

        # Print summary statistics
        print("\nSummary:")
        print(f"- Total segments: {len(all_segments)}")
        print(f"- Total duration: {df['end_time'].sum() - df['start_time'].sum():.2f} seconds")
        print(f"- Speakers: {df['speaker_id'].n_unique()}")
        print(f"- Average segment length: {(df['end_time'] - df['start_time']).mean():.2f} seconds")

        return str(parquet_path)


def main():
    """Main function"""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Extract Fluency Bank dataset from video and .cha files")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip extracting audio segments that already exist"
    )
    parser.add_argument(
        "--video-dir", type=str, help="Directory containing video files (default: data/fluencybank/raw/video)"
    )
    parser.add_argument(
        "--cha-dir",
        type=str,
        help="Directory containing .cha transcript files (default: data/fluencybank/raw/chat)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for processed files (default: data/fluencybank/processed)",
    )

    args = parser.parse_args()

    # Set up paths
    base_dir = Path("/Users/Benjamin/dev/ssa")
    video_dir = Path(args.video_dir) if args.video_dir else base_dir / "data/fluencybank/raw/video"
    cha_dir = Path(args.cha_dir) if args.cha_dir else base_dir / "data/fluencybank/raw/chat"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "data/fluencybank/processed"

    # Check if input directories exist
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        sys.exit(1)

    if not cha_dir.exists():
        print(f"Error: Chat directory not found: {cha_dir}")
        sys.exit(1)

    # Create extractor and run
    extractor = FluencyBankExtractor(
        video_dir=str(video_dir),
        cha_dir=str(cha_dir),
        output_dir=str(output_dir),
        skip_existing=args.skip_existing,
    )

    if args.skip_existing:
        print("Note: Skipping extraction for existing audio files")

    parquet_file = extractor.extract_all()

    if parquet_file:
        print(f"\nExtraction completed successfully!")
        print(f"Output parquet file: {parquet_file}")
    else:
        print("Extraction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
