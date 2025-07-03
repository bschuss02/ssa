import re
from typing import Dict, List, Tuple, Optional


class CHATParser:
    def __init__(self):
        self.headers = {}
        self.participants = {}
        self.utterances = []

    def parse_file(self, file_path: str):
        """Parse a CHAT format (.cha) file"""
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

    def _parse_header(self, line: str):
        """Parse header lines starting with @"""
        if ":" in line:
            key, value = line[1:].split(":", 1)
            self.headers[key.strip()] = value.strip()

            # Special handling for participant info
            if key.strip() == "Participants":
                self._parse_participants(value.strip())

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

        # Clean text (remove timestamps)
        clean_text = self._clean_text(text)

        # Count disfluencies
        disfluencies = self._count_disfluencies(text)

        utterance = {
            "speaker": speaker,
            "speaker_description": self.participants.get(speaker, ""),
            "text": clean_text,
            "original_text": text,
            "timestamps": timestamps,
            "disfluencies": disfluencies,
        }

        self.utterances.append(utterance)

    def _extract_timestamps(self, text: str) -> List[Tuple[int, int]]:
        """Extract timestamp ranges from text"""
        timestamp_pattern = r"(\d+)_(\d+)"
        matches = re.findall(timestamp_pattern, text)
        return [(int(start), int(end)) for start, end in matches]

    def _clean_text(self, text: str) -> str:
        """Remove timestamps and special annotations from text"""
        # Remove timestamps
        text = re.sub(r"\d+_\d+", "", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _count_disfluencies(self, text: str) -> Dict[str, int]:
        """Count various types of disfluencies"""
        disfluencies = {
            "filled_pauses": len(re.findall(r"&-\w+", text)),  # &-um, &-uh, etc.
            "pauses": text.count(" . "),  # Unfilled pauses
            "unintelligible": text.count("xxx"),  # Unintelligible speech
            "repetitions": 0,  # Would need more complex analysis
            "blocks": 0,  # Would need more complex analysis
        }
        return disfluencies

    def to_dataframe(self):
        """Convert to pandas DataFrame for analysis"""
        try:
            import pandas as pd

            df = pd.DataFrame(self.utterances)
            return df
        except ImportError:
            print("pandas not installed. Install with: pip install pandas")
            return None

    def export_to_csv(self, filename: str):
        """Export utterances to CSV"""
        df = self.to_dataframe()
        if df is not None:
            df.to_csv(filename, index=False)
            print(f"Data exported to {filename}")

    def get_speaker_stats(self) -> Dict:
        """Get statistics for each speaker"""
        stats = {}
        for speaker in set(utt["speaker"] for utt in self.utterances):
            speaker_utts = [utt for utt in self.utterances if utt["speaker"] == speaker]

            total_disfluencies = {}
            for utt in speaker_utts:
                for disf_type, count in utt["disfluencies"].items():
                    total_disfluencies[disf_type] = total_disfluencies.get(disf_type, 0) + count

            stats[speaker] = {
                "utterance_count": len(speaker_utts),
                "total_words": sum(len(utt["text"].split()) for utt in speaker_utts),
                "disfluencies": total_disfluencies,
            }

        return stats


# Example usage
if __name__ == "__main__":
    # Parse the file
    parser = CHATParser()
    parser.parse_file("your_file.cha")  # Replace with your file path

    # Print basic info
    print("Headers:", parser.headers)
    print("Participants:", parser.participants)
    print(f"Total utterances: {len(parser.utterances)}")

    # Get speaker statistics
    stats = parser.get_speaker_stats()
    for speaker, speaker_stats in stats.items():
        print(f"\n{speaker} ({parser.participants.get(speaker, '')}):")
        print(f"  Utterances: {speaker_stats['utterance_count']}")
        print(f"  Total words: {speaker_stats['total_words']}")
        print(f"  Disfluencies: {speaker_stats['disfluencies']}")

    # Export to CSV
    parser.export_to_csv("fluency_data.csv")

    # Show first few utterances
    print(f"\nFirst 3 utterances:")
    for i, utt in enumerate(parser.utterances[:3]):
        print(f"{i+1}. {utt['speaker']}: {utt['text']}")
        if utt["disfluencies"]["filled_pauses"] > 0:
            print(f"   -> {utt['disfluencies']['filled_pauses']} filled pauses")
