"""Download a random subset of the MAESTRO v3.0.0 MIDI files (CC BY-NC-SA 4.0) into data/midi_dataset/.

    python scripts/download_dataset.py --count 150

Dataset: https://magenta.tensorflow.org/datasets/maestro  (The MAESTRO Dataset, Hawthorne et al., ICLR 2019)
If the URL ever changes, download the MIDI-only zip from that page and extract any .mid files into data/midi_dataset/.
"""
import argparse
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "midi_dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=150, help="number of MIDI files to keep")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    zip_path = ROOT / "data" / "maestro-midi.zip"
    if not zip_path.exists():
        print("Downloading MAESTRO MIDI zip (~60 MB)...")
        try:
            urllib.request.urlretrieve(URL, zip_path)
        except Exception as exc:
            sys.exit(f"Download failed: {exc}\nDownload it manually from the MAESTRO page and extract .mid files to {DEST}")
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith((".mid", ".midi")))
        pick = random.Random(args.seed).sample(names, min(args.count, len(names)))
        for n in pick:
            (DEST / Path(n).name).write_bytes(z.read(n))
    print(f"Extracted {len(pick)} files to data/midi_dataset/.\n"
          "License: CC BY-NC-SA 4.0 - credit the MAESTRO dataset (Google Magenta) and keep it non-commercial.")


if __name__ == "__main__":
    main()
