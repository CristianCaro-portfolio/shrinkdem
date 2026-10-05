"""Download and unpack Google's public surface-code data (Zenodo 6804040, 315 MB)."""
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/api/records/6804040/files/google_qec3v5_experiment_data.zip/content"


def main(target: str = "data") -> None:
    target = Path(target)
    raw = target / "raw"
    if (raw / "README.txt").exists():
        print(f"data already present in {raw}")
        return
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "google_qec3v5.zip"
    if not archive.exists():
        print("downloading 315 MB from Zenodo ...")
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(raw)
    print(f"unpacked into {raw}")


if __name__ == "__main__":
    main(*sys.argv[1:])
