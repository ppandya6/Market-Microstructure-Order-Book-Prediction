"""Download the public LOBSTER sample without disabling TLS or extracting unsafe ZIP paths."""

import argparse
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://lobsterdata.com/info/sample/LOBSTER_SampleFile_AAPL_2012-06-21_10.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/aapl-2012-06-21"))
    parser.add_argument("--sha256", help="Optional independently obtained expected archive SHA-256")
    args = parser.parse_args()
    with urlopen(URL, timeout=60) as response:
        raw = response.read(100_000_001)
    if len(raw) > 100_000_000:
        raise ValueError("Archive exceeds 100 MB limit")
    digest = hashlib.sha256(raw).hexdigest()
    if args.sha256 and digest.lower() != args.sha256.lower():
        raise ValueError("Archive checksum mismatch")
    with ZipFile(io.BytesIO(raw)) as archive:
        files = [
            f
            for f in archive.infolist()
            if f.filename.endswith(".csv") and ("_message_" in f.filename or "_orderbook_" in f.filename)
        ]
        if len(files) != 2 or sum(f.file_size for f in files) > 600_000_000:
            raise ValueError("Unexpected archive contents or expanded size")
        args.output.mkdir(parents=True, exist_ok=True)
        for info in files:
            destination = args.output / Path(info.filename).name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
        for info in files:
            with archive.open(info) as src:
                (args.output / Path(info.filename).name).write_bytes(src.read())
    print(f"Saved public sample in {args.output}; archive SHA-256: {digest}")
    print("Source: " + URL)


if __name__ == "__main__":
    main()
