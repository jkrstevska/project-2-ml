"""Download the raw data for Project 2 into data/raw/.

Usage
-----
    python download_data.py houses     # King County house sales (regression)
    python download_data.py telco      # Telco customer churn (classification)
    python download_data.py bank       # Bank Marketing (imbalanced classification)
    python download_data.py flowers    # Flower photos (image option, 218 MB)
    python download_data.py tabular    # the three tabular datasets

Nothing in data/ is committed, so this script is how anyone reproduces your
project: they clone the repo, run it, and have the same files you had.

See data/README.md for what each dataset contains and how it is licensed.
"""

import argparse
import io
import tarfile
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"

HOUSES_URL = "https://www.openml.org/data/get_csv/21578898/house_sales.arff"
TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
BANK_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
FLOWERS_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "example_images/flower_photos.tgz"
)

# UCI rejects the default urllib user agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (Ironhack DSML Project 2)"}


def read_url(url: str) -> bytes:
    """Return the body of `url`, printing its size."""
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request) as response:
        payload = response.read()
    print(f" {len(payload) / 1e6:.1f} MB")
    return payload


def fetch(url: str, dest: Path) -> None:
    """Download `url` to `dest`, skipping the download if it is already there."""
    if dest.exists():
        print(f"  {dest.name} already downloaded, skipping")
        return
    print(f"  downloading {dest.name} ...", end="", flush=True)
    dest.write_bytes(read_url(url))


def download_houses() -> None:
    """King County house sales, May 2014 to May 2015, from OpenML."""
    print("King County house sales")
    fetch(HOUSES_URL, RAW / "kc_house_sales.csv")


def download_telco() -> None:
    """IBM's Telco customer churn sample."""
    print("Telco customer churn")
    fetch(TELCO_URL, RAW / "telco_churn.csv")


def download_bank() -> None:
    """Bank Marketing (UCI): the full 'additional' file plus its documentation."""
    print("Bank Marketing (UCI)")
    wanted = {
        "bank-additional/bank-additional-full.csv": "bank_marketing.csv",
        "bank-additional/bank-additional-names.txt": "bank_marketing_names.txt",
    }
    if all((RAW / name).exists() for name in wanted.values()):
        print("  bank_marketing.csv already downloaded, skipping")
        return
    print("  downloading bank+marketing.zip ...", end="", flush=True)
    outer = zipfile.ZipFile(io.BytesIO(read_url(BANK_URL)))
    # The UCI zip holds two more zips; the 'additional' one has the full file.
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip")))
    for member, name in wanted.items():
        (RAW / name).write_bytes(inner.read(member))
        print(f"  wrote {name}")


def download_flowers() -> None:
    """3,670 flower photos in five folders, one per class."""
    print("Flower photos (image option)")
    folder = RAW / "flower_photos"
    if folder.exists():
        print("  flower_photos/ already downloaded, skipping")
        return
    print("  downloading flower_photos.tgz (218 MB, a minute or two) ...", end="", flush=True)
    archive = tarfile.open(fileobj=io.BytesIO(read_url(FLOWERS_URL)), mode="r:gz")
    try:
        archive.extractall(RAW, filter="data")
    except TypeError:  # Python older than 3.12 has no `filter` argument
        archive.extractall(RAW)
    counts = {d.name: len(list(d.glob("*.jpg"))) for d in sorted(folder.iterdir()) if d.is_dir()}
    print(f"  extracted {sum(counts.values()):,} photos: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=["houses", "telco", "bank", "flowers", "tabular"])
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("houses", "tabular"):
        download_houses()
    if args.dataset in ("telco", "tabular"):
        download_telco()
    if args.dataset in ("bank", "tabular"):
        download_bank()
    if args.dataset == "flowers":
        download_flowers()

    print(f"\nDone. Files are in {RAW.resolve().relative_to(Path.cwd().resolve()) if RAW.resolve().is_relative_to(Path.cwd().resolve()) else RAW}/")


if __name__ == "__main__":
    main()
