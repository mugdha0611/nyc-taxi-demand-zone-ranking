from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--months",
        nargs="+",
        default=["2025-01", "2025-02", "2025-03"],
        help="Months in YYYY-MM format, e.g. 2025-10 2025-11 2025-12",
    )
    args = parser.parse_args()

    for month in args.months:
        target = RAW / f"yellow_tripdata_{month}.parquet"
        if target.exists():
            print(f"Exists: {target}")
            continue
        url = BASE.format(month=month)
        print(f"Downloading {month} ...")
        urlretrieve(url, target)
        print(f"Saved {target}")

    zone_target = RAW / "taxi_zone_lookup.csv"
    if not zone_target.exists():
        print("Downloading taxi-zone lookup ...")
        urlretrieve(ZONE_URL, zone_target)
        print(f"Saved {zone_target}")


if __name__ == "__main__":
    main()
