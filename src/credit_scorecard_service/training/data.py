"""Download, load and split the Lending Club granting dataset."""

from __future__ import annotations

import argparse
import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from credit_scorecard_service.config import DATA_SOURCE, TIME_SPLIT, DataSource, TimeSplit
from credit_scorecard_service.features import DATE_COLUMN, ID_COLUMN, RAW_FEATURES, TARGET

log = logging.getLogger(__name__)


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(source: DataSource = DATA_SOURCE, force: bool = False) -> Path:
    """Fetch the dataset if missing and verify its checksum."""
    path = source.local_path
    if path.exists() and not force:
        log.info("Dataset already present at %s", path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        log.info("Downloading %s", source.url)
        urllib.request.urlretrieve(source.url, tmp)
        tmp.rename(path)

    actual = md5sum(path)
    if actual != source.md5:
        raise ValueError(f"Checksum mismatch for {path}: expected {source.md5}, got {actual}")
    log.info("Checksum OK (%s)", actual)
    return path


def load(path: Path | None = None) -> pd.DataFrame:
    """Load the columns the model may use, with issue date parsed to a month."""
    path = path or DATA_SOURCE.local_path
    df = pd.read_csv(path, usecols=[ID_COLUMN, DATE_COLUMN, *RAW_FEATURES, TARGET])
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format="%b-%Y").dt.to_period("M")
    return df.sort_values([DATE_COLUMN, ID_COLUMN], ignore_index=True)


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                "rows": len(part),
                "default_rate": round(float(part[TARGET].mean()), 4),
                "from": str(part[DATE_COLUMN].min()),
                "to": str(part[DATE_COLUMN].max()),
            }
            for name, part in (("train", self.train), ("valid", self.valid), ("test", self.test))
        }


def time_split(df: pd.DataFrame, split: TimeSplit = TIME_SPLIT) -> Splits:
    month = df[DATE_COLUMN]
    p = pd.Period
    train = df[month <= p(split.train_end, "M")]
    valid = df[(month >= p(split.valid_start, "M")) & (month <= p(split.valid_end, "M"))]
    test = df[(month >= p(split.test_start, "M")) & (month <= p(split.test_end, "M"))]
    return Splits(
        train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Lending Club granting dataset.")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download(force=args.force)


if __name__ == "__main__":
    main()
