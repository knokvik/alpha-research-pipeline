"""Walk-forward and purged/embargoed validation splitters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from alpha_pipeline.io import write_json


@dataclass(frozen=True)
class Fold:
    """Metadata for one out-of-sample validation fold."""

    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train_rows: int
    n_test_rows: int
    purged_rows: int
    embargoed_rows: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PurgedWalkForwardSplitter:
    """Rolling walk-forward splitter with label-window purging and embargo support."""

    train_window_days: int = 504
    test_window_days: int = 63
    step_days: int = 63
    embargo_days: int = 5
    allow_future_training: bool = False

    def split(self, dataset: pd.DataFrame) -> Iterator[tuple[Fold, pd.DataFrame, pd.DataFrame]]:
        frame = _validate_dataset(dataset)
        dates = pd.Series(sorted(frame["date"].unique()))
        if len(dates) < self.train_window_days + self.test_window_days:
            raise ValueError("Not enough dates for the requested walk-forward split.")

        fold_id = 0
        for test_start_idx in range(
            self.train_window_days,
            len(dates) - self.test_window_days + 1,
            self.step_days,
        ):
            train_start = dates.iloc[test_start_idx - self.train_window_days]
            train_end = dates.iloc[test_start_idx - 1]
            test_start = dates.iloc[test_start_idx]
            test_end = dates.iloc[test_start_idx + self.test_window_days - 1]
            embargo_end = test_end + pd.tseries.offsets.BDay(self.embargo_days)

            test_mask = frame["date"].between(test_start, test_end)
            if self.allow_future_training:
                train_candidate_mask = ~test_mask
            else:
                train_candidate_mask = frame["date"].between(train_start, train_end)

            overlap_mask = (frame["label_start_date"] <= test_end) & (frame["label_end_date"] >= test_start)
            purged_mask = train_candidate_mask & overlap_mask
            embargo_mask = train_candidate_mask & frame["date"].gt(test_end) & frame["date"].le(embargo_end)
            train_mask = train_candidate_mask & ~purged_mask & ~embargo_mask

            train = frame.loc[train_mask].copy()
            test = frame.loc[test_mask].copy()
            if train.empty or test.empty:
                continue

            fold = Fold(
                fold_id=fold_id,
                train_start=str(train["date"].min().date()),
                train_end=str(train["date"].max().date()),
                test_start=str(test_start.date()),
                test_end=str(test_end.date()),
                n_train_rows=int(len(train)),
                n_test_rows=int(len(test)),
                purged_rows=int(purged_mask.sum()),
                embargoed_rows=int(embargo_mask.sum()),
            )
            fold_id += 1
            yield fold, train.reset_index(drop=True), test.reset_index(drop=True)


def write_fold_manifest(path: str | Path, folds: list[Fold]) -> None:
    """Persist fold metadata for auditability."""

    write_json(path, {"folds": [fold.to_dict() for fold in folds]})


def _validate_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "asset", "label_start_date", "label_end_date", "forward_return"}
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"Validation dataset is missing required columns: {sorted(missing)}")
    frame = dataset.copy()
    for column in ("date", "label_start_date", "label_end_date"):
        frame[column] = pd.to_datetime(frame[column])
    if (frame["label_start_date"] <= frame["date"]).any():
        raise ValueError("label_start_date must be after date.")
    if (frame["label_end_date"] < frame["label_start_date"]).any():
        raise ValueError("label_end_date must be on or after label_start_date.")
    return frame.sort_values(["date", "asset"]).reset_index(drop=True)
