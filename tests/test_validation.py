from __future__ import annotations

import pandas as pd

from alpha_pipeline.validation import PurgedWalkForwardSplitter


def _validation_frame(n_days: int = 16) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for asset in ["A", "B"]:
        for idx, date in enumerate(dates[:-3]):
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "factor": float(idx),
                    "forward_return": 0.01,
                    "label_start_date": dates[idx + 1],
                    "label_end_date": dates[idx + 3],
                }
            )
    return pd.DataFrame(rows)


def test_purged_splitter_removes_overlapping_training_labels() -> None:
    dataset = _validation_frame()
    splitter = PurgedWalkForwardSplitter(train_window_days=6, test_window_days=3, step_days=3, embargo_days=1)
    fold, train, test = next(splitter.split(dataset))

    assert fold.purged_rows > 0
    assert not ((train["label_start_date"] <= test["date"].max()) & (train["label_end_date"] >= test["date"].min())).any()


def test_embargo_excludes_post_test_buffer_when_future_training_is_allowed() -> None:
    dataset = _validation_frame(n_days=18)
    splitter = PurgedWalkForwardSplitter(
        train_window_days=5,
        test_window_days=2,
        step_days=2,
        embargo_days=2,
        allow_future_training=True,
    )
    fold, train, _ = next(splitter.split(dataset))
    test_end = pd.Timestamp(fold.test_end)
    embargo_end = test_end + pd.tseries.offsets.BDay(2)

    assert fold.embargoed_rows > 0
    assert not train["date"].between(test_end + pd.tseries.offsets.BDay(1), embargo_end).any()
