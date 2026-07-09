"""Data interfaces and reproducible demo data generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import pandas as pd


REQUIRED_PRICE_COLUMNS = ("date", "asset", "open", "high", "low", "close", "volume")


class PriceLoader(Protocol):
    """Interface for long-form price data sources."""

    def load_prices(self) -> pd.DataFrame:
        """Return prices with date, asset, OHLCV columns."""


@dataclass(frozen=True)
class Universe:
    """A named collection of tradable assets."""

    name: str
    assets: tuple[str, ...]
    survivorship_bias_free: bool = False
    notes: str = ""


@dataclass(frozen=True)
class DataQualityReport:
    """Audit information to include in dashboards and memos."""

    start_date: str
    end_date: str
    n_assets: int
    n_rows: int
    missing_price_rows: int
    missing_volume_rows: int
    stale_price_rows: int
    survivorship_bias_free: bool
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SyntheticPriceLoader:
    """Generate deterministic long-form OHLCV data with a weak latent signal."""

    n_assets: int = 40
    n_days: int = 900
    start_date: str = "2018-01-01"
    seed: int = 7

    def load_prices(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        dates = pd.bdate_range(self.start_date, periods=self.n_days)
        assets = [f"EQ{i:03d}" for i in range(self.n_assets)]

        market_returns = rng.normal(0.0002, 0.008, size=len(dates))
        sector_count = max(4, min(10, self.n_assets // 5))
        sector_returns = rng.normal(0.0001, 0.006, size=(len(dates), sector_count))

        rows: list[pd.DataFrame] = []
        for asset_idx, asset in enumerate(assets):
            sector = asset_idx % sector_count
            quality = rng.normal()
            value = rng.normal()
            size = rng.normal()
            idiosyncratic = rng.normal(0.0, 0.014 + 0.002 * abs(size), size=len(dates))

            returns = (
                0.55 * market_returns
                + 0.25 * sector_returns[:, sector]
                + 0.00010 * quality
                - 0.00005 * size
                + idiosyncratic
            )
            returns = np.clip(returns, -0.18, 0.18)
            close = 40.0 * np.exp(np.cumsum(returns))
            overnight = rng.normal(0.0, 0.002, size=len(dates))
            open_price = close * (1.0 + overnight)
            high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.0005, 0.018, len(dates)))
            low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.0005, 0.018, len(dates)))
            base_volume = np.exp(13.0 - 0.25 * size + 0.08 * value)
            volume = base_volume * rng.lognormal(0.0, 0.35, size=len(dates))

            rows.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "asset": asset,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume.astype(int),
                        "as_of_date": dates,
                        "sector": f"S{sector}",
                    }
                )
            )

        return validate_prices(pd.concat(rows, ignore_index=True))


@dataclass(frozen=True)
class CsvPriceLoader:
    """Load prices from a long-form CSV or Parquet file."""

    path: str | Path

    def load_prices(self) -> pd.DataFrame:
        source = Path(self.path)
        if source.suffix.lower() == ".parquet":
            frame = pd.read_parquet(source)
        else:
            frame = pd.read_csv(source)
        return validate_prices(frame)


def load_universe(assets: Iterable[str], name: str = "custom", notes: str = "") -> Universe:
    clean_assets = tuple(dict.fromkeys(str(asset) for asset in assets))
    if not clean_assets:
        raise ValueError("Universe must contain at least one asset.")
    return Universe(name=name, assets=clean_assets, notes=notes)


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in prices.columns]
    if missing:
        raise ValueError(f"Price data is missing required columns: {missing}")

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "as_of_date" in frame.columns:
        frame["as_of_date"] = pd.to_datetime(frame["as_of_date"])
    else:
        frame["as_of_date"] = frame["date"]

    numeric_columns = ["open", "high", "low", "close", "volume"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame["asset"] = frame["asset"].astype(str)
    frame = frame.sort_values(["date", "asset"]).reset_index(drop=True)

    if (frame["as_of_date"] > frame["date"]).any():
        raise ValueError("Found rows whose as_of_date is after the market date.")
    if (frame["close"] <= 0).any():
        raise ValueError("Close prices must be positive.")
    if (frame["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")
    return frame


def make_data_quality_report(
    prices: pd.DataFrame,
    universe: Universe | None = None,
    stale_return_threshold: int = 5,
) -> DataQualityReport:
    frame = validate_prices(prices)
    price_missing = frame[["open", "high", "low", "close"]].isna().any(axis=1)
    volume_missing = frame["volume"].isna()
    returns = frame.groupby("asset", group_keys=False)["close"].pct_change()
    stale = returns.eq(0.0).groupby(frame["asset"]).rolling(stale_return_threshold).sum()
    stale_rows = int((stale.reset_index(level=0, drop=True) >= stale_return_threshold).sum())

    limitations = [
        "Synthetic/free-data MVP is not a substitute for institutional point-in-time data.",
        "Survivorship bias must be reassessed when replacing the demo universe.",
        "Corporate actions and delistings depend on the selected production data source.",
    ]
    survivorship_bias_free = bool(universe.survivorship_bias_free) if universe else False

    return DataQualityReport(
        start_date=str(frame["date"].min().date()),
        end_date=str(frame["date"].max().date()),
        n_assets=int(frame["asset"].nunique()),
        n_rows=int(len(frame)),
        missing_price_rows=int(price_missing.sum()),
        missing_volume_rows=int(volume_missing.sum()),
        stale_price_rows=stale_rows,
        survivorship_bias_free=survivorship_bias_free,
        limitations=tuple(limitations),
    )
