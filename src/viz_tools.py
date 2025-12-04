# src/viz_tools.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# Helper for optional display() used in notebooks
try:
    from IPython.display import display as _display
except Exception:
    _display = None


def load_data(path: str | Path) -> pd.DataFrame:
    """Load CSV into a DataFrame."""
    return pd.read_csv(path)


def basic_inspect(df: pd.DataFrame, n: int = 5) -> None:
    """Show head, info and numeric description. Works in notebook and script."""
    print("HEAD:")
    if _display:
        _display(df.head(n))
    else:
        print(df.head(n).to_string())
    print("\nINFO:")
    # df.info() prints to stdout; capture it just in case
    buffer = []
    df.info(buf=buffer)
    # If buffer is a list (older pandas), just print info normally
    if isinstance(buffer, list):
        print()  # just fallback
        df.info()
    else:
        print(buffer.getvalue() if hasattr(buffer, "getvalue") else buffer)
    print("\nDESCRIPTION (numeric):")
    if _display:
        _display(df.describe())
    else:
        print(df.describe().to_string())


def prepare_datetime(df: pd.DataFrame, date_col: str = "date", date_format: Optional[str] = None) -> pd.DataFrame:
    """Convert date column to datetime and set as index."""
    df = df.copy()
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in DataFrame.")
    df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col)
    return df


def clean_df(df: pd.DataFrame, cols_keep: Optional[list] = None) -> pd.DataFrame:
    """Keep relevant columns and handle missing values sensibly."""
    df = df.copy()
    if cols_keep:
        keep = [c for c in cols_keep if c in df.columns]
        df = df.loc[:, keep]
    # Apply numeric missing value handling
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(method="ffill").fillna(method="bfill")
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


def _ensure_column(df: pd.DataFrame, col: str):
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in DataFrame.")


def daily_temperature_plot(df: pd.DataFrame, temp_col: str = "temp_avg", out_path: Optional[str | Path] = None):
    """Plot daily temperature (expects datetime index)."""
    _ensure_column(df, temp_col)
    try:
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df[temp_col], linewidth=0.9)
        plt.title("Daily Temperature Trend")
        plt.xlabel("Date")
        plt.ylabel("Temperature")
        plt.tight_layout()
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150)
        plt.show()
    except Exception as e:
        print(f"Error plotting daily temperature: {e}")
    finally:
        plt.close()


def monthly_rainfall_bar(df: pd.DataFrame, rain_col: str = "rain_mm", out_path: Optional[str | Path] = None):
    """Plot monthly rainfall totals (resamples by month and sums)."""
    _ensure_column(df, rain_col)
    try:
        monthly = df[rain_col].resample("M").sum()
        # convert index labels to a nicer string for bar x-ticks if many months
        labels = monthly.index.strftime("%Y-%m")
        plt.figure(figsize=(10, 4))
        plt.bar(labels, monthly.values, width=0.6)
        plt.xticks(rotation=45, ha="right")
        plt.title("Monthly Rainfall Totals")
        plt.xlabel("Month")
        plt.ylabel("Rainfall (mm)")
        plt.tight_layout()
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150)
        plt.show()
    except Exception as e:
        print(f"Error plotting monthly rainfall: {e}")
    finally:
        plt.close()


def humidity_vs_temp_scatter(df: pd.DataFrame, temp_col: str = "temp_avg", hum_col: str = "humidity", out_path: Optional[str | Path] = None):
    """Scatter plot humidity vs temperature."""
    _ensure_column(df, temp_col)
    _ensure_column(df, hum_col)
    try:
        plt.figure(figsize=(6, 6))
        plt.scatter(df[temp_col], df[hum_col], alpha=0.6)
        plt.title("Humidity vs Temperature")
        plt.xlabel("Temperature")
        plt.ylabel("Humidity")
        plt.tight_layout()
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150)
        plt.show()
    except Exception as e:
        print(f"Error plotting humidity vs temperature: {e}")
    finally:
        plt.close()


def combined_figure(df: pd.DataFrame, temp_col: str = "temp_avg", rain_col: str = "rain_mm", out_path: Optional[str | Path] = None):
    """Combined figure: temperature line + daily rainfall bars (resampled)."""
    _ensure_column(df, temp_col)
    _ensure_column(df, rain_col)
    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        # Temperature line
        df[temp_col].plot(ax=axes[0], title="Daily Temperature")
        axes[0].set_ylabel("Temperature")
        # Daily rainfall bars
        daily_rain = df[rain_col].resample("D").sum()
        axes[1].bar(daily_rain.index.strftime("%Y-%m-%d"), daily_rain.values, width=1.0)
        axes[1].set_title("Daily Rainfall (bars)")
        axes[1].set_ylabel("Rainfall (mm)")
        axes[1].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150)
        plt.show()
    except Exception as e:
        print(f"Error creating combined figure: {e}")
    finally:
        plt.close()


def save_cleaned(df: pd.DataFrame, path: str | Path):
    """Save cleaned DataFrame to CSV (ensures parent directory exists)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=True)
