from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.etherscan_client import EtherscanClient
from src.preprocessing import load_kaggle_features


def enrich_with_etherscan_txcount(
    registry: pd.DataFrame,
    kaggle_path: Path,
    cache_path: Path | None = None,
    client: EtherscanClient | None = None,
) -> pd.DataFrame:
    cache_path = cache_path or Path("data/processed/etherscan_enrichment.csv")
    kaggle_features = load_kaggle_features(kaggle_path)
    kaggle_cols = kaggle_features[
        ["address", "sent_tx", "received_tx", "total_tx"]
    ].copy()

    base = registry.merge(kaggle_cols, on="address", how="left")
    base["address"] = base["address"].astype(str).str.lower()

    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        cached["address"] = cached["address"].astype(str).str.lower()
        base = base.merge(
            cached[["address", "etherscan_txcount"]],
            on="address",
            how="left",
        )
    else:
        base["etherscan_txcount"] = pd.NA

    client = client or EtherscanClient()
    rows_to_fetch = base[base["etherscan_txcount"].isna()].copy()

    if not rows_to_fetch.empty:
        fetched_rows = []
        total = len(rows_to_fetch)
        for i, row in enumerate(rows_to_fetch.itertuples(index=False), start=1):
            address = str(row.address).lower()
            txcount = pd.NA
            status = "ok"
            for attempt in range(2):
                try:
                    txcount = client.get_outgoing_tx_count(address)
                    status = "ok"
                    break
                except Exception as exc:  # noqa: BLE001
                    status = f"error: {exc}"
                    if attempt == 0:
                        time.sleep(1.0)
            fetched_rows.append(
                {
                    "address": address,
                    "etherscan_txcount": txcount,
                    "fetch_status": status,
                }
            )
            if i % 50 == 0 or i == total:
                print(f"Etherscan: {i}/{total} адресов обработано")

        fetched_df = pd.DataFrame(fetched_rows)
        if cache_path.exists():
            cached = pd.read_csv(cache_path)
            cached["address"] = cached["address"].astype(str).str.lower()
            combined = pd.concat([cached, fetched_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["address"], keep="last")
        else:
            combined = fetched_df

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(cache_path, index=False)

        if "fetch_status" in combined.columns:
            errors = int(combined["fetch_status"].astype(str).str.startswith("error").sum())
            if errors:
                print(f"Предупреждение: {errors} адресов без ответа API (см. fetch_status в кэше)")

        base = base.drop(columns=["etherscan_txcount"], errors="ignore")
        base = base.merge(
            combined[["address", "etherscan_txcount"]],
            on="address",
            how="left",
        )

    base["etherscan_txcount"] = pd.to_numeric(base["etherscan_txcount"], errors="coerce")
    base["diff_sent_tx"] = base["sent_tx"] - base["etherscan_txcount"]
    base["diff_total_tx"] = base["total_tx"] - base["etherscan_txcount"]

    return base


def enrichment_summary(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=["etherscan_txcount", "sent_tx"])
    exact_sent_match = int((valid["diff_sent_tx"] == 0).sum())
    return {
        "addresses_total": len(df),
        "addresses_with_api": int(df["etherscan_txcount"].notna().sum()),
        "exact_match_sent_tx": exact_sent_match,
        "exact_match_sent_tx_pct": round(100 * exact_sent_match / max(len(valid), 1), 1),
        "mean_abs_diff_sent": float(valid["diff_sent_tx"].abs().mean()) if len(valid) else None,
        "mean_abs_diff_total": float(
            df.dropna(subset=["etherscan_txcount", "total_tx"])["diff_total_tx"].abs().mean()
        )
        if df["etherscan_txcount"].notna().any()
        else None,
    }
