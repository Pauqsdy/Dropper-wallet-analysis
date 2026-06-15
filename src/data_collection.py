from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.api_clients import enrich_wallet_via_apis
from src.label_sources import build_wallet_registry, save_registry


def collect_labeled_addresses(
    raw_dir: Path,
    processed_dir: Path,
    sample_kaggle: int = 1000,
) -> pd.DataFrame:
    """Сбор размеченных адресов из источников."""
    registry = build_wallet_registry(raw_dir, sample_kaggle=sample_kaggle)
    save_registry(registry, processed_dir / "wallet_registry.csv")
    return registry


def select_api_holdout(
    registry: pd.DataFrame,
    holdout_size: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Holdout-выборка из Kaggle ETH для внешней валидации через API.

    Не пересекается с обучением по смыслу: признаки для этих адресов
    строятся только из сырых транзакций API, а не из Kaggle CSV.
    """
    pool = registry[
        (registry["source"] == "kaggle_mirror")].copy()

    if pool.empty:
        raise ValueError("В реестре нет Kaggle ETH-адресов для API holdout.")

    if holdout_size >= len(pool):
        return pool.reset_index(drop=True)

    per_label = max(1, holdout_size // pool["label"].nunique())
    parts = [
        group.sample(n=min(len(group), per_label), random_state=random_state)
        for _, group in pool.groupby("label", sort=False)
    ]
    holdout = pd.concat(parts, ignore_index=True)

    if len(holdout) > holdout_size:
        holdout = holdout.sample(n=holdout_size, random_state=random_state).reset_index(drop=True)

    return holdout


def enrich_wallets(
    wallets: pd.DataFrame,
    processed_dir: Path,
    max_api_pages: int = 3,
) -> pd.DataFrame:
    """Загрузка транзакций через API для заданного списка кошельков."""
    summaries = []
    tx_dir = processed_dir / "transactions"
    tx_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(wallets.iterrows(), total=len(wallets), desc="API enrichment"):
        address = row["address"]
        wallet_key = address.replace("0x", "")[:16]

        try:
            tx_data = enrich_wallet_via_apis(
                address, max_pages=max_api_pages
            )
        except Exception as exc:
            print(f"Ошибка для {address}: {exc}")
            summaries.append(
                {
                    "address": address,
                    "label": row["label"],
                    "category": row.get("category"),
                    "source": row.get("source"),
                    "api_status": "error",
                    "error": str(exc),
                }
            )
            continue

        summary = {
            "address": address,
            "label": row["label"],
            "category": row.get("category"),
            "source": row.get("source"),
            "api_status": "ok",
        }

        for tx_type, tx_df in tx_data.items():
            summary[f"{tx_type}_count"] = len(tx_df)
            if not tx_df.empty:
                file_name = f"{wallet_key}_{tx_type}.csv"
                tx_df.to_csv(tx_dir / file_name, index=False)

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(processed_dir / "api_enrichment_summary.csv", index=False)
    return summary_df


def enrich_sample_wallets(
    registry: pd.DataFrame,
    processed_dir: Path,
    sample_size: int = 100,
    max_api_pages: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Совместимость: holdout из Kaggle + обогащение."""
    holdout = select_api_holdout(registry, holdout_size=sample_size, random_state=random_state)
    holdout.to_csv(processed_dir / "api_holdout_addresses.csv", index=False)
    return enrich_wallets(holdout, processed_dir, max_api_pages=max_api_pages)


def run_data_collection(
    project_root: Path,
    sample_kaggle: int = 1000,
    api_holdout_size: int = 100,
    max_api_pages: int = 3,
) -> dict[str, pd.DataFrame]:
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    registry = collect_labeled_addresses(
        raw_dir, processed_dir, sample_kaggle=sample_kaggle
    )

    holdout = select_api_holdout(registry, holdout_size=api_holdout_size)
    holdout.to_csv(processed_dir / "api_holdout_addresses.csv", index=False)

    metadata = {
        "total_wallets": len(registry),
        "by_label": registry["label"].value_counts().to_dict(),
        "by_source": registry["source"].value_counts().to_dict(),
        "by_chain": registry["chain"].value_counts().to_dict(),
        "kaggle_sample": sample_kaggle,
        "api_holdout_size": len(holdout),
    }
    with open(processed_dir / "collection_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    api_summary = enrich_wallets(holdout, processed_dir, max_api_pages=max_api_pages)

    return {
        "registry": registry,
        "holdout": holdout,
        "api_summary": api_summary,
        "metadata": metadata,
    }
