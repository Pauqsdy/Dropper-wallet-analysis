from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

KAGGLE_MIRROR_URL = (
    "https://raw.githubusercontent.com/Lemoninmountain/"
    "Enhancing-Fraud-Detection-in-the-Ethereum-Blockchain-"
    "Using-Ensemble-Stacking-Machine-Learning/v1.0.0/"
    "transaction_dataset.csv"
)

def _download_text(url: str, timeout: int = 120) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def fetch_kaggle_fraud_dataset(save_path: Path | None = None) -> pd.DataFrame:
    """Kaggle Ethereum Fraud Detection (локальный CSV или зеркало на GitHub)."""
    if save_path and save_path.exists():
        df = pd.read_csv(save_path)
    else:
        csv_text = _download_text(KAGGLE_MIRROR_URL)
        df = pd.read_csv(io.StringIO(csv_text))
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)

    if "Address" not in df.columns:
        raise ValueError("Не найдена колонка Address в датасете Kaggle.")

    result = pd.DataFrame(
        {
            "address": df["Address"].astype(str),
            "label": df["FLAG"].map({0: "legitimate", 1: "fraud"}),
            "category": df["FLAG"].map({0: "kaggle_legit", 1: "kaggle_fraud"}),
            "entity": None,
            "source": "kaggle_mirror",
            "chain": "ETH",
        }
    )
    result["raw_features"] = df.drop(
        columns=["Address", "FLAG"], errors="ignore"
    ).to_dict(orient="records")

    return result


def build_wallet_registry(
    raw_dir: Path,
    sample_kaggle: int | None = 1000,
) -> pd.DataFrame:
    """Реестр: стратифицированная подвыборка Kaggle."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    kaggle_df = fetch_kaggle_fraud_dataset(raw_dir / "kaggle_transaction_dataset.csv")

    if sample_kaggle and len(kaggle_df) > sample_kaggle:
        per_label = max(1, sample_kaggle // 2)
        sampled_parts = [
            group.sample(n=min(len(group), per_label), random_state=42)
            for _, group in kaggle_df.groupby("label", sort=False)
        ]
        kaggle_df = pd.concat(sampled_parts, ignore_index=True)

    registry = kaggle_df.copy()

    eth_mask = registry["chain"] == "ETH"
    registry.loc[eth_mask, "address"] = registry.loc[eth_mask, "address"].str.lower()

    registry = registry.drop_duplicates(subset=["address"], keep="first").reset_index(drop=True)
    registry["wallet_id"] = range(1, len(registry) + 1)

    return registry


def save_registry(registry: pd.DataFrame, path: Path) -> None:
    export_df = registry.drop(columns=["raw_features"], errors="ignore")
    path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(path, index=False)
