from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from src.api_clients import NO_PROXIES

KAGGLE_MIRROR_URL = (
    "https://raw.githubusercontent.com/Lemoninmountain/"
    "Enhancing-Fraud-Detection-in-the-Ethereum-Blockchain-"
    "Using-Ensemble-Stacking-Machine-Learning/v1.0.0/"
    "transaction_dataset.csv"
)

CONTROL_WALLETS = [
    {
        "address": "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B",
        "label": "legitimate",
        "category": "public_figure",
        "entity": "Vitalik Buterin",
        "source": "public",
    },
    {
        "address": "0x28C6c06298d514Db089934071355E03B1219d626",
        "label": "legitimate",
        "category": "exchange",
        "entity": "Binance Hot Wallet",
        "source": "public",
    },
    {
        "address": "0x71660c4005ba85c37ccec55d0c4494e57966cfe8",
        "label": "legitimate",
        "category": "exchange",
        "entity": "Coinbase",
        "source": "public",
    },
    {
        "address": "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",
        "label": "legitimate",
        "category": "exchange",
        "entity": "Binance",
        "source": "public",
    },
    {
        "address": "0x47ac0Fb4F2D84898e4D9E7b4DaB3C24507a6D503",
        "label": "legitimate",
        "category": "bridge",
        "entity": "Binance Peg",
        "source": "public",
    },
]

def _download_text(url: str, timeout: int = 120) -> str:
    response = requests.get(url, proxies=NO_PROXIES, timeout=timeout)
    response.raise_for_status()
    return response.text


def fetch_kaggle_fraud_dataset(save_path: Path | None = None) -> pd.DataFrame:
    """Kaggle Ethereum Fraud Detection (зеркало на GitHub)"""
    csv_text = _download_text(KAGGLE_MIRROR_URL)
    df = pd.read_csv(io.StringIO(csv_text))

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

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    return result


def get_control_wallets_df() -> pd.DataFrame:
    """Контрольные легитимные кошельки (ETH/TRON) для демонстрации API."""
    rows = []
    for wallet in CONTROL_WALLETS:
        rows.append({**wallet, "raw_features": None, "chain": "ETH"})
    return pd.DataFrame(rows)


def build_wallet_registry(
    raw_dir: Path,
    sample_kaggle: int | None = 1000,
    include_controls: bool = True,
) -> pd.DataFrame:
    """Реестр: стратифицированная выборка Kaggle + контрольные адреса бирж."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    kaggle_df = fetch_kaggle_fraud_dataset(raw_dir / "kaggle_transaction_dataset.csv")

    if sample_kaggle and len(kaggle_df) > sample_kaggle:
        per_label = max(1, sample_kaggle // 2)
        sampled_parts = [
            group.sample(n=min(len(group), per_label), random_state=42)
            for _, group in kaggle_df.groupby("label", sort=False)
        ]
        kaggle_df = pd.concat(sampled_parts, ignore_index=True)

    parts = [kaggle_df]
    if include_controls:
        parts.append(get_control_wallets_df())

    registry = pd.concat(parts, ignore_index=True)

    eth_mask = registry["chain"] == "ETH"
    registry.loc[eth_mask, "address"] = registry.loc[eth_mask, "address"].str.lower()

    registry = registry.drop_duplicates(subset=["address"], keep="first").reset_index(drop=True)
    registry["wallet_id"] = range(1, len(registry) + 1)

    return registry


def save_registry(registry: pd.DataFrame, path: Path) -> None:
    export_df = registry.drop(columns=["raw_features"], errors="ignore")
    path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(path, index=False)
