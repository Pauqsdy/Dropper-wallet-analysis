"""Очистка, предобработка и feature engineering для анализа кошельков."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

USDT_MICRO_THRESHOLD = 0.01
SHORT_LIFETIME_DAYS = 30
MANY_SENDERS_THRESHOLD = 100

KAGGLE_KEY_COLS = {
    "Address": "address",
    "FLAG": "flag",
    "Time Diff between first and last (Mins)": "lifetime_mins",
    "Sent tnx": "sent_tx",
    "Received Tnx": "received_tx",
    "Unique Received From Addresses": "unique_senders",
    "Unique Sent To Addresses": "unique_receivers",
    "total transactions (including tnx to create contract": "total_tx",
    "total Ether sent": "total_eth_sent",
    "total ether received": "total_eth_received",
    "total ether balance": "total_eth_balance",
    " Total ERC20 tnxs": "erc20_tx",
    " ERC20 total Ether received": "erc20_eth_received",
    " ERC20 total ether sent": "erc20_eth_sent",
}

BASE_NUMERIC_COLS = [
    "lifetime_mins",
    "sent_tx",
    "received_tx",
    "unique_senders",
    "unique_receivers",
    "total_tx",
    "total_eth_sent",
    "total_eth_received",
    "total_eth_balance",
    "erc20_tx",
    "erc20_eth_received",
    "erc20_eth_sent",
]

OUTLIER_COLS = [
    "wallet_lifetime_days",
    "tx_per_day",
    "outbound_velocity",
    "unique_senders",
    "in_out_tx_ratio",
]


def clean_kaggle_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()
    cleaned["address"] = cleaned["Address"].astype(str).str.lower()
    return cleaned


def load_kaggle_features(kaggle_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(kaggle_path)
    raw = clean_kaggle_columns(raw)

    available = {src: dst for src, dst in KAGGLE_KEY_COLS.items() if src in raw.columns}
    features = raw[list(available.keys())].rename(columns=available)
    features["address"] = raw["address"]

    numeric_cols = [c for c in features.columns if c != "address"]
    for col in numeric_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    if "flag" in features.columns:
        features["flag"] = features["flag"].astype("Int64")

    return features


def _col_series(df: pd.DataFrame, col: str, default: float = 0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _lifetime_days(df: pd.DataFrame) -> pd.Series:
    days = _col_series(df, "lifetime_mins", default=np.nan) / (60 * 24)
    if "api_lifetime_days" in df.columns:
        days = days.fillna(pd.to_numeric(df["api_lifetime_days"], errors="coerce"))
    return days


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    lifetime_days = _lifetime_days(result)
    result["wallet_lifetime_days"] = lifetime_days
    result["is_short_lived"] = (lifetime_days < SHORT_LIFETIME_DAYS).astype("Int64")

    safe_lifetime = lifetime_days.clip(lower=1)
    total_tx = _col_series(result, "total_tx")
    sent_tx = _col_series(result, "sent_tx").clip(lower=0)
    received_tx = _col_series(result, "received_tx").clip(lower=0)

    result["tx_per_day"] = total_tx / safe_lifetime
    result["outbound_velocity"] = sent_tx / safe_lifetime
    result["in_out_tx_ratio"] = received_tx / sent_tx.replace(0, np.nan)
    result["unique_sender_density"] = (
        _col_series(result, "unique_senders") / received_tx.replace(0, np.nan)
    )
    result["has_many_senders"] = (
        _col_series(result, "unique_senders") >= MANY_SENDERS_THRESHOLD
    ).astype("Int64")

    total_tx_nz = total_tx.replace(0, np.nan)
    result["erc20_tx_share"] = _col_series(result, "erc20_tx") / total_tx_nz
    result["net_eth_flow"] = (
        _col_series(result, "total_eth_received") - _col_series(result, "total_eth_sent")
    )

    return result


def fill_base_from_api(df: pd.DataFrame) -> pd.DataFrame:
    """Для адресов с API-транзакциями подставляем реальные счётчики до медианы."""
    result = df.copy()
    has_api = _col_series(result, "api_has_transactions") > 0

    mappings = [
        ("api_lifetime_days", "lifetime_mins", lambda v: v * 60 * 24),
        ("api_sent_tx", "sent_tx", lambda v: v),
        ("api_received_tx", "received_tx", lambda v: v),
        ("api_total_tx", "total_tx", lambda v: v),
        ("api_unique_inbound_senders", "unique_senders", lambda v: v),
        ("api_unique_outbound_receivers", "unique_receivers", lambda v: v),
        ("api_erc20_tx", "erc20_tx", lambda v: v),
    ]

    for api_col, target_col, fn in mappings:
        if api_col not in result.columns or target_col not in result.columns:
            continue
        mask = has_api & result[target_col].isna()
        if not mask.any():
            continue
        values = pd.to_numeric(result.loc[mask, api_col], errors="coerce")
        result.loc[mask, target_col] = values.apply(fn)

    return result


def impute_missing_numeric(
    df: pd.DataFrame,
    group_col: str = "label",
    exclude: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Медиана внутри label (fraud / legitimate), затем глобальная медиана.
    """
    exclude = set(exclude or ["address", "wallet_id", "flag"])
    result = df.copy()
    report_rows = []

    numeric_cols = [
        c for c in result.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    for col in numeric_cols:
        missing_before = int(result[col].isna().sum())
        if missing_before == 0:
            continue

        if group_col in result.columns:
            result[col] = result.groupby(group_col, observed=True)[col].transform(
                lambda s: s.fillna(s.median())
            )

        global_median = result[col].median()
        fill_value = global_median if pd.notna(global_median) else 0
        result[col] = result[col].fillna(fill_value)
        missing_after = int(result[col].isna().sum())

        report_rows.append(
            {
                "column": col,
                "strategy": "median_by_label",
                "missing_before": missing_before,
                "missing_after": missing_after,
                "fill_value": fill_value,
            }
        )

    return result, pd.DataFrame(report_rows)


def finalize_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Убираем оставшиеся NaN в служебных и текстовых полях."""
    result = df.copy()

    if "entity" in result.columns:
        result["entity"] = result["entity"].fillna("unknown")

    if "flag" in result.columns:
        result["flag"] = result["flag"].fillna(-1).astype("Int64")

    for col in [c for c in result.columns if c.startswith("api_")]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)

    for col in result.select_dtypes(include=[np.number]).columns:
        if result[col].isna().any():
            med = result[col].median()
            result[col] = result[col].fillna(med if pd.notna(med) else 0)

    return result


def detect_iqr_anomalies(
    df: pd.DataFrame,
    columns: list[str],
    suffix: str = "_is_anomaly",
) -> pd.DataFrame:
    result = df.copy()

    for col in columns:
        if col not in result.columns:
            continue

        series = result[col].dropna()
        if series.empty:
            result[f"{col}{suffix}"] = 0
            continue

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        result[f"{col}{suffix}"] = (
            (result[col] < lower) | (result[col] > upper)
        ).astype(int)

    return result


def winsorize_outliers(
    df: pd.DataFrame,
    columns: list[str],
    suffix: str = "_was_winsorized",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Винзоризация: экстремумы заменяются границами IQR (строки не удаляются).
    """
    result = df.copy()
    report_rows = []

    for col in columns:
        if col not in result.columns:
            continue

        series = result[col].dropna()
        if series.empty:
            result[f"{col}{suffix}"] = 0
            continue

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        original = result[col].copy()
        result[col] = result[col].clip(lower=lower, upper=upper)
        changed = original.notna() & (original != result[col])
        result[f"{col}{suffix}"] = changed.astype(int)

        report_rows.append(
            {
                "column": col,
                "lower": lower,
                "upper": upper,
                "rows_adjusted": int(changed.sum()),
            }
        )

    return result, pd.DataFrame(report_rows)


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    label_map = {"legitimate": 0, "fraud": 1}
    result["label_code"] = result["label"].map(label_map)
    result["is_high_risk"] = (result["label"] == "fraud").astype(int)

    for col in ["category", "source", "chain"]:
        if col in result.columns:
            result[f"{col}_code"] = result[col].astype("category").cat.codes

    return result


def _parse_erc20_amount(value: float, decimals: float) -> float:
    if pd.isna(value) or pd.isna(decimals):
        return np.nan
    return float(value) / (10 ** float(decimals))


def _count_eth_activity(txs: pd.DataFrame, address: str) -> dict[str, int]:
    addr = address.lower()
    inbound = txs[txs["to"].astype(str).str.lower() == addr]
    outbound = txs[txs["from"].astype(str).str.lower() == addr]
    return {
        "api_sent_tx": len(outbound),
        "api_received_tx": len(inbound),
        "api_total_tx": len(
            txs[
                txs["from"].astype(str).str.lower().eq(addr)
                | txs["to"].astype(str).str.lower().eq(addr)
            ]
        ),
        "api_unique_inbound_senders": inbound["from"].nunique(),
        "api_unique_outbound_receivers": outbound["to"].nunique(),
        "api_erc20_tx": int(txs["tokenSymbol"].notna().sum()) if "tokenSymbol" in txs else 0,
    }


def _extract_eth_api_features(address: str, tx_dir: Path) -> dict[str, float | int | bool]:
    wallet_key = address.replace("0x", "")[:16]
    erc20_path = tx_dir / f"{wallet_key}_erc20_transactions.csv"
    eth_path = tx_dir / f"{wallet_key}_eth_transactions.csv"

    features: dict[str, float | int | bool] = {
        "api_has_transactions": 0,
        "api_lifetime_days": np.nan,
        "api_unique_inbound_senders": 0,
        "api_sent_tx": 0,
        "api_received_tx": 0,
        "api_total_tx": 0,
        "api_unique_outbound_receivers": 0,
        "api_erc20_tx": 0,
        "api_has_test_micro_transfer": 0,
        "api_outbound_24h_share": np.nan,
    }

    frames = []
    if erc20_path.exists():
        frames.append(pd.read_csv(erc20_path))
    if eth_path.exists():
        frames.append(pd.read_csv(eth_path))
    if not frames:
        return features

    features["api_has_transactions"] = 1
    txs = pd.concat(frames, ignore_index=True)
    features.update(_count_eth_activity(txs, address))

    if "timeStamp" in txs.columns:
        ts = pd.to_numeric(txs["timeStamp"], errors="coerce").dropna()
        if not ts.empty:
            features["api_lifetime_days"] = (ts.max() - ts.min()) / 86400

    addr_lower = address.lower()
    if "from" in txs.columns:
        inbound = txs[txs["to"].astype(str).str.lower() == addr_lower]
        if not inbound.empty and "timeStamp" in inbound.columns:
            inbound = inbound.copy()
            inbound["timeStamp"] = pd.to_numeric(inbound["timeStamp"], errors="coerce")
            first_in = inbound["timeStamp"].min()
            window_end = first_in + 86400
            outbound = txs[
                (txs["from"].astype(str).str.lower() == addr_lower)
                & pd.to_numeric(txs["timeStamp"], errors="coerce").between(first_in, window_end)
            ]
            total_out = txs[txs["from"].astype(str).str.lower() == addr_lower].shape[0]
            features["api_outbound_24h_share"] = (
                outbound.shape[0] / total_out if total_out else np.nan
            )

    if erc20_path.exists():
        erc20 = pd.read_csv(erc20_path)
        if {"value", "tokenDecimal", "to", "from"}.issubset(erc20.columns):
            erc20 = erc20.copy()
            erc20["amount"] = erc20.apply(
                lambda r: _parse_erc20_amount(r["value"], r["tokenDecimal"]), axis=1
            )
            inbound = erc20[erc20["to"].astype(str).str.lower() == addr_lower].sort_values(
                "timeStamp"
            )
            if not inbound.empty:
                micro = inbound[inbound["amount"] <= USDT_MICRO_THRESHOLD]
                large = inbound[inbound["amount"] > USDT_MICRO_THRESHOLD]
                if not micro.empty and not large.empty:
                    t_micro = pd.to_numeric(micro.iloc[0]["timeStamp"], errors="coerce")
                    t_large = pd.to_numeric(large.iloc[0]["timeStamp"], errors="coerce")
                    if pd.notna(t_micro) and pd.notna(t_large) and t_micro <= t_large:
                        features["api_has_test_micro_transfer"] = 1

    return features


def _extract_tron_api_features(address: str, tx_dir: Path) -> dict[str, float | int | bool]:
    wallet_key = address.replace("0x", "")[:16]
    trc20_path = tx_dir / f"{wallet_key}_trc20_transactions.csv"
    trx_path = tx_dir / f"{wallet_key}_trx_transactions.csv"

    features: dict[str, float | int | bool] = {
        "api_has_transactions": 0,
        "api_lifetime_days": np.nan,
        "api_unique_inbound_senders": 0,
        "api_sent_tx": 0,
        "api_received_tx": 0,
        "api_total_tx": 0,
        "api_unique_outbound_receivers": 0,
        "api_erc20_tx": 0,
        "api_has_test_micro_transfer": 0,
        "api_outbound_24h_share": np.nan,
    }

    frames = []
    if trc20_path.exists():
        frames.append(pd.read_csv(trc20_path))
    if trx_path.exists():
        frames.append(pd.read_csv(trx_path))
    if not frames:
        return features

    txs = pd.concat(frames, ignore_index=True)
    if txs.empty:
        return features

    features["api_has_transactions"] = 1
    if "to_address" in txs.columns:
        inbound = txs[txs["to_address"] == address]
        outbound = txs[txs["from_address"] == address]
        features.update(
            {
                "api_received_tx": len(inbound),
                "api_sent_tx": len(outbound),
                "api_total_tx": len(txs),
                "api_unique_inbound_senders": inbound["from_address"].nunique(),
                "api_unique_outbound_receivers": outbound["to_address"].nunique(),
            }
        )
        if trc20_path.exists():
            features["api_erc20_tx"] = len(pd.read_csv(trc20_path))

    if "block_ts" in txs.columns:
        ts = pd.to_numeric(txs["block_ts"], errors="coerce").dropna() / 1000
        if not ts.empty:
            features["api_lifetime_days"] = (ts.max() - ts.min()) / 86400

    if trc20_path.exists():
        trc20 = pd.read_csv(trc20_path)
        inbound = trc20[trc20["to_address"] == address]
        if not inbound.empty and "quant" in inbound.columns:
            inbound = inbound.copy()
            inbound["amount"] = pd.to_numeric(inbound["quant"], errors="coerce") / 1e6
            micro = inbound[inbound["amount"] <= USDT_MICRO_THRESHOLD]
            large = inbound[inbound["amount"] > USDT_MICRO_THRESHOLD]
            if not micro.empty and not large.empty:
                features["api_has_test_micro_transfer"] = 1

    return features


def build_api_features(registry: pd.DataFrame, tx_dir: Path) -> pd.DataFrame:
    rows = []
    for _, row in registry.iterrows():
        address = row["address"]
        chain = row.get("chain", "ETH")
        if chain == "TRON":
            feats = _extract_tron_api_features(address, tx_dir)
        else:
            feats = _extract_eth_api_features(address, tx_dir)
        feats["address"] = address
        rows.append(feats)
    return pd.DataFrame(rows)


def api_features_to_kaggle_format(api_feats: dict, address: str) -> dict:
    """Агрегаты API в те же поля, что и Kaggle CSV (для внешней валидации)."""
    lifetime_days = api_feats.get("api_lifetime_days", np.nan)
    addr = address.lower() if str(address).startswith("0x") else address
    return {
        "address": addr,
        "lifetime_mins": lifetime_days * 60 * 24 if pd.notna(lifetime_days) else np.nan,
        "sent_tx": api_feats.get("api_sent_tx", 0),
        "received_tx": api_feats.get("api_received_tx", 0),
        "unique_senders": api_feats.get("api_unique_inbound_senders", 0),
        "unique_receivers": api_feats.get("api_unique_outbound_receivers", 0),
        "total_tx": api_feats.get("api_total_tx", 0),
        "erc20_tx": api_feats.get("api_erc20_tx", 0),
        "feature_source": "api_aggregate",
        "api_has_test_micro_transfer": api_feats.get("api_has_test_micro_transfer", 0),
        "api_outbound_24h_share": api_feats.get("api_outbound_24h_share", np.nan),
    }


def build_api_holdout_features(
    holdout: pd.DataFrame,
    tx_dir: Path,
    kaggle_path: Path | None = None,
) -> pd.DataFrame:
    """
    Признаки holdout-кошельков только из API-транзакций (формат Kaggle).

    Истинные метки (`flag`) берём из реестра; Kaggle-колонки — для сравнения
    «API vs эталон» на этапе 3 (обучение модели).
    """
    rows = []
    for _, row in holdout.iterrows():
        address = row["address"]
        chain = row.get("chain", "ETH")
        if chain == "TRON":
            raw = _extract_tron_api_features(address, tx_dir)
        else:
            raw = _extract_eth_api_features(address, tx_dir)

        kaggle_fmt = api_features_to_kaggle_format(raw, address)
        kaggle_fmt["label"] = row["label"]
        kaggle_fmt["flag"] = 1 if row["label"] == "fraud" else 0
        rows.append(kaggle_fmt)

    result = pd.DataFrame(rows)

    if kaggle_path and kaggle_path.exists():
        truth = load_kaggle_features(kaggle_path)
        compare_cols = [
            c for c in BASE_NUMERIC_COLS
            if c in truth.columns and c in result.columns
        ]
        truth_subset = truth[["address"] + compare_cols].rename(
            columns={c: f"{c}_kaggle" for c in compare_cols}
        )
        result = result.merge(truth_subset, on="address", how="left")

    return create_derived_features(result)


def save_api_holdout_features(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_feature_matrix(
    registry: pd.DataFrame,
    kaggle_path: Path,
    tx_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    kaggle_features = load_kaggle_features(kaggle_path)
    merged = registry.merge(kaggle_features, on="address", how="left")

    if tx_dir and tx_dir.exists():
        merged = merged.merge(build_api_features(registry, tx_dir), on="address", how="left")

    merged = fill_base_from_api(merged)
    imputed, imputation_report = impute_missing_numeric(merged)
    with_derived = create_derived_features(imputed)
    with_derived, _ = impute_missing_numeric(
        with_derived,
        exclude=["address", "wallet_id", "flag"],
    )

    flagged = detect_iqr_anomalies(with_derived, OUTLIER_COLS)
    winsorized, winsor_report = winsorize_outliers(flagged, OUTLIER_COLS)
    encoded = encode_categorical_features(winsorized)
    final = finalize_matrix(encoded)

    quality_report = {
        "imputation": imputation_report,
        "winsorization": winsor_report,
        "missing_summary": final.isna().sum().sort_values(ascending=False).head(20),
        "rows": len(final),
        "kaggle_coverage": int(final["flag"].isin([0, 1]).sum()),
        "api_coverage": int((_col_series(final, "api_has_transactions") > 0).sum()),
        "nan_total": int(final.isna().sum().sum()),
    }

    return final, quality_report


def save_feature_matrix(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_quality_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _to_records(key: str) -> list[dict]:
        val = report.get(key, pd.DataFrame())
        return val.to_dict(orient="records") if isinstance(val, pd.DataFrame) else val

    serializable = {
        "rows": report["rows"],
        "kaggle_coverage": report["kaggle_coverage"],
        "api_coverage": report.get("api_coverage", 0),
        "nan_total": report.get("nan_total", 0),
        "imputation": _to_records("imputation"),
        "winsorization": _to_records("winsorization"),
        "missing_summary": report["missing_summary"].to_dict(),
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(serializable, file, ensure_ascii=False, indent=2)
