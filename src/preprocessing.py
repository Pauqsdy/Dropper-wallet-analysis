from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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

    return features.drop_duplicates(subset=["address"], keep="first")


def _col_series(df: pd.DataFrame, col: str, default: float = 0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _lifetime_days(df: pd.DataFrame) -> pd.Series:
    return _col_series(df, "lifetime_mins", default=np.nan) / (60 * 24)


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


MODEL_FEATURE_COLS = [
    "wallet_lifetime_days",
    "outbound_velocity",
    "in_out_tx_ratio",
    "unique_senders",
]

_IMPUTE_EXCLUDE = ["address", "wallet_id", "flag"]


def _numeric_feature_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]


def _imputation_stats(
    df: pd.DataFrame,
    group_col: str = "label",
    exclude: list[str] | None = None,
) -> dict:
    exclude_set = set(exclude or _IMPUTE_EXCLUDE)
    by_group: dict[str, dict[str, float]] = {}
    global_medians: dict[str, float] = {}

    for col in _numeric_feature_cols(df, exclude_set):
        if group_col in df.columns:
            medians = df.groupby(group_col, observed=True)[col].median()
            by_group[col] = {
                str(label): float(value)
                for label, value in medians.items()
                if pd.notna(value)
            }

        global_median = df[col].median()
        global_medians[col] = (
            float(global_median) if pd.notna(global_median) else 0.0
        )

    return {"by_group": by_group, "global": global_medians}


def _apply_imputation_stats(
    df: pd.DataFrame,
    stats: dict,
    group_col: str = "label",
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    exclude_set = set(exclude or _IMPUTE_EXCLUDE)
    result = df.copy()

    for col in _numeric_feature_cols(result, exclude_set):
        if group_col in result.columns and col in stats["by_group"]:
            for label, fill_value in stats["by_group"][col].items():
                mask = result[group_col].astype(str) == label
                result.loc[mask, col] = result.loc[mask, col].fillna(fill_value)

        fill_value = stats["global"].get(col, 0.0)
        result[col] = result[col].fillna(fill_value)

    return result


def _iqr_bounds(df: pd.DataFrame, columns: list[str]) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}

    for col in columns:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))

    return bounds


def winsorize_with_bounds(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    suffix: str = "_was_winsorized",
) -> pd.DataFrame:
    result = df.copy()

    for col, (lower, upper) in bounds.items():
        if col not in result.columns:
            continue

        original = result[col].copy()
        result[col] = result[col].clip(lower=lower, upper=upper)
        changed = original.notna() & (original != result[col])
        result[f"{col}{suffix}"] = changed.astype(int)

    return result


def build_base_frame(registry: pd.DataFrame, kaggle_path: Path) -> pd.DataFrame:
    """Merge Kaggle + производные признаки без импутации и винзоризации."""
    kaggle_features = load_kaggle_features(kaggle_path)
    merged = registry.merge(kaggle_features, on="address", how="left")
    return create_derived_features(merged)


def fit_preprocess_state(
    train_df: pd.DataFrame,
    group_col: str = "label",
) -> dict:
    """Статистики предобработки только по train (без утечки из test)."""
    after_pass1 = _apply_imputation_stats(
        train_df, _imputation_stats(train_df, group_col), group_col
    )
    pass2_stats = _imputation_stats(after_pass1, group_col)
    after_pass2 = _apply_imputation_stats(after_pass1, pass2_stats, group_col)

    return {
        "pass1": _imputation_stats(train_df, group_col),
        "pass2": pass2_stats,
        "iqr": _iqr_bounds(after_pass2, OUTLIER_COLS),
        "group_col": group_col,
    }


def apply_preprocess_state(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Применить к train/test статистики, посчитанные на train."""
    group_col = state.get("group_col", "label")
    after_pass1 = _apply_imputation_stats(df, state["pass1"], group_col)
    after_pass2 = _apply_imputation_stats(after_pass1, state["pass2"], group_col)
    winsorized = winsorize_with_bounds(after_pass2, state["iqr"])
    encoded = encode_categorical_features(winsorized)
    return finalize_matrix(encoded)


def prepare_modeling_split(
    registry: pd.DataFrame,
    kaggle_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Разбиение до предобработки; импутация и винзоризация — только по train."""
    from sklearn.model_selection import train_test_split

    base = build_base_frame(registry, kaggle_path)
    labeled = base[base["flag"].isin([0, 1])].copy()

    train_base, test_base = train_test_split(
        labeled,
        test_size=test_size,
        random_state=random_state,
        stratify=labeled["flag"],
    )
    state = fit_preprocess_state(train_base)
    train_df = apply_preprocess_state(train_base, state)
    test_df = apply_preprocess_state(test_base, state)
    return train_df, test_df, state


def impute_missing_numeric(
    df: pd.DataFrame,
    group_col: str = "label",
    exclude: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    result = df.copy()

    if "entity" in result.columns:
        result["entity"] = result["entity"].fillna("unknown")

    if "flag" in result.columns:
        result["flag"] = result["flag"].fillna(-1).astype("Int64")

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


def build_feature_matrix(
    registry: pd.DataFrame,
    kaggle_path: Path,
) -> tuple[pd.DataFrame, dict]:
    kaggle_features = load_kaggle_features(kaggle_path)
    merged = registry.merge(kaggle_features, on="address", how="left")
    merged = merged.drop(columns=["raw_features"], errors="ignore")

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
        "nan_total": report.get("nan_total", 0),
        "imputation": _to_records("imputation"),
        "winsorization": _to_records("winsorization"),
        "missing_summary": report["missing_summary"].to_dict(),
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(serializable, file, ensure_ascii=False, indent=2)
