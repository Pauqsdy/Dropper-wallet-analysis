from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

NO_PROXIES = {"http": None, "https": None}
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")

ETHERSCAN_V2_URL = "https://api.etherscan.io/v2/api"
ETH_CHAIN_ID = 1
REQUEST_DELAY_SEC = 0.25


def _sleep_between_requests() -> None:
    time.sleep(REQUEST_DELAY_SEC)


def _etherscan_get(params: dict[str, Any]) -> dict[str, Any]:
    """Запрос к Etherscan API"""
    query = {
        "chainid": ETH_CHAIN_ID,
        "apikey": ETHERSCAN_API_KEY,
        **params,
    }
    response = requests.get(
        ETHERSCAN_V2_URL,
        params=query,
        proxies=NO_PROXIES,
        timeout=60,
    )
    response.raise_for_status()
    _sleep_between_requests()
    return response.json()

def get_eth_transactions(
    address: str,
    start_block: int = 0,
    end_block: int = 99999999,
    page_size: int = 1000,
    max_pages: int = 20,
) -> pd.DataFrame:
    """Получение обычных ETH-транзакций с пагинацией"""
    if not ETHERSCAN_API_KEY:
        raise ValueError(
            "ETHERSCAN_API_KEY не задан. Добавьте ключ в .env "
            "(https://etherscan.io/apidashboard)."
        )

    all_rows: list[dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        data = _etherscan_get(
            {
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": start_block,
                "endblock": end_block,
                "page": page,
                "offset": page_size,
                "sort": "asc",
            }
        )

        if data.get("status") != "1":
            if page == 1:
                print(f"Etherscan [{address}]: {data.get('message')}")
            break

        batch = data.get("result", [])
        if not batch:
            break

        all_rows.extend(batch)
        if len(batch) < page_size:
            break

    return pd.DataFrame(all_rows)


def get_erc20_transfers(
    address: str,
    page_size: int = 1000,
    max_pages: int = 20,
) -> pd.DataFrame:
    """Получение ERC-20 переводов с пагинацией"""
    if not ETHERSCAN_API_KEY:
        raise ValueError("ETHERSCAN_API_KEY не задан.")

    all_rows: list[dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        data = _etherscan_get(
            {
                "module": "account",
                "action": "tokentx",
                "address": address,
                "page": page,
                "offset": page_size,
                "sort": "asc",
            }
        )

        if data.get("status") != "1":
            if page == 1:
                print(f"Etherscan ERC20 [{address}]: {data.get('message')}")
            break

        batch = data.get("result", [])
        if not batch:
            break

        all_rows.extend(batch)
        if len(batch) < page_size:
            break

    return pd.DataFrame(all_rows)




def enrich_wallet_via_apis(
    address: str,
    chain: str = "ETH",
    max_pages: int = 5,
) -> dict[str, pd.DataFrame]:
    """Обогащение кошелька транзакциями через API"""
    result: dict[str, pd.DataFrame] = {}

    if chain.upper() == "ETH":
        result["eth_transactions"] = get_eth_transactions(
            address, max_pages=max_pages
        )
        result["erc20_transactions"] = get_erc20_transfers(
            address, max_pages=max_pages
        )
    else:
        raise ValueError(f"Неподдерживаемая сеть: {chain}")

    return result
