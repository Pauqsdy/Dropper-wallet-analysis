from __future__ import annotations

import os
import time
from typing import Any

import requests

ETHERSCAN_V2_URL = "https://api.etherscan.io/v2/api"
ETH_MAINNET_CHAIN_ID = 1
DEFAULT_MIN_INTERVAL_SEC = 0.2


class EtherscanClient:
    def __init__(
        self,
        api_key: str | None = None,
        min_interval_sec: float = DEFAULT_MIN_INTERVAL_SEC,
    ) -> None:
        self.api_key = api_key or os.getenv("ETHERSCAN_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Не найден ETHERSCAN_API_KEY. Добавьте ключ в файл .env "
                "(ключ не коммитится в git)."
            )
        self.min_interval_sec = min_interval_sec
        self._last_call_at = 0.0
        self._session = requests.Session()
        self._session.trust_env = False

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_at
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)
        self._last_call_at = time.monotonic()

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        self._throttle()
        query = {
            "chainid": ETH_MAINNET_CHAIN_ID,
            "apikey": self.api_key,
            **params,
        }
        response = self._session.get(ETHERSCAN_V2_URL, params=query, timeout=60)
        response.raise_for_status()
        return response.json()

    def get_outgoing_tx_count(self, address: str) -> int:
        payload = self._get(
            {
                "module": "proxy",
                "action": "eth_getTransactionCount",
                "address": address,
                "tag": "latest",
            }
        )
        if "result" not in payload:
            raise RuntimeError(f"Etherscan: неожиданный ответ: {payload}")
        return int(payload["result"], 16)
