# Анализ признаков дропперских кошельков в блокчейне

Итоговый проект по курсу анализа данных.

## Тема

Обнаружение поведенческих паттернов кошельков, используемых в дропперских схемах (антифрод, блокчейн-аналитика).

## Источник данных

| Источник | URL |
|----------|-----|
| Kaggle Ethereum Fraud Detection | [kaggle.com](https://www.kaggle.com/datasets/vagifa/ethereum-fraud-detection) |

Датасет содержит готовые агрегаты по Ethereum-адресам (транзакции, срок жизни, отправители) и метки `fraud` / `legitimate`.

## Структура проекта

- `notebooks/dropper_wallets_analysis.ipynb` — основной отчёт (этапы 1–4)
- `src/label_sources.py` — загрузка Kaggle и реестр адресов
- `src/preprocessing.py` — признаки, импутация, выбросы
- `data/raw/` — сырой CSV Kaggle
- `data/processed/` — `wallet_features.csv`, метаданные

## Запуск

```bash
pip install -r requirements.txt
jupyter notebook notebooks/dropper_wallets_analysis.ipynb
```

Ячейки ноутбука выполняются последовательно сверху вниз.
