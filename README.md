# Анализ дропперских кошельков Ethereum

Итоговый проект по курсу **«Анализ данных на Python»**.

**Дропперский кошелёк** — временный адрес в блокчейне, через который в мошеннических схемах проходят переводы. Задача: найти поведенческие признаки таких адресов и отличать их от обычных (`fraud` / `legitimate`).

## Источники данных

| Источник | Роль |
|----------|------|
| Kaggle — Ethereum Fraud Detection | разметка и агрегаты по адресам |
| Etherscan API | независимая проверка числа исходящих транзакций (`etherscan_txcount`) |

Выборка: **1000** адресов (500 fraud + 500 legitimate). Kaggle загружается из CSV; Etherscan — через API.

## Этапы

1. **Сбор** — реестр адресов + обогащение Etherscan  
2. **Предобработка** — импутация, признаки, выбросы  
3. **Анализ** — гипотезы H1–H4 (Mann–Whitney, Fisher)  
4. **Модели** — LogReg, KNN, метрики на test  


## Запуск

```bash
pip install -r requirements.txt
```

Создайте файл `.env` в корне проекта:

```
ETHERSCAN_API_KEY=ваш_ключ
```

```bash
jupyter notebook notebooks/dropper_wallets_analysis.ipynb
```

Выполняйте ячейки **сверху вниз**. Блок Etherscan (~1000 запросов) занимает **3–4 минуты**; повторный запуск берёт данные из кэша.

## Структура

```
notebooks/dropper_wallets_analysis.ipynb
src/label_sources.py      # Kaggle, реестр
src/etherscan_client.py   # API Etherscan V2
src/enrichment.py         # обогащение и сравнение с Kaggle
src/preprocessing.py      # признаки, train/test
data/processed/etherscan_enrichment.csv   # кэш API (создаётся при запуске)
```

## Состав команды

| Участник         | Роль |
|------------------|------|
| *Балашов Павел*  | *сбор данных, предобработка* |
| *Першин Илья*    | *анализ, модели* |
| *Шадрина Марина* | *модели* |