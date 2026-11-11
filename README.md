# Лабораторная работа №2 — Spaceship Titanic

Проект продолжает первую лабораторную работу с датасетом [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data).

Скрипт решает две задачи на одном `train.csv`:

1. **Регрессия:** прогноз непрерывного признака `Age`.
2. **Классификация:** прогноз целевого признака `Transported`.

В обеих задачах сравниваются базовая и улучшенная модели. Все предобработки включены в `Pipeline`, поэтому заполнение пропусков, нормализация и кодирование категорий обучаются только на обучающей выборке — без утечки данных из тестовой части.

## Структура

```text
lab2_spaceship_titanic/
├── data/
│   └── train.csv              # Скачать с Kaggle, в репозиторий не добавлять
├── results/                   # Создаётся после запуска
├── lab2_ml.py                 # Основной скрипт
├── requirements.txt
└── README.md
```

## Запуск в VS Code

1. Откройте папку `lab2_spaceship_titanic` в VS Code.
2. Скачайте `train.csv` со страницы соревнования Kaggle и поместите его в `data/train.csv`.
3. Создайте виртуальное окружение и установите библиотеки:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4. Запустите работу:

```powershell
python lab2_ml.py
```

Можно указать другой путь к данным и другую долю теста:

```powershell
python lab2_ml.py --data C:\path\to\train.csv --test-size 0.2
```

## Что делает программа

### Подготовка признаков

- исключает `PassengerId` и `Name`, так как это почти уникальные идентификаторы;
- извлекает из `Cabin` палубу `CabinDeck` и сторону `CabinSide`, а номер каюты не использует;
- заполняет числовые пропуски медианой;
- заполняет категориальные пропуски модой;
- нормализует числовые признаки с `MinMaxScaler`;
- кодирует категории с `OneHotEncoder(handle_unknown="ignore")`.

### Регрессия

Цель — `Age`. Строки, в которых возраст отсутствует, исключаются только из задачи регрессии. Для признаков не используются ни `Age`, ни `Transported`, чтобы не было утечки правильного ответа. Сравниваются:

- `LinearRegression` — базовая линейная модель;
- `RandomForestRegressor` — улучшенная модель, учитывающая нелинейные зависимости.

Метрики: **MAE**, **RMSE**, **R²**.

### Классификация

Цель — `Transported`. Сравниваются:

- `LogisticRegression` — базовая модель;
- `RandomForestClassifier` — улучшенная нелинейная модель.

Метрики: **accuracy**, **precision**, **recall**, **F1**. Разделение стратифицировано, поэтому соотношение классов сохраняется в обучающей и тестовой выборках.

## Результаты

После запуска в `results/` будут:

- `report.txt` — компактный текст для отчёта;
- `metrics.json` — все метрики и classification report;
- `regression_test_predictions.csv` и `classification_test_predictions.csv` — прогнозы на тесте;
- `regression_actual_vs_predicted.png` и `regression_residuals.png` — оценка регрессии;
- `classification_confusion_matrix.png` — матрица ошибок;
- `classification_feature_importance.png` — наиболее важные признаки случайного леса.

## Как сформулировать вывод

После запуска перенесите численные значения из `results/report.txt`. В выводе сравните две модели каждой задачи: меньшее MAE/RMSE и большее R² означают лучшую регрессию; для классификации сравнивайте прежде всего F1 вместе с precision и recall, а не только accuracy.
