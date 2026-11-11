"""Лабораторная работа №2: регрессия и классификация Spaceship Titanic.

Скрипт продолжает первую лабораторную. Он не изменяет исходный train.csv:
все преобразования обучаются только на обучающей части Pipeline, поэтому
статистики заполнения, масштабирования и категории не попадают из теста.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Позволяет сохранять графики без открытия окон.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "train.csv"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"
RANDOM_STATE = 42
CLASSIFICATION_TARGET = "Transported"
REGRESSION_TARGET = "Age"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Регрессия возраста и классификация пассажиров Spaceship Titanic."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Путь к train.csv (по умолчанию: data/train.csv).",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Папка для метрик, прогнозов и графиков.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Доля тестовой выборки, например 0.2.",
    )
    return parser.parse_args()


def make_features(df: pd.DataFrame, excluded_columns: list[str]) -> pd.DataFrame:
    """Удаляет идентификаторы и извлекает из Cabin только полезные категории."""
    features = df.drop(columns=excluded_columns + ["PassengerId", "Name"], errors="ignore").copy()

    if "Cabin" in features.columns:
        # Не используем pandas StringDtype: его pd.NA несовместим с
        # SimpleImputer(strategy="most_frequent") в некоторых версиях sklearn.
        # Обычный object-столбец хранит пропуски как np.nan и обрабатывается штатно.
        cabin_parts = features["Cabin"].astype(object).str.split("/", expand=True)
        features["CabinDeck"] = cabin_parts[0]
        features["CabinSide"] = cabin_parts[2]
        features = features.drop(columns="Cabin")

    return features


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Создаёт преобразователь, который будет обучаться внутри Pipeline."""
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_regression_plots(
    y_true: pd.Series, y_pred: np.ndarray, results_dir: Path
) -> None:
    residuals = y_true.to_numpy() - y_pred

    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(y_true, y_pred, alpha=0.55, color="#286090")
    boundaries = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axis.plot(boundaries, boundaries, "r--", label="Идеальный прогноз")
    axis.set(xlabel="Фактический возраст", ylabel="Предсказанный возраст")
    axis.set_title("Регрессия: фактические и предсказанные значения")
    axis.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "regression_actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    sns.histplot(residuals, bins=30, kde=True, ax=axis, color="#5cb85c")
    axis.axvline(0, color="black", linestyle="--")
    axis.set(xlabel="Остаток: фактическое значение - прогноз")
    axis.set_title("Регрессия: распределение остатков")
    fig.tight_layout()
    fig.savefig(results_dir / "regression_residuals.png", dpi=160)
    plt.close(fig)


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, results_dir: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["False", "True"],
        yticklabels=["False", "True"],
        ax=axis,
    )
    axis.set(xlabel="Предсказанный класс", ylabel="Фактический класс")
    axis.set_title("Матрица ошибок для Transported")
    fig.tight_layout()
    fig.savefig(results_dir / "classification_confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_feature_importance(
    pipeline: Pipeline, results_dir: Path, top_count: int = 15
) -> None:
    """Сохраняет наиболее важные признаки улучшенной модели классификации."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    importance = pd.Series(model.feature_importances_, index=names).nlargest(top_count).sort_values()

    fig, axis = plt.subplots(figsize=(8, 6))
    importance.plot.barh(ax=axis, color="#f0ad4e")
    axis.set(xlabel="Важность признака", ylabel="")
    axis.set_title("Топ признаков модели классификации")
    fig.tight_layout()
    fig.savefig(results_dir / "classification_feature_importance.png", dpi=160)
    plt.close(fig)


def run_regression(df: pd.DataFrame, results_dir: Path, test_size: float) -> dict[str, Any]:
    """Прогнозирует непрерывный признак Age двумя регрессионными моделями."""
    regression_data = df.dropna(subset=[REGRESSION_TARGET]).copy()
    X = make_features(regression_data, [REGRESSION_TARGET, CLASSIFICATION_TARGET])
    y = regression_data[REGRESSION_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    models = {
        "LinearRegression_baseline": LinearRegression(),
        "RandomForestRegressor_improved": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    metrics: dict[str, dict[str, float]] = {}
    final_predictions: np.ndarray | None = None

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor(X_train)), ("model", model)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics[model_name] = regression_metrics(y_test, predictions)
        if model_name == "RandomForestRegressor_improved":
            final_predictions = predictions

    assert final_predictions is not None
    pd.DataFrame(
        {"actual_age": y_test.to_numpy(), "predicted_age": final_predictions}
    ).to_csv(results_dir / "regression_test_predictions.csv", index=False)
    save_regression_plots(y_test, final_predictions, results_dir)

    return {
        "target": REGRESSION_TARGET,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "models": metrics,
    }


def normalise_binary_target(target: pd.Series) -> pd.Series:
    """Приводит Transported к 0/1 и даёт понятную ошибку при неожиданном формате."""
    values = target.astype("string").str.strip().str.lower()
    converted = values.map({"true": 1, "false": 0, "1": 1, "0": 0})
    if converted.isna().any():
        unexpected = sorted(values[converted.isna()].dropna().unique().tolist())
        raise ValueError(f"В {CLASSIFICATION_TARGET} обнаружены неизвестные значения: {unexpected}")
    return converted.astype(int)


def run_classification(df: pd.DataFrame, results_dir: Path, test_size: float) -> dict[str, Any]:
    """Предсказывает Transported и сохраняет метрики классификации."""
    if df[CLASSIFICATION_TARGET].isna().any():
        raise ValueError(f"В {CLASSIFICATION_TARGET} не должно быть пропусков.")

    X = make_features(df, [CLASSIFICATION_TARGET])
    y = normalise_binary_target(df[CLASSIFICATION_TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    models = {
        "LogisticRegression_baseline": LogisticRegression(max_iter=2000),
        "RandomForestClassifier_improved": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    metrics: dict[str, dict[str, float]] = {}
    final_predictions: np.ndarray | None = None
    final_pipeline: Pipeline | None = None
    final_report: dict[str, Any] | None = None

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor(X_train)), ("model", model)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics[model_name] = classification_metrics(y_test, predictions)
        if model_name == "RandomForestClassifier_improved":
            final_predictions = predictions
            final_pipeline = pipeline
            final_report = classification_report(
                y_test,
                predictions,
                target_names=["False", "True"],
                output_dict=True,
                zero_division=0,
            )

    assert final_predictions is not None and final_pipeline is not None and final_report is not None
    pd.DataFrame(
        {"actual_transported": y_test.to_numpy(), "predicted_transported": final_predictions}
    ).to_csv(results_dir / "classification_test_predictions.csv", index=False)
    save_confusion_matrix(y_test, final_predictions, results_dir)
    save_feature_importance(final_pipeline, results_dir)

    return {
        "target": CLASSIFICATION_TARGET,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "models": metrics,
        "random_forest_report": final_report,
    }


def make_text_report(metrics: dict[str, Any]) -> str:
    regression = metrics["regression"]
    classification = metrics["classification"]
    lines = [
        "Лабораторная работа №2 — Spaceship Titanic",
        "",
        "Регрессия",
        f"Целевая переменная: {regression['target']}",
        f"Строк в train/test: {regression['train_rows']}/{regression['test_rows']}",
    ]
    for name, values in regression["models"].items():
        lines.append(
            f"{name}: MAE={values['MAE']:.4f}; RMSE={values['RMSE']:.4f}; R2={values['R2']:.4f}"
        )

    lines.extend(
        [
            "",
            "Классификация",
            f"Целевая переменная: {classification['target']}",
            f"Строк в train/test: {classification['train_rows']}/{classification['test_rows']}",
        ]
    )
    for name, values in classification["models"].items():
        lines.append(
            f"{name}: accuracy={values['accuracy']:.4f}; precision={values['precision']:.4f}; "
            f"recall={values['recall']:.4f}; F1={values['f1']:.4f}"
        )
    lines.extend(
        [
            "",
            "Улучшение: вместо линейной/логистической базовых моделей добавлен "
            "случайный лес, который улавливает нелинейные зависимости и взаимодействия признаков.",
            "Итоговые метрики и графики следует брать из results после запуска на train.csv.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size должен быть числом между 0 и 1.")
    if not args.data.exists():
        raise FileNotFoundError(
            f"Не найден {args.data}. Скачайте train.csv из Spaceship Titanic и положите его в data/train.csv."
        )

    args.results.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    needed_columns = {CLASSIFICATION_TARGET, REGRESSION_TARGET}
    missing_columns = needed_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"В датасете нет обязательных столбцов: {sorted(missing_columns)}")

    metrics = {
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "regression": run_regression(df, args.results, args.test_size),
        "classification": run_classification(df, args.results, args.test_size),
    }
    (args.results / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = make_text_report(metrics)
    (args.results / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nРезультаты сохранены в: {args.results.resolve()}")


if __name__ == "__main__":
    main()
