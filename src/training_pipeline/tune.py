"""
Hyperparameter tuning with Optuna + MLflow.

- Optimizes XGBoost parameters using evaluation-set RMSE.
- Logs every trial to MLflow.
- Retrains the best model.
- Saves the best model to model_output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from joblib import dump
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor


DEFAULT_TRAIN = Path(
    "data/processed/feature_engineered_train.csv"
)

DEFAULT_EVAL = Path(
    "data/processed/feature_engineered_eval.csv"
)

DEFAULT_OUT = Path(
    "models/xgb_best_model.pkl"
)


def _normalize_tracking_uri(
    tracking_uri: str | Path,
) -> str:
    """
    Convert a local Windows/Linux path into a valid
    MLflow file URI.

    Example:

    C:\\project\\mlruns

    becomes:

    file:///C:/project/mlruns
    """

    uri = str(tracking_uri)

    # Already a valid URI, for example:
    # file:///...
    # http://...
    # sqlite:///...
    if "://" in uri:
        return uri

    # Convert local path to an absolute Path
    tracking_path = Path(uri).expanduser().resolve()

    # Create the directory if it does not exist
    tracking_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Convert Windows path to file:/// URI
    return tracking_path.as_uri()


def _maybe_sample(
    df: pd.DataFrame,
    sample_frac: Optional[float],
    random_state: int,
) -> pd.DataFrame:
    """
    Return a sample of the dataframe if sample_frac is valid.
    """

    if sample_frac is None:
        return df

    sample_frac = float(sample_frac)

    if sample_frac <= 0 or sample_frac >= 1:
        return df

    return df.sample(
        frac=sample_frac,
        random_state=random_state,
    ).reset_index(drop=True)


def _load_data(
    train_path: Path | str,
    eval_path: Path | str,
    sample_frac: Optional[float],
    random_state: int,
):
    """
    Load train and evaluation datasets.
    """

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    train_df = _maybe_sample(
        train_df,
        sample_frac,
        random_state,
    )

    eval_df = _maybe_sample(
        eval_df,
        sample_frac,
        random_state,
    )

    target = "price"

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_eval = eval_df.drop(columns=[target])
    y_eval = eval_df[target]

    return X_train, y_train, X_eval, y_eval


def tune_model(
    train_path: Path | str = DEFAULT_TRAIN,
    eval_path: Path | str = DEFAULT_EVAL,
    model_output: Path | str = DEFAULT_OUT,
    n_trials: int = 15,
    sample_frac: Optional[float] = None,
    tracking_uri: Optional[str | Path] = None,
    experiment_name: str = "xgboost_optuna_housing",
    random_state: int = 42,
) -> Tuple[Dict, Dict]:
    """
    Run Optuna hyperparameter tuning.

    Returns:
        best_params, best_metrics
    """

    # If tracking URI is not provided,
    # use a local mlruns folder.
    if tracking_uri is None:
        tracking_uri = Path("mlruns")

    # Convert Windows path into valid MLflow URI
    tracking_uri = _normalize_tracking_uri(tracking_uri)

    print("MLflow tracking URI:")
    print(tracking_uri)

    # Configure MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load data
    X_train, y_train, X_eval, y_eval = _load_data(
        train_path=train_path,
        eval_path=eval_path,
        sample_frac=sample_frac,
        random_state=random_state,
    )

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        """

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                200,
                800,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                10,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.3,
                log=True,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.5,
                1.0,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.5,
                1.0,
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                1,
                10,
            ),
            "gamma": trial.suggest_float(
                "gamma",
                0.0,
                5.0,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                10.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-8,
                10.0,
                log=True,
            ),
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        # Start one MLflow run for this Optuna trial
        with mlflow.start_run(nested=True):
            model = XGBRegressor(**params)

            model.fit(
                X_train,
                y_train,
            )

            y_pred = model.predict(X_eval)

            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_eval,
                        y_pred,
                    )
                )
            )

            mae = float(
                mean_absolute_error(
                    y_eval,
                    y_pred,
                )
            )

            r2 = float(
                r2_score(
                    y_eval,
                    y_pred,
                )
            )

            mlflow.log_params(params)

            mlflow.log_metrics(
                {
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                }
            )

        return rmse

    # Create and run Optuna study
    study = optuna.create_study(
        direction="minimize"
    )

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    best_params = study.best_trial.params

    print()
    print("Best params from Optuna:")
    print(best_params)

    # Retrain the best model
    best_model_params = {
        **best_params,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    best_model = XGBRegressor(
        **best_model_params
    )

    best_model.fit(
        X_train,
        y_train,
    )

    # Evaluate the best model
    y_pred = best_model.predict(X_eval)

    best_metrics = {
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_eval,
                    y_pred,
                )
            )
        ),
        "mae": float(
            mean_absolute_error(
                y_eval,
                y_pred,
            )
        ),
        "r2": float(
            r2_score(
                y_eval,
                y_pred,
            )
        ),
    }

    print()
    print("Best tuned model metrics:")
    print(best_metrics)

    # Save the best model locally
    output_path = Path(model_output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dump(
        best_model,
        output_path,
    )

    print()
    print(f"Best model saved to: {output_path}")

    # Log final best model to MLflow
    with mlflow.start_run(
        run_name="best_xgb_model"
    ):
        mlflow.log_params(best_params)

        mlflow.log_metrics(best_metrics)

        mlflow.xgboost.log_model(
            best_model,
            "model",
        )

    return best_params, best_metrics


if __name__ == "__main__":
    tune_model(
        tracking_uri=Path("mlruns")
    )