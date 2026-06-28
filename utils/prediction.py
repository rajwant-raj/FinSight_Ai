"""
FinSight AI — ML Prediction
Random Forest Regressor for next-day closing price prediction.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils.preprocessing import prepare_ml_features


# ── Directory for persisted models ──
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


@dataclass
class PredictionResult:
    """Container for prediction output and evaluation metrics."""

    predicted_price: float
    mae: float
    rmse: float
    r2: float
    train_size: int
    test_size: int
    feature_names: list
    feature_importances: np.ndarray


def train_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Optional[PredictionResult], Optional[str]]:
    """
    Train a Random Forest Regressor to predict the next trading day's
    closing price.

    Parameters
    ----------
    df : pd.DataFrame
        Historical OHLCV data.
    test_size : float
        Fraction of data used for testing (default: 0.2).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple
        (PredictionResult, error_message)
        On success error_message is None; on failure PredictionResult is None.
    """
    try:
        # Prepare features
        X, y = prepare_ml_features(df)

        if len(X) < 50:
            return None, (
                "Insufficient data for training. At least 50 data points "
                "are required after feature engineering. Try expanding your "
                "date range."
            )

        # Train / test split (chronological — no shuffle for time series)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train the model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Predict next trading day (use the last available row of features)
        last_features = X.iloc[[-1]]
        predicted_price = float(model.predict(last_features)[0])

        # Save the trained model
        save_model(model, "random_forest_model.pkl")

        return PredictionResult(
            predicted_price=predicted_price,
            mae=mae,
            rmse=rmse,
            r2=r2,
            train_size=len(X_train),
            test_size=len(X_test),
            feature_names=list(X.columns),
            feature_importances=model.feature_importances_,
        ), None

    except Exception as exc:
        return None, (
            f"Model training failed. Please ensure sufficient historical "
            f"data is available.\n\n*Error detail:* `{exc}`"
        )


def save_model(model: RandomForestRegressor, filename: str) -> str:
    """Persist a trained model to disk and return the file path."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    filepath = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, filepath)
    return filepath


def load_model(filename: str) -> Optional[RandomForestRegressor]:
    """Load a previously saved model. Returns None if not found."""
    filepath = os.path.join(MODEL_DIR, filename)
    if os.path.exists(filepath):
        return joblib.load(filepath)
    return None
