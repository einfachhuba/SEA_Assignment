import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .config import (
    DEFAULT_RF_ESTIMATORS,
    DEFAULT_RF_MAX_DEPTH,
    DEFAULT_RF_RANDOM_STATE,
    DEFAULT_RF_WARM_START,
    DEFAULT_RF_INCREMENTAL,
    DEFAULT_RF_INCREMENTAL_STEP,
)

DATA_DIR = "app/data/particle_swarm_optimization"
DATA_FILE = os.path.join(DATA_DIR, "covtype.parquet")

@st.cache_data
def load_data(sample_size=5000, random_state=42):
    """
    Loads the Covertype dataset. Downloads and saves if not present.
    Returns X, y, feature_names, and a sample dataframe for display.
    """
    if os.path.exists(DATA_FILE):
        df = pd.read_parquet(DATA_FILE)
        X = df.drop(columns=["Cover_Type"]).values
        y = df["Cover_Type"].values
        feature_names = df.drop(columns=["Cover_Type"]).columns.tolist()
    else:
        # Fetch data with feature names
        data = fetch_covtype(as_frame=True)
        df = data.frame
        
        # Ensure directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Save to parquet for faster subsequent loads
        df.to_parquet(DATA_FILE)
        
        X = df.drop(columns=["Cover_Type"]).values
        y = df["Cover_Type"].values
        feature_names = df.drop(columns=["Cover_Type"]).columns.tolist()
    
    # Return a small sample for display BEFORE standardization and subsampling
    display_df = df.sample(n=5, random_state=random_state)

    # Subsample for performance
    if sample_size and sample_size < len(y):
        indices = np.random.RandomState(random_state).choice(len(y), sample_size, replace=False)
        X = X[indices]
        y = y[indices]
        
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, feature_names, display_df


def evaluate_subset(X_train, X_test, y_train, y_test, selected_indices, model_params=None):
    """Evaluates accuracy of a Random Forest classifier on selected features."""
    if len(selected_indices) == 0:
        return 0.0

    params = model_params or {}
    n_estimators = int(params.get("n_estimators", DEFAULT_RF_ESTIMATORS))
    max_depth = params.get("max_depth", DEFAULT_RF_MAX_DEPTH)
    random_state = int(params.get("random_state", DEFAULT_RF_RANDOM_STATE))
    warm_start = bool(params.get("warm_start", DEFAULT_RF_WARM_START))
    incremental = bool(params.get("incremental", DEFAULT_RF_INCREMENTAL)) and warm_start
    chunk = params.get("incremental_step", DEFAULT_RF_INCREMENTAL_STEP) or DEFAULT_RF_INCREMENTAL_STEP
    chunk = int(max(1, min(chunk, n_estimators)))

    rf = RandomForestClassifier(
        n_estimators=chunk if incremental else n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        warm_start=warm_start,
        n_jobs=-1,
    )

    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]

    if incremental:
        built = 0
        while built < n_estimators:
            built = min(n_estimators, built + chunk)
            rf.n_estimators = built
            rf.fit(X_train_sel, y_train)
    else:
        rf.fit(X_train_sel, y_train)

    y_pred = rf.predict(X_test_sel)
    return accuracy_score(y_test, y_pred)
