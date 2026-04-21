import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import os

def load_and_preprocess_data():
    print("Fetching dataset from UCI ML Repository...")
    # fetch dataset 
    air_quality = fetch_ucirepo(id=360) 
      
    # Convert to pandas DataFrame
    # Sometimes ucimlrepo splits into features and targets, let's combine them first if they are separated
    df_features = air_quality.data.features
    df_targets = air_quality.data.targets
    
    if df_targets is not None and not df_targets.empty:
        df = pd.concat([df_features, df_targets], axis=1)
    else:
        df = df_features.copy()

    print("Data fetched successfully. Preprocessing...")
    
    # Missing values are marked as -200
    df = df.replace(-200, np.nan)
    
    # Target variable will be CO(GT) (Carbon Monoxide)
    target_col = 'CO(GT)'
    
    if target_col not in df.columns:
        print(f"Warning: '{target_col}' not found. Available columns: {df.columns.tolist()}")
        # Fallback to the first numeric column that has missing values or just the first column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = numeric_cols[0]
        print(f"Falling back to target: {target_col}")
    
    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])
    
    # Fill remaining missing values in features with the median
    df = df.fillna(df.median(numeric_only=True))
    
    # Select features (all numeric columns except target)
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in numeric_features if col != target_col]
    
    X = df[features]
    y = df[target_col]
    
    return X, y, features, target_col

def train_and_evaluate(X, y):
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print(f"Model Evaluation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    
    return model, metrics

def save_model(model, features, target_col, metrics, filename="xgboost_air_quality_model.pkl"):
    print(f"Saving model to {filename}...")
    model_data = {
        'model': model,
        'features': features,
        'target_col': target_col,
        'metrics': metrics
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print("Model saved successfully!")

if __name__ == "__main__":
    X, y, features, target_col = load_and_preprocess_data()
    model, metrics = train_and_evaluate(X, y)
    save_model(model, features, target_col, metrics)
    print("Training process completed.")
