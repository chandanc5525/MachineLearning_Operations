# Boosting Regressor Model Performance
'''
Boosting Model Performance Comparison
                        Model  R2 Score     RMSE      MAE
                 XGBRegressor  0.921747 4.528884 3.221243
HistGradientBoostingRegressor  0.909440 4.871998 3.333344
                LGBMRegressor  0.909378 4.873677 3.416037
            CatBoostRegressor  0.899486 5.132777 3.778087
    GradientBoostingRegressor  0.891961 5.321436 3.999671
            AdaBoostRegressor  0.727832 8.446131 6.781410
'''
# --------------------------------------------------------------------------------------------
import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# Logging Configuration
os.makedirs("log", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

logging.basicConfig(
    filename="log/boosting_model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------------------------------------------
# Step 1: Data Ingestion
def data_ingestion():

    try:
        filepath = r"C:\MachineLearning_Operations\data\Concrete_Data.csv"

        df = pd.read_csv(filepath)

        logging.info("Data ingestion completed successfully")

        return df

    except Exception as e:
        logging.exception("Data ingestion failed")
        raise e


# ---------------------------------------------------------------------------------------
# Step 2: Data Preprocessing and Feature Engineering
def data_preprocessing(df):

    try:

        # Feature Engineering
        df['Water_Binder_Ratio'] = (
            df['Water  (component 4)(kg in a m^3 mixture)'] /
            (
                df['Cement (component 1)(kg in a m^3 mixture)'] +
                df['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'] +
                df['Fly Ash (component 3)(kg in a m^3 mixture)']
            ).replace(0, np.nan)
        )

        # Remove Missing Values
        df = df.dropna()

        # Independent and Dependent Variables
        target_col = 'Concrete compressive strength(MPa, megapascals) '

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=1
        )

        logging.info("Data preprocessing completed successfully")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.exception("Data preprocessing failed")
        raise e


# ---------------------------------------------------------------------------------------
# Step 3: Boosting Model Building
def model_build(X_train, X_test, y_train, y_test):

    try:

        # Initialize Boosting Algorithms
        models = {

            'AdaBoostRegressor': AdaBoostRegressor(
                n_estimators=100,
                learning_rate=0.05,
                random_state=1
            ),

            'GradientBoostingRegressor': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=1
            ),

            'HistGradientBoostingRegressor': HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=1
            ),

            'XGBRegressor': XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=1,
                objective='reg:squarederror',
                n_jobs=-1
            ),

            'LGBMRegressor': LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=1,
                verbosity=-1,
                n_jobs=-1
            ),

            'CatBoostRegressor': CatBoostRegressor(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                random_seed=1,
                verbose=0
            )
        }

        scores = []
        predictions = {}
        trained_models = {}

        # Iterate Through All Models
        for model_name, model in models.items():

            logging.info(f"Training started: {model_name}")

            # Model Training
            model.fit(X_train, y_train)

            # Model Prediction
            y_pred = model.predict(X_test)

            # Model Evaluation
            r2 = r2_score(y_test, y_pred)

            rmse = np.sqrt(
                mean_squared_error(y_test, y_pred)
            )

            mae = mean_absolute_error(y_test, y_pred)

            # Store Evaluation Results
            scores.append({

                'Model': model_name,
                'R2 Score': r2,
                'RMSE': rmse,
                'MAE': mae

            })

            predictions[model_name] = y_pred
            trained_models[model_name] = model

            logging.info(
                f"{model_name} | R2: {r2:.4f} | RMSE: {rmse:.4f}"
            )

            print(f"\nModel: {model_name}")
            print(f"R2 Score: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")

        # Model Comparison
        scores_df = pd.DataFrame(scores)

        scores_df = scores_df.sort_values(
            by='R2 Score',
            ascending=False
        )

        # Identify Best Model
        best_model_name = scores_df.iloc[0]['Model']

        best_model = trained_models[best_model_name]
        best_y_pred = predictions[best_model_name]

        logging.info(f"Best Model Selected: {best_model_name}")

        return (
            scores_df,
            best_model_name,
            best_model,
            best_y_pred
        )

    except Exception as e:
        logging.exception("Model building failed")
        raise e


# ---------------------------------------------------------------------------------------
# Step 4: Model Performance Visualization
def plot_model_performance(y_test, y_pred, model_name):

    try:

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Actual vs Predicted Values
        sns.regplot(
            x=y_test,
            y=y_pred,
            color="red",
            ax=ax[0]
        )

        ax[0].set_title("Actual vs Predicted Values")
        ax[0].set_xlabel("Actual Values")
        ax[0].set_ylabel("Predicted Values")

        # Residual Error Distribution
        residuals = y_test - y_pred

        sns.kdeplot(
            residuals,
            fill=True,
            ax=ax[1]
        )

        ax[1].set_title("Residual Error Distribution")
        ax[1].set_xlabel("Residual Errors")
        ax[1].set_ylabel("Density")

        fig.suptitle(
            f"{model_name} - Model Performance"
        )

        plt.tight_layout()

        plt.savefig(
            f"reports/{model_name}_performance.png"
        )

        plt.show()

        logging.info("Model performance visualization completed")

    except Exception as e:
        logging.exception("Model visualization failed")
        raise e


# ---------------------------------------------------------------------------------------
# Step 5: Model Comparison Visualization
def plot_model_comparison(scores_df):

    try:

        plt.figure(figsize=(12, 6))

        sns.barplot(
            data=scores_df,
            x='Model',
            y='R2 Score'
        )

        plt.title("Boosting Algorithms - R2 Score Comparison")
        plt.xlabel("Boosting Algorithms")
        plt.ylabel("R2 Score")

        plt.xticks(rotation=30)

        plt.tight_layout()

        plt.savefig("reports/boosting_model_comparison.png")

        plt.show()

    except Exception as e:
        logging.exception("Model comparison visualization failed")
        raise e


# ---------------------------------------------------------------------------------------
# Step 6: Model Serialization
def save_model(model, model_name):

    try:

        filepath = f"models/{model_name}.pkl"

        joblib.dump(model, filepath)

        logging.info(f"Model saved successfully: {filepath}")

        print(f"\nModel saved successfully: {filepath}")

    except Exception as e:
        logging.exception("Model serialization failed")
        raise e


# ---------------------------------------------------------------------------------------
# Step 7: Main Orchestration Layer
def main():

    try:

        print("\nStarting Boosting Algorithms Pipeline")

        # Data Ingestion
        df = data_ingestion()

        # Data Preprocessing
        X_train, X_test, y_train, y_test = data_preprocessing(df)

        # Model Building
        (
            scores_df,
            best_model_name,
            best_model,
            best_y_pred
        ) = model_build(
            X_train,
            X_test,
            y_train,
            y_test
        )

        # Model Comparison Report
        print("\nBoosting Model Performance Comparison")

        print(scores_df.to_string(index=False))

        scores_df.to_csv(
            "reports/boosting_model_scores.csv",
            index=False
        )

        # Best Model
        print(f"\nBest Boosting Model: {best_model_name}")

        # Model Performance Visualization
        plot_model_performance(
            y_test,
            best_y_pred,
            best_model_name
        )

        # Model Comparison Visualization
        plot_model_comparison(scores_df)

        # Model Serialization
        save_model(best_model, best_model_name)

        logging.info("Boosting pipeline executed successfully")

        print("\nBoosting Pipeline Completed Successfully")

    except Exception as e:

        logging.exception("Boosting pipeline execution failed")

        print(f"Pipeline Error: {e}")


# Entry Point
if __name__ == "__main__":

    main()

# ---------------------------------------------------------------------------------------