
# Import Necessary Libraries
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from flaml import AutoML

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# Create Required Directories
os.makedirs("log", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# Logging Configuration
logging.basicConfig(
    filename="log/flaml_model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


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


# Step 2: Data Preprocessing and Feature Engineering
def data_preprocessing(df):

    try:

        # Feature Engineering
        binder = (
            df['Cement (component 1)(kg in a m^3 mixture)'] +
            df['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'] +
            df['Fly Ash (component 3)(kg in a m^3 mixture)']
        )

        df['Water_Binder_Ratio'] = (
            df['Water  (component 4)(kg in a m^3 mixture)'] /
            binder.replace(0, np.nan)
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


# Step 3: FLAML Automated Model Building
def model_build(X_train, X_test, y_train, y_test):

    try:

        # Initialize FLAML
        automl = AutoML()

        # FLAML Configuration
        automl_settings = {

            "time_budget": 120,

            "task": "regression",

            "metric": "rmse",

            "estimator_list": [
                "lgbm",
                "xgboost",
                "xgb_limitdepth",
                "catboost",
                "histgb"
            ],

            "eval_method": "holdout",

            "split_ratio": 0.2,

            "n_jobs": 2,

            "seed": 1,

            "log_file_name": "log/flaml_automl.log",

            "verbose": 2

        }

        logging.info("FLAML AutoML training started")

        # Train FLAML
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            **automl_settings
        )

        # Best Model Information
        best_model_name = automl.best_estimator

        best_config = automl.best_config

        # Prediction on Unseen Data
        y_pred = automl.predict(X_test)

        logging.info(
            f"Best FLAML model selected: {best_model_name}"
        )

        logging.info(
            f"Best configuration: {best_config}"
        )

        return automl, best_model_name, best_config, y_pred

    except Exception as e:

        logging.exception("FLAML model building failed")
        raise e


# Step 4: Model Evaluation
def model_evaluation(y_test, y_pred, best_model_name):

    try:

        # Evaluation Metrics
        r2 = r2_score(y_test, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )

        mae = mean_absolute_error(y_test, y_pred)

        # Evaluation Report
        results = pd.DataFrame({

            "Model": [best_model_name],
            "R2 Score": [r2],
            "RMSE": [rmse],
            "MAE": [mae]

        })

        print("\nFLAML Model Evaluation")
        print(results.to_string(index=False))

        # Save Evaluation Report
        results.to_csv(
            "reports/flaml_model_scores.csv",
            index=False
        )

        logging.info(
            f"Evaluation completed | R2: {r2:.4f} | RMSE: {rmse:.4f}"
        )

        return results

    except Exception as e:

        logging.exception("Model evaluation failed")
        raise e


# Step 5: Model Performance Visualization
def plot_model_performance(y_test, y_pred, model_name):

    try:

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Actual vs Predicted
        sns.regplot(
            x=y_test,
            y=y_pred,
            color="red",
            ax=ax[0]
        )

        ax[0].set_title("Actual vs Predicted Values")
        ax[0].set_xlabel("Actual Values")
        ax[0].set_ylabel("Predicted Values")

        # Residual Distribution
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
            "reports/flaml_model_performance.png"
        )

        plt.show()

        logging.info("Model performance visualization completed")

    except Exception as e:

        logging.exception("Visualization failed")
        raise e


# Step 6: Model Serialization
def save_model(automl):

    try:

        filepath = "models/concrete_flaml_automl.pkl"

        # Save Complete FLAML AutoML Object
        automl.pickle(filepath)

        logging.info(
            f"FLAML model saved successfully: {filepath}"
        )

        print(f"\nModel saved successfully: {filepath}")

    except Exception as e:

        logging.exception("Model serialization failed")
        raise e


# Step 7: Main Orchestration Layer
def main():

    try:

        print("\nStarting FLAML AutoML Pipeline")

        # Data Ingestion
        df = data_ingestion()

        # Data Preprocessing
        X_train, X_test, y_train, y_test = data_preprocessing(df)

        # Automated Model Building
        (
            automl,
            best_model_name,
            best_config,
            y_pred
        ) = model_build(
            X_train,
            X_test,
            y_train,
            y_test
        )

        # Model Evaluation
        results = model_evaluation(
            y_test,
            y_pred,
            best_model_name
        )

        # Display Best Model
        print("\nBest FLAML Model:", best_model_name)

        print("\nBest Hyperparameters:")
        print(best_config)

        # Model Performance Visualization
        plot_model_performance(
            y_test,
            y_pred,
            best_model_name
        )

        # Model Serialization
        save_model(automl)

        logging.info("FLAML pipeline completed successfully")

        print("\nFLAML Pipeline Completed Successfully")

    except Exception as e:

        logging.exception("FLAML pipeline execution failed")

        print(f"Pipeline Error: {e}")


# Entry Point
if __name__ == "__main__":

    main()
