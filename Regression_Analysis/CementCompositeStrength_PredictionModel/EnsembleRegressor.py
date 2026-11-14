
# Ensembel Model Performance Comparison:
'''
                       Model  R2 Score      RMSE
0      RandomForestRegressor  0.896581  5.206432
1        ExtraTreesRegressor  0.900127  5.116394
2  GradientBoostingRegressor  0.894113  5.268181

Best Performing Model: ExtraTreesRegressor

'''
# ----------------------------------------------------------------------------------------
# Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

# ----------------------------------------------------------------------------------------
# Step 1: Data Ingestion
def data_ingestion():

    filepath = r"C:\MachineLearning_Operations\data\Concrete_Data.csv"

    df = pd.read_csv(filepath)

    return df
# ----------------------------------------------------------------------------------------
# Step 2: Data Preprocessing and Cleaning
def data_preprocessing(df):

    # Feature Engineering
    df['Water_Binder_Ratio'] = (
        df['Water  (component 4)(kg in a m^3 mixture)'] /
        (
            df['Cement (component 1)(kg in a m^3 mixture)'] +
            df['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'] +
            df['Fly Ash (component 3)(kg in a m^3 mixture)']
        )
    )

    # Split Independent and Dependent Variables
    target_col = 'Concrete compressive strength(MPa, megapascals) '

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=1
    )

    return X_train, X_test, y_train, y_test
# ----------------------------------------------------------------------------------------
# Step 3: Model Building and Evaluation
def model_build(X_train, X_test, y_train, y_test):

    models = {
        'RandomForestRegressor': RandomForestRegressor(random_state=1),
        'ExtraTreesRegressor': ExtraTreesRegressor(random_state=1),
        'GradientBoostingRegressor': GradientBoostingRegressor(random_state=1)
    }

    scores = []
    predictions = {}

    for model_name, model in models.items():

        # Model Training
        model.fit(X_train, y_train)

        # Model Prediction
        y_pred = model.predict(X_test)

        # Model Evaluation
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        scores.append({
            'Model': model_name,
            'R2 Score': r2,
            'RMSE': rmse
        })

        predictions[model_name] = y_pred

        print(f"\nModel: {model_name}")
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")

    # Create Model Comparison DataFrame
    scores_df = pd.DataFrame(scores)

    # Identify Best Model
    best_model_name = scores_df.loc[
        scores_df['R2 Score'].idxmax(), 'Model'
    ]

    best_y_pred = predictions[best_model_name]

    return scores_df, best_model_name, best_y_pred

# ----------------------------------------------------------------------------------------
# Step 4: Model Performance Visualization
def plot_model_performance(y_test, y_pred, model_name):

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

    # Residual Error Distribution
    residuals = y_test - y_pred

    sns.kdeplot(
        residuals,
        ax=ax[1],
        fill=True
    )

    ax[1].set_title("Residual Error Distribution")
    ax[1].set_xlabel("Residual Errors")
    ax[1].set_ylabel("Density")

    fig.suptitle(f"{model_name} - Model Performance")

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------------------
# Step 5: Orchestration Layer
def main():

    # Data Ingestion
    df = data_ingestion()

    # Data Preprocessing
    X_train, X_test, y_train, y_test = data_preprocessing(df)

    # Model Building and Evaluation
    scores_df, best_model_name, best_y_pred = model_build(
        X_train, X_test, y_train, y_test
    )

    # Display Model Comparison
    print("\nModel Performance Comparison:")
    print(scores_df)

    print(f"\nBest Performing Model: {best_model_name}")

    # Model Performance Visualization
    plot_model_performance(
        y_test,
        best_y_pred,
        best_model_name
    )

# Entry Point
if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------------------
