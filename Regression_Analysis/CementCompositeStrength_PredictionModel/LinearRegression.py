''' 
Composite Cement Strength Prediction Model using Linear Regression and other Regression Models Comparison
                       Model        MSE      RMSE       MAE        R2  CV Mean R2
            Ridge Regression 115.402475 10.742554  8.585569  0.571350    0.608906
              Bayesian Ridge 116.320755 10.785210  8.576524  0.567939    0.608798
           Linear Regression 116.849172 10.809680  8.575480  0.565976    0.608083
              ARD Regression 116.852579 10.809837  8.572007  0.565963    0.607313
               SGD Regressor 113.368821 10.647480  8.530404  0.578903    0.602026
             Huber Regressor 121.747946 11.033945  8.391437  0.547780    0.593306
Passive Aggressive Regressor 118.681137 10.894087  8.446009  0.559171    0.562029
           Poisson Regressor 133.784173 11.566511  9.639201  0.503073    0.535274
         Theil-Sen Regressor 197.830026 14.065206  8.857141  0.265181    0.427590
 Orthogonal Matching Pursuit 174.599788 13.213621 10.833673  0.351468    0.352450
                  Lasso LARS 188.010560 13.711694 11.240517  0.301655    0.303414
            Lasso Regression 188.010923 13.711708 11.240528  0.301653    0.303411
            RANSAC Regressor 949.319558 30.811030 14.574732 -2.526147    0.187511
                 Elastic Net 236.423221 15.376060 12.427823  0.121831    0.120437
           Tweedie Regressor 240.307449 15.501853 12.526746  0.107403    0.103975
             Gamma Regressor 240.822895 15.518469 12.546077  0.105489    0.102744
          Quantile Regressor 271.704143 16.483451 13.294868 -0.009216   -0.016043

Best Model: Ridge Regression
'''
# Import Necessary Libraries
# ---------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDRegressor,
    BayesianRidge,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor,
    PoissonRegressor,
    GammaRegressor,
    TweedieRegressor,
    QuantileRegressor,
    OrthogonalMatchingPursuit,
    ARDRegression,
    PassiveAggressiveRegressor,
    LassoLars
)

from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ---------------------------------------------------------------------------------
# Step1: Data Ingestion 
def data_ingestion():
    data = pd.read_csv(
        r'C:\MachineLearning_Operations\data\Concrete_Data.csv')
    return data

# Feature Engineering
def feature_engineering(df):
    # Create a copy of the DataFrame to avoid modifying the original data
    df = df.copy()

    cement = 'Cement (component 1)(kg in a m^3 mixture)'
    slag = 'Blast Furnace Slag (component 2)(kg in a m^3 mixture)'
    fly_ash = 'Fly Ash (component 3)(kg in a m^3 mixture)'
    water = 'Water  (component 4)(kg in a m^3 mixture)'
    superplasticizer = 'Superplasticizer (component 5)(kg in a m^3 mixture)'
    coarse = 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)'
    fine = 'Fine Aggregate (component 7)(kg in a m^3 mixture)'

    # Total Binder
    df['Total Binder'] = (
        df[cement] + df[slag] + df[fly_ash]
    )

    # Water to Binder Ratio
    df['Water to Binder Ratio'] = (
        df[water] /
        df['Total Binder'].replace(0, np.nan)
    )

    # Total Aggregate
    df['Total Aggregate'] = (
        df[coarse] + df[fine]
    )
    # Total Concrete Components
    df['Total Concrete Components'] = (
        df['Total Binder'] +
        df[water] +
        df[superplasticizer] +
        df['Total Aggregate']
    )

    return df


# Step2: Data Preprocessing
def data_preprocessing(df):

    # Remove Duplicate Rows
    df.drop_duplicates(inplace=True)
    # Target Column
    target_col = 'Concrete compressive strength(MPa, megapascals) '
    # Separate Features and Target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=1
    )

    # Numerical Columns After Feature Engineering
    numerical_columns = list(
        feature_engineering(X_train).columns
    )

    # Numerical Pipeline
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    
    # Categorical Pipeline (if needed in future)
    # categorical_pipeline = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='most_frequent')),
    #     ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
    # ])
    

    # Column Transformer
    pipeline = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_columns)],
        remainder='drop'
    )

    # Feature Engineering
    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)

    # Fit and Transform Training Data
    X_train = pipeline.fit_transform(X_train)

    # Transform Testing Data
    X_test = pipeline.transform(X_test)

    print("\nData Preprocessing Completed!")
    print("X_train Shape:", X_train.shape)
    print("X_test Shape:", X_test.shape)
    return X_train, X_test, y_train, y_test, pipeline


# Model Building and Evaluation

def model_building(X_train, X_test, y_train, y_test):

    models = {

        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(max_iter=10000),
        'Elastic Net': ElasticNet(max_iter=10000),
        'SGD Regressor': SGDRegressor(
            max_iter=2000,
            random_state=1),
        'Bayesian Ridge': BayesianRidge(),
        'Huber Regressor': HuberRegressor(
            max_iter=1000),
        'RANSAC Regressor': RANSACRegressor(
            random_state=1),
        'Theil-Sen Regressor': TheilSenRegressor(
            random_state=1),
        'Poisson Regressor': PoissonRegressor(
            max_iter=1000),
        'Gamma Regressor': GammaRegressor(
            max_iter=1000),
        'Tweedie Regressor': TweedieRegressor(
            max_iter=1000),
        'Quantile Regressor': QuantileRegressor(),
        'Orthogonal Matching Pursuit': OrthogonalMatchingPursuit(),
        'ARD Regression': ARDRegression(),
        'Passive Aggressive Regressor': PassiveAggressiveRegressor(
            max_iter=2000,
            random_state=1),
        'Lasso LARS': LassoLars()
    }

    results = []
    trained_models = {}
    for model_name, model in models.items():

        try:
            # Model Training
            model.fit(X_train, y_train)

            # Model Prediction
            y_pred = model.predict(X_test)

            # Model Evaluation
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Cross Validation
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=5,
                scoring='r2',
                n_jobs=-1
            )

            # Store Results

            results.append({
                'Model': model_name,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'CV Mean R2': cv_scores.mean()
            })

            trained_models[model_name] = model
            print(f"\n{model_name}")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R2: {r2:.4f}")
            print(f"CV Mean R2: {cv_scores.mean():.4f}")

        except Exception as e:
            print(f"\n{model_name} Failed: {e}")
            
    # Model Comparison

    results_df = pd.DataFrame(results)
    results_df.sort_values(
        by='CV Mean R2',
        ascending=False,
        inplace=True
    )

    print("\nModel Comparison")
    print(results_df.to_string(index=False))
    return trained_models, results_df


# Model Performance Visualization

def plot_model_performance(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=y_test,
        y=y_pred
    )
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'k--',
        lw=2
    )
    plt.tight_layout()
    plt.show()


# Model Comparison Visualization

def plot_model_comparison(results_df):
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=results_df,
        x='CV Mean R2',
        y='Model'
    )
    plt.title('Regression Model Comparison')
    plt.xlabel('Cross Validation R2')
    plt.ylabel('Regression Models')
    plt.tight_layout()
    plt.show()


# Main Orchestration Function

def main():

    # Step1: Data Ingestion
    df = data_ingestion()

    # Step2: Data Preprocessing
    X_train, X_test, y_train, y_test, pipeline = (
        data_preprocessing(df)
    )

    # Step3: Model Building
    trained_models, results_df = model_building(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Step4: Model Selection
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    print("\nBest Model:", best_model_name)

    # Predictions
    y_pred = best_model.predict(X_test)

    # Model Visualization
    plot_model_performance(
        y_test,
        y_pred
    )
    
    # Model Comparison Visualization
    plot_model_comparison(results_df)
    
    # Save Best Model and Preprocessing Pipeline
    joblib.dump(
        best_model,
        'best_model.pkl'
    )

    joblib.dump(
        pipeline,
        'preprocessing_pipeline.pkl'
    )

    print("\nBest Model Saved Successfully!")
    print("Preprocessing Pipeline Saved Successfully!")

# Execute Main Function

if __name__ == "__main__":

    main()
