# Import Neccessory Libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# Step1: Data Ingestion
def data_ingestion():
    filepath = r"C:\MachineLearning_Operations\data\Concrete_Data.csv"
    df = pd.read_csv(filepath)
    return df
# Step2: Data Preprocessing and Cleaning
def data_preprocessing(df):
  '''
  Step1: Feature Engineering and Selection

  Step2: Split the dataset into independent and dependent columns i.e X and y

  Step3: Use train and test split i.e. Split the dataset into Seen and Unseen
  (Seen Data: train data and Unseen Data: test data)

  Step4: Use Scaling Technique/ Normalization Technique
  (StandardScaler/MinMaxScaler/RobustScaler)

  Step5: Return Seen and Unseen Data
  '''
  # Step1: Feature Engineering : Create a new feature based on existing feature

  df['Water_Binder_Ratio'] = df['Water  (component 4)(kg in a m^3 mixture)'] / (df['Cement (component 1)(kg in a m^3 mixture)'] + df['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'] + df['Fly Ash (component 3)(kg in a m^3 mixture)'])

  # Step2: Split the dataset into X and y
  target_col = 'Concrete compressive strength(MPa, megapascals) '
  X = df.drop(columns = [target_col])  # Independent Columns
  y = df[target_col]                   # Target Column

  # Step3: Split the dataset into Seen and Unseen (train and test data)
  from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
  from sklearn.model_selection import train_test_split

  X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                  test_size = 0.3,
                                                  random_state = 1)

  # Step4: Use Scaling/Normalization Technique
  '''
  * Scaling for training data: fit_transform()

  * Scaling for test data: transform()
  '''
  sc = MinMaxScaler()
  X_train = sc.fit_transform(X_train)   # Seen Data
  X_test = sc.transform(X_test)         # Unseen Data

  # Step5: return seen and unseen data
  return X_train,X_test,y_train,y_test

# Step3: Model Building
def model_build(X_train,X_test,y_train,y_test):
    
    model = DecisionTreeRegressor(random_state = 1) # Model Initialization
    model.fit(X_train,y_train) # Seen Data
    y_pred = model.predict(X_test) # Unseen Data 
    
    score = r2_score(y_test,y_pred) # (actual value,predicted value)
    
    return score,y_pred 

# Plot Model Performance

def plot_model_performance(y_test, y_pred):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    sns.regplot(x=y_test, y=y_pred, color="red", ax=ax[0])
    ax[0].set_title("Actual vs Predicted Values")
    ax[0].set_xlabel("Actual Values")
    ax[0].set_ylabel("Predicted Values")

    sns.kdeplot(y_test - y_pred, ax=ax[1])
    ax[1].set_title("Residual Error Distribution")
    ax[1].set_xlabel("Residual Errors")
    ax[1].set_ylabel("Density")
    plt.title('Decision Tree Model Performance')
    plt.tight_layout()
    plt.show()

# Define Orchestration Layer Design : Entry Point 
def main(): # Function Definition
    # Step1: Data Ingestion
    df = data_ingestion() 
    # Step2: Data Preprocessing and Cleaning
    X_train,X_test,y_train,y_test = data_preprocessing(df)
    # Step3: Model Building
    score,y_pred = model_build(X_train,X_test,y_train,y_test)  
    print("DecisionTreeRegressor r2 score : ",score)
    # Step4: Model Performance Plotting
    plot_model_performance(y_test,y_pred)
    
main() # Function calling
