"""
Trains ML model using training dataset. Saves trained model.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn



TARGET_COL = ["charges"]
NUM_COL = ["age", "bmi", "children"]
CAT_ORD_COL = ["sex", "smoker", "region"]



def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")

    # regressor specific arguments
 
    parser.add_argument("--n_estimators", dest='n_estimators', type=int, default=100)
    parser.add_argument("--learning_rate", dest='learning_rate', type=float, default=0.1)
    parser.add_argument("--max_depth", dest='max_depth', type=int, default=3)
    parser.add_argument("--random_state", dest='random_state', type=int, default=123)

    args = parser.parse_args()

    return args


def main(args):
    '''Read train dataset, train model, save trained model'''

    # Read train data
    train_data = pd.read_parquet(Path(args.train_data))


    # Split the data into input(X) and output(y)
    y_train = train_data[TARGET_COL]
    X_train = train_data[NUM_COL  + CAT_ORD_COL]

    # Preprocessing for numerical data
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='first'))])

    
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUM_COL),
            ('cat', categorical_transformer, CAT_ORD_COL)
        ])





    # Create the pipeline and  Train model with the train set
    model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('gbr', GradientBoostingRegressor(
                n_estimators= args.n_estimators, 
                learning_rate= args.learning_rate, 
                max_depth= args.max_depth, 
                random_state= args.random_state))
        ]).fit(X_train, y_train)


    # Save the model
    mlflow.sklearn.save_model(sk_model=model_pipeline, args.model_output))


    # log model hyperparameters
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("random_state", args.random_state)


    # Predict using the Regression Model
    yhat_train = model_pipeline.predict(X_train)


    # Evaluate Regression performance with the train set
    r2 = r2_score(y_train, yhat_train)
    mse = mean_squared_error(y_train, yhat_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, yhat_train)


    # log model performance metrics
    mlflow.log_metric("train r2", r2)
    mlflow.log_metric("train mse", mse)
    mlflow.log_metric("train rmse", rmse)
    mlflow.log_metric("train mae", mae)


    # Visualize results
    plt.scatter(y_train, yhat_train,  color='black')
    plt.plot(y_train, y_train, color='blue', linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.savefig("regression_results.png")
    mlflow.log_artifact("regression_results.png")




    if __name__ == "__main__":
    
        mlflow.start_run()

        # ---------- Parse Arguments ----------- #
        # -------------------------------------- #

        args = parse_args()


        lines = [
        f"Train dataset input path: {args.train_data}",
        f"Model output path: {args.model_output}",
        f"n_estimators: {args.n_estimators}",
        f"learning_rate: {args.learning_rate}",
        f"max_depth: {args.max_depth}",
        f"random_state: {args.random_state}"
        ]

        for line in lines:
            print(line)

        main(args)

        mlflow.end_run()


