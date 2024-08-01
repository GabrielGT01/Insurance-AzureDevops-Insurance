
"""
Prepares raw data and provides training, validation and test datasets
"""

import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer

TARGET_COL = ["charges"]
NUM_COL = ["age", "bmi", "children"]
CAT_ORD_COL = ["sex", "smoker", "region" ]


def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--val_data", type=str, help="Path to val dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--enable_monitoring", type=str, help="enable logging to ADX")
    parser.add_argument("--table_name", type=str, default="mlmonitoring", help="Table name in ADX for logging")
    args = parser.parse_args()
    return args

def log_training_data(df, table_name):
    from obs.collector import Online_Collector
    collector = Online_Collector(table_name)
    collector.batch_collect(df)

def main(args):
    '''Read, split, and save datasets'''
    # ------------ Reading Data ------------ #
    data = pd.read_csv(Path(args.raw_data))
    data = data[NUM_COL + CAT_ORD_COL  + TARGET_COL]
    
    # Remove duplicate rows
    data = data.drop_duplicates()
    
    # Retrieve index location for prices greater than 50k as outliers
    outliers = data[data['charges'] >= 50000].index
    data.drop(outliers, axis=0, inplace=True)
    

    # Split the data into train (60%) and temp (40%)
    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)

    # Split the temp data into validation (50% of temp, which is 20% of original) and test (50% of temp, which is 20% of original)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Save the datasets to CSV files
    #train_data.to_csv('train_data.csv', index=False)
    #val_data.to_csv('val_data.csv', index=False)
    #test_data.to_csv('test_data.csv', index=False)


    
    
    # Save datasets
    train_data.to_parquet((Path(args.train_data) / "train_data.parquet"))
    val_data.to_parquet((Path(args.val_data) / "val_data.parquet"))
    test_data.to_parquet((Path(args.test_data) / "test_data.parquet"))
    
    if args.enable_monitoring.lower() in ['true', '1', 'yes']:
        log_training_data(data, args.table_name)

if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()
    
    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Val dataset output path: {args.val_data}",
        f"Test dataset path: {args.test_data}"
    ]
    
    for line in lines:
        print(line)
    
    main(args)
    mlflow.end_run()
