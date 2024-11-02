import os
import csv
import pandas as pd
from utils import gen_path_label_csv
from sklearn.model_selection import train_test_split
def read_path_label_csv():
    df=pd.read_csv("./path_label.csv")
    return df

if __name__=="__main__":
    if not os.path.exists("./path_label.csv"):
        path="./datasets"
        gen_path_label_csv.gen_csv(path)
        print("path_label.csv created successfully")
    else:
        print("path_label.csv already exists")
        df=read_path_label_csv()
        X = df['path']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        print("X_train 筆數:", len(X_train))
        print("X_test 筆數:", len(X_test))
        print("y_train 筆數:", len(y_train))
        print("y_test 筆數:", len(y_test))

        print(X_train.head())
        print(y_train.head())
        print(X_test.head())
        print(y_test.head())
        print(y_train.value_counts())
        print(y_test.value_counts())
        
        
