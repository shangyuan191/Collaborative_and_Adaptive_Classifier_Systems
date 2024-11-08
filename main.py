import os
import csv
import pandas as pd
from utils import gen_path_label_csv
from utils.dataset import MyDataset
from utils.dataset import MyImbalancedDataset
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from utils import data_util
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
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224))])
        ################################# Q1 ####################################################
        # start_time=time.time()
        # train_size=500
        # test_size=100
        # myDataset=MyDataset("./path_label.csv",train_size=train_size,test_size=test_size, transform=transform)
        # train_data,test_data=myDataset.train_data,myDataset.test_data
        # train_loader=DataLoader(train_data,batch_size=64,shuffle=True,num_workers=16)
        # test_loader=DataLoader(test_data,batch_size=64,shuffle=True,num_workers=16)
        # knn = KNeighborsClassifier(n_neighbors=5)
        # svm = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True))
        # decision_tree = DecisionTreeClassifier()
        # random_forest = RandomForestClassifier()
        # estimators=[('knn', knn),('random_forest', random_forest)]
        # voting_clf = data_util.init_classifier(estimators)

        # X_train, y_train = data_util.feature_extraction(train_loader)
        # X_test, y_test = data_util.feature_extraction(test_loader)
        # print("Begin training...")
        # voting_clf.fit(X_train, y_train)
        # print("Training finished.")
        # y_pred = voting_clf.predict(X_test)
        # acc=accuracy_score(y_test, y_pred)
        # f1=f1_score(y_test, y_pred, average='weighted')
        # precision=precision_score(y_test, y_pred, average='weighted')
        # recall=recall_score(y_test, y_pred, average='weighted')
        # print("Accuracy:", acc)
        # print("F1 Score:", f1)
        # print("Precision:", precision)
        # print("Recall:", recall)
        
        # time_taken=time.time()-start_time
        # print("Time taken:", time_taken)
        # data_util.write_results_to_file("results.txt", acc, f1, precision, recall,estimators,train_size,test_size,time_taken)
        
        
        ################################# Q2 ####################################################
        start_time=time.time()
        myImbalancedDataset=MyImbalancedDataset("./path_label.csv",transform=transform)
        train_data,test_data=myImbalancedDataset.train_data,myImbalancedDataset.test_data
        data_util.show_imbalance(myImbalancedDataset.train_df, save_path=f'imbalance_{myImbalancedDataset.average_train_samples}.png')
        train_loader=DataLoader(train_data,batch_size=64,shuffle=True,num_workers=16)
        test_loader=DataLoader(test_data,batch_size=64,shuffle=True,num_workers=16)
        rf = RandomForestClassifier(class_weight='balanced')
        knn = KNeighborsClassifier(weights='distance')
        estimators=[('rf', rf), ('knn', knn)]
        pipeline=data_util.init_pipeline_for_imbalance_scenario(estimators)
        X_train, y_train = data_util.feature_extraction(train_loader)
        X_test, y_test = data_util.feature_extraction(test_loader)
        print("Begin training...")
        pipeline.fit(X_train, y_train)
        print("Training finished.")
        y_pred = pipeline.predict(X_test)
        acc=accuracy_score(y_test, y_pred)
        f1=f1_score(y_test, y_pred, average='weighted')
        precision=precision_score(y_test, y_pred, average='weighted')
        recall=recall_score(y_test, y_pred, average='weighted')
        print("Accuracy:", acc)
        print("F1 Score:", f1)
        print("Precision:", precision)
        print("Recall:", recall)
        time_taken=time.time()-start_time
        print("Time taken:", time_taken)
        data_util.write_results_to_file("imbalance_results.txt", acc, f1, precision, recall,estimators,myImbalancedDataset.average_train_samples,100,time_taken)
        
