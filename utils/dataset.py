import numpy as np
import os
import cv2
import csv
import pandas as pd
import time
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split

class MyImbalancedDataset:
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.train_df = pd.DataFrame(columns=self.df.columns)
        self.test_df = pd.DataFrame(columns=self.df.columns)

        total_train_samples = 0
        for label in self.df['label'].unique():
            now_df = self.df[self.df['label'] == label]

            train_size = np.random.randint(30, 101)
            total_train_samples += train_size

            train_data = now_df.sample(n=train_size, random_state=42)
            test_data = now_df.drop(train_data.index).sample(n=100, random_state=42)

            self.train_df = pd.concat([self.train_df, train_data])
            self.test_df = pd.concat([self.test_df, test_data])

        self.average_train_samples = total_train_samples // len(self.df['label'].unique())
        print(f"Average number of training samples per class: {self.average_train_samples}")

        self.train_data = MiniImageNet(self.df2list(self.train_df), transform=self.transform)
        self.test_data = MiniImageNet(self.df2list(self.test_df), transform=self.transform)

    def df2list(self, df):
        data = []
        for i in range(len(df)):
            data.append([df.iloc[i, 0], df.iloc[i, 1]])
        return data

class MyDataset:
    def __init__(self, csv_file,train_size=500,test_size=100, transform=None):
        self.df = pd.read_csv(csv_file)
        self.train_size=train_size
        self.test_size=test_size
        self.transform = transform
        self.train_df=pd.DataFrame(columns=self.df.columns)
        self.test_df=pd.DataFrame(columns=self.df.columns)
        for label in self.df['label'].unique():
            now_df = self.df[self.df['label'] == label]
            train_data = now_df.sample(n=self.train_size, random_state=42)
            test_data = now_df.drop(train_data.index).sample(n=self.test_size, random_state=42) 
            
            self.train_df = pd.concat([self.train_df, train_data])
            self.test_df = pd.concat([self.test_df, test_data])
        # self.train_df.to_csv(f'train_{self.train_size}.csv', index=False)
        # self.test_df.to_csv(f'test_{self.test_size}.csv', index=False)
        self.train_data = MiniImageNet(self.df2list(self.train_df), transform=self.transform)
        self.test_data = MiniImageNet(self.df2list(self.test_df), transform=self.transform)
    def df2list(self, df):
        data = []
        for i in range(len(df)):
            data.append([df.iloc[i, 0], df.iloc[i, 1]])
        return data
        
    

class MiniImageNet(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name, label = self.data[idx]
        img = self.executor.submit(self.load_image, file_name).result()
        label = int(label)
        return img, label

    def load_image(self, file_name):
        img = cv2.imread(file_name)
        new_img = cv2.resize(img, (64, 64))
        if self.transform:
            new_img = self.transform(new_img)
        return new_img
    

