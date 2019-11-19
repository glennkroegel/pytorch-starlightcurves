'''
created_by: Glenn Kroegel
date: 3 August 2019

description: Create dataloaders to feed for training

'''
import pandas as pd
import numpy as np
from config import TRAIN_DATA, CV_DATA
from utils import pooling

import torch
import os
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class TSDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)

    def __getitem__(self, i):
        x = self.x[i]
        y = self.y[i]
        return x, y
    
    def __len__(self):
        return len(self.x)

class VAEDataset(torch.utils.data.Dataset):
    def __init__(self, x):
        self.x = torch.FloatTensor(x)

    def __getitem__(self, i):
        x = self.x[i]
        return x
    
    def __len__(self):
        return len(self.x)

class DataLoaderFactory():
    '''Standard dataloaders with regular collating/sampling from padded dataset'''
    def __init__(self, train_path=TRAIN_DATA, cv_path=CV_DATA):
        pass

    def gen_loaders(self, batch_size=50):
        train_data = pd.read_csv(TRAIN_DATA, delimiter='\t', header=None)
        cv_data = pd.read_csv(CV_DATA, delimiter='\t', header=None)
        train_y = train_data[0]-1
        cv_y = cv_data[0]-1
        train_data.drop(0, axis=1, inplace=True)
        cv_data.drop(0, axis=1, inplace=True)
        train_data = train_data.values.astype(np.float32)
        cv_data = cv_data.values.astype(np.float32)
        train_set = TSDataset(train_data, train_y)
        cv_set = TSDataset(cv_data, cv_y)
        train_loader = DataLoader(train_set, batch_size=batch_size)
        cv_loader = DataLoader(cv_set, batch_size=batch_size)
        torch.save(train_loader, 'train_loader.pt')
        torch.save(cv_loader, 'cv_loader.pt')

    def collate_fn(self, batch):
        xs = torch.stack([x[0] for x in batch])
        ys = torch.stack([x[1] for x in batch])
        return xs, ys

    def save_loaders(self):
        torch.save(self.train_loader, 'train_loader.pt')
        torch.save(self.cv_loader, 'cv_loader.pt')

class BalancedDataLoaderFactory():
    '''Dataloader generation for balanced classes given a sample size selection'''
    def __init__(self, train_path=TRAIN_DATA, cv_path=CV_DATA):
        pass

    def gen_loaders(self, n_samples=1500, test_size=0.1, batch_size=16, pool=None):
        train_data = pd.read_csv(TRAIN_DATA, delimiter='\t', header=None)
        cv_data = pd.read_csv(CV_DATA, delimiter='\t', header=None)
        df = pd.concat([train_data, cv_data], axis=0, ignore_index=True)
        df.rename({0: 'y'}, inplace=True, axis=1)
        d = dict(df['y'].value_counts())
        class_cnt = n_samples // 3
        df1 = df.loc[df['y']==1].sample(n=min(class_cnt, d[1]))
        df2 = df.loc[df['y']==2].sample(n=min(class_cnt, d[2]))
        df3 = df.loc[df['y']==3].sample(n=min(class_cnt, d[3]))
        df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
        df['y'] = df['y'] - 1
        df_train, df_cv = train_test_split(df, test_size=test_size, stratify=df['y'], random_state=42)
        print(df_train['y'].value_counts(), df_cv['y'].value_counts())
        train_y = df_train['y'].values
        cv_y = df_cv['y'].values
        df_train.drop('y', axis=1, inplace=True)
        df_cv.drop('y', axis=1, inplace=True)
        train_data = df_train.values.astype(np.float32)
        cv_data = df_cv.values.astype(np.float32)
        if pool:
            train_data = pooling(train_data, (1,pool))
            cv_data = pooling(cv_data, (1,pool))
        train_set = TSDataset(train_data, train_y)
        cv_set = TSDataset(cv_data, cv_y)
        train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
        cv_loader = DataLoader(cv_set, batch_size=batch_size, drop_last=True)
        torch.save(train_loader, 'train_loader.pt')
        torch.save(cv_loader, 'cv_loader.pt')

    def collate_fn(self, batch):
        xs = torch.stack([x[0] for x in batch])
        ys = torch.stack([x[1] for x in batch])
        return xs, ys

    def save_loaders(self):
        torch.save(self.train_loader, 'train_loader.pt')
        torch.save(self.cv_loader, 'cv_loader.pt')


class VAEDataLoaderFactory():
    '''Dataloader generation for balanced classes given a sample size selection'''
    def __init__(self, train_path=TRAIN_DATA, cv_path=CV_DATA):
        pass

    def gen_loaders(self, n_samples=3000, test_size=0.1, batch_size=50):
        train_data = pd.read_csv(TRAIN_DATA, delimiter='\t', header=None)
        cv_data = pd.read_csv(CV_DATA, delimiter='\t', header=None)
        df = pd.concat([train_data, cv_data], axis=0, ignore_index=True)
        df = df.sample(n=n_samples)
        df.drop(0, axis=1, inplace=True)
        df_train, df_cv = train_test_split(df, test_size=test_size, random_state=42)
        train_data = df_train.values.astype(np.float32)
        cv_data = df_cv.values.astype(np.float32)
        train_set = VAEDataset(train_data)
        cv_set = VAEDataset(cv_data)
        train_loader = DataLoader(train_set, batch_size=batch_size)
        cv_loader = DataLoader(cv_set, batch_size=batch_size)
        torch.save(train_loader, 'vae_train_loader.pt')
        torch.save(cv_loader, 'vae_cv_loader.pt')

