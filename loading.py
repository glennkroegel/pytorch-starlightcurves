'''
created_by: Glenn Kroegel
date: 3 August 2019

description: Create dataloaders to feed for training

'''
import pandas as pd
import numpy as np
import glob
from config import TRAIN_DATA, CV_DATA
from utils import pooling

import torch
import torch.nn.functional as F
import os
from torch.distributions import Bernoulli
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class GaiaDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.sources = list(data.keys())

    def __getitem__(self, i):
        source = self.sources[i]
        data = self.data[source]
        data = torch.FloatTensor(data)
        data.transpose_(0,1)
        return source, data

    def __len__(self):
        return len(self.sources)

class GaiaLoaderFactory():
    '''Load Gaia time series data'''
    def __init__(self):
        self.path = 'gaia/'
        self.file = 'joined.csv'

    def generate(self, batch_size=1, train_size=0.9):
        df = pd.read_csv(os.path.join(self.path, self.file))
        sources = df['source_id'].unique()
        df = df.loc[df['band'] == 'G']
        df.drop('band', axis=1, inplace=True)
        df['time'] = df['time'].astype(np.float32)
        import pdb; pdb.set_trace()
        df = df.groupby('source_id')[['time','flux_over_error']].apply(lambda x: list(x.values.astype(np.float32)))
        data_dict = df.to_dict()
        L = int(train_size*len(sources))
        train_srcs = sources[:L]
        cv_srcs = sources[L:]
        train_data = {k:np.stack(v) for k,v in data_dict.items() if k in train_srcs}
        cv_data = {k:np.stack(v) for k,v in data_dict.items() if k in cv_srcs}
        train_ds = GaiaDataset(train_data)
        cv_ds = GaiaDataset(cv_data)
        train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=self.collate_ae)
        cv_loader = DataLoader(cv_ds, batch_size=batch_size, collate_fn=self.collate_ae)
        torch.save(train_loader, 'gaia_train.pt')
        torch.save(cv_loader, 'gaia_cv.pt')

    def collate_ae(self, batch):
        return batch

class TessDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

    def __getitem__(self, i):
        filename = self.files[i]
        data = np.load(self.files[i]).astype(np.float32)
        data = torch.FloatTensor(data)
        return filename, data

    def __len__(self):
        return len(self.files)

class TessLoaderFactory():
    '''Load Tess time series data in batches for DL models'''
    def __init__(self):
        self.path = 'tess/processed'
        self.files = glob.glob(os.path.join(self.path, '*.npy'))

    def generate(self, batch_size=1, train_size=0.9):
        L = int(train_size*len(self.files))
        train_files = self.files[:L]
        cv_files = self.files[L:]
        train_ds = TessDataset(train_files)
        cv_ds = TessDataset(cv_files)
        train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=self.collate_ae)
        cv_loader = DataLoader(cv_ds, batch_size=batch_size, collate_fn=self.collate_ae)
        torch.save(train_loader, 'tess_train.pt')
        torch.save(cv_loader, 'tess_cv.pt')

    def collate_ae(self, data, train_p=0.75):
        fname, batch = data
        batch = torch.stack(batch)
        bs = batch.size(0)
        sl = batch.size(2)
        ts = batch[:, 0]
        ys = batch[:, 1].unsqueeze(-1)
        p = torch.FloatTensor([train_p])
        mask_train = Bernoulli(p).sample(torch.Size([bs, sl]))
        y_train = ys * mask_train
        y_pred = ys
        batch_dict = {'observed_data': y_train.to(device), 
                      'observed_tp': ts[0].view(-1).to(device), 
                      'data_to_predict': y_pred.to(device), 
                      'tp_to_predict': ts[0].view(-1).to(device), 
                      'observed_mask': mask_train.to(device), 
                      'mask_predicted_data': None, 
                      'labels': None, 'mode': 'interp'}
        return batch_dict

    def collate_ar(self, batch, p=0.85):
        batch = torch.stack(batch)
        bs = batch.size(0)
        sl = batch.size(2)
        ts = batch[:, 0]
        ts = ts - ts[:,0].view(bs, 1)
        # ts = ts[0].squeeze() # check
        ys = batch[:, 1]
        L = int(p*sl)
        t_train = ts[:, :L]
        ys_train = ys[:, :L].unsqueeze(-1)
        mask_train = ~torch.isnan(ys_train)
        mask_train = mask_train.float()
        t_pred = ts[:, L:]
        y_pred = ys[:, L:].unsqueeze(-1)
        mask_pred = ~torch.isnan(y_pred)
        mask_pred = mask_pred.float()
        batch_dict = {'observed_data': ys_train.to(device), 
                      'observed_tp': t_train[0].view(-1).to(device), 
                      'data_to_predict': y_pred.to(device), 
                      'tp_to_predict': t_pred[0].view(-1).to(device), 
                      'observed_mask': mask_train.to(device), 
                      'mask_predicted_data': mask_pred.to(device), 
                      'labels': None, 'mode': 'extrap'}
        return batch_dict

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

    def gen_loaders(self, n_samples=2000, test_size=0.2, batch_size=200, pool=8):
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

    def gen_loaders(self, n_samples=3000, test_size=0.1, batch_size=50, pool=8):
        train_data = pd.read_csv(TRAIN_DATA, delimiter='\t', header=None)
        cv_data = pd.read_csv(CV_DATA, delimiter='\t', header=None)
        df = pd.concat([train_data, cv_data], axis=0, ignore_index=True)
        df = df.sample(n=n_samples)
        df.drop(0, axis=1, inplace=True)
        df_train, df_cv = train_test_split(df, test_size=test_size, random_state=42)
        train_data = df_train.values.astype(np.float32)
        cv_data = df_cv.values.astype(np.float32)
        if pool:
            train_data = pooling(train_data, (1,pool))
            cv_data = pooling(cv_data, (1,pool))
        train_set = VAEDataset(train_data)
        cv_set = VAEDataset(cv_data)
        train_loader = DataLoader(train_set, batch_size=batch_size)
        cv_loader = DataLoader(cv_set, batch_size=batch_size)
        torch.save(train_loader, 'vae_train_loader.pt')
        torch.save(cv_loader, 'vae_cv_loader.pt')

