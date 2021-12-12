import torch
import torch.optim
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pydatagen
from pydatagen import dfutil
from pydatagen.functions import ascii_values
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

class DocDataset(torch.utils.data.TensorDataset):
    # required to perform a torch.save()
    def new_empty(self):
        return DocDataset(torch.as_tensor([0]).float(), torch.as_tensor([0]).float())
        
class DocPipeline(object):
    def __init__(self):
        self.vector_fn = partial(ascii_values, length=12, class_boost=['letter', 'number', 'special'], concat=True, multiplier=2)
        self.scaler = StandardScaler()
        
    def transform(self, df, feature_col, is_training=False):
        feature_df = df[feature_col]
        vectorized_feature_df = dfutil.vectorize(df, self.vector_fn, feature_col)
        if is_training:
            self.scaler.fit(vectorized_feature_df)
            
        scaled_feature_df = self.scaler.transform(vectorized_feature_df)
        return scaled_feature_df
        
    def create_training_data(self, df, feature_col, label_col, test_size=0.2, batch_size=256):
        label_df = df[label_col]
        scaled_feature_df = self.transform(df, feature_col, is_training=True)
        
        train_x_df, valid_x_df, train_y_df, valid_y_df = dfutil.split(scaled_feature_df, label_df, test_size=test_size)
        
        train_x_tensor = torch.as_tensor(train_x_df).float()
        valid_x_tensor = torch.as_tensor(valid_x_df).float()
        train_y_tensor = torch.as_tensor(train_y_df.values).float().unsqueeze(1)
        valid_y_tensor = torch.as_tensor(valid_y_df.values).float().unsqueeze(1)
        
        train_dataset = DocDataset(train_x_tensor, train_y_tensor)
        valid_dataset = DocDataset(valid_x_tensor, valid_y_tensor)
        
        traindl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validdl = DataLoader(valid_dataset, batch_size=batch_size)
        
        return traindl, validdl
        
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.vector_fn, f)
            pickle.dump(self.scaler, f)
            
    @classmethod
    def load(self, filename):
        pipeline = DocPipeline()
        with open(filename, "rb") as f:
            pipeline.vector_fn = pickle.load(f)
            pipeline.scaler = pickle.load(f)
            
        return pipeline
