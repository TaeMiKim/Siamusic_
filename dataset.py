import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import json
import random

genre_dic_medium = np.load('./data/genre_dic_medium.npy', allow_pickle=True).item()
genre_dic_small = np.load('./data/genre_dic_small.npy', allow_pickle=True).item()


class FMA_medium(Dataset):
    def __init__(self, split, input_length=22050*5):
        self.split = split #training, validation, test
        self.input_length = input_length

        self.audio_path = './data/fma_npy'
        self.df = pd.read_csv('./data/tracks_medium.csv')
        self.tracks_df = self.df[self.df['split'] == self.split] 
    
    def __len__(self):
        return len(self.tracks_df)

    def __getitem__(self, idx):
        genre = self.tracks_df.iloc[idx]['genre_top']
        label = np.array(genre_dic_medium[genre])

        id = self.tracks_df.iloc[idx]['track_id']
        audio_id = '0'*(6-len(str(id))) + str(id)
        audio = np.load(f'{self.audio_path}/{audio_id}.npy', allow_pickle=True).item()['audio']
        audio_len = audio.shape[0]
        try:
            start = random.randint(0, audio_len-self.input_length)
            end = start + self.input_length
            audio = audio[start:end]
        except:
            audio = audio[0:self.input_length]

        return audio, label


class FMA_small(Dataset):
    def __init__(self, split, input_length=22050*5):
        self.split = split #training, validation, test
        self.input_length = input_length

        self.audio_path = './data/fma_npy'
        self.df = pd.read_csv('./data/tracks_small.csv')
        self.tracks_df = self.df[self.df['split'] == self.split] 
    
    def __len__(self):
        return len(self.tracks_df)

    def __getitem__(self, idx):
        genre = self.tracks_df.iloc[idx]['genre_top']
        label = np.array(genre_dic_small[genre])

        id = self.tracks_df.iloc[idx]['track_id']
        audio_id = '0'*(6-len(str(id))) + str(id)
        audio = np.load(f'{self.audio_path}/{audio_id}.npy', allow_pickle=True).item()['audio']
        audio_len = audio.shape[0]
        try:
            start = random.randint(0, audio_len-self.input_length)
            end = start + self.input_length
            audio = audio[start:end]
        except:
            audio = audio[0:self.input_length]

        return audio, label


