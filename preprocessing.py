import numpy as np
import json
import os
import librosa
import pandas as pd
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings(action='ignore')
# parser
parser = argparse.ArgumentParser(description='Pre-processing')
parser.add_argument('--sr', type=int, default=22050, help='Sampling rate')
args = parser.parse_args()


def mp3_to_npy(from_dir, to_dir, audio_df, sr):
    path = './data/fma_medium/' + from_dir
    mp3_list = os.listdir(path)
    
    for mp3 in tqdm(mp3_list):
        mp3_name = mp3.split('.')[0]
        try:
            audio,_ = librosa.load(path + '/' + mp3, sr=sr)
            audio_dict = {'audio' : audio}
            np.save(to_dir + f'{mp3_name}.npy', audio_dict)
            audio_df = audio_df.append({'track' : mp3.strip('.mp3')},ignore_index=True)
        except:
            print(f'{mp3}은(는) 변환되지 않습니다.')
    audio_df.to_csv('./data/audio_df.csv')
    return audio_df


def main():
    try:
        os.mkdir('./data/fma_medium_npy')
    except:
        pass

    folder_list = os.listdir('./data/fma_medium')
    folder_list.remove('README.txt')
    folder_list.remove('checksums')

    audio_df = pd.DataFrame(columns = ['track'])
    for folder in folder_list:
        audio_df = mp3_to_npy(folder,'./data/fma_medium_npy/', audio_df, args.sr)
        

if __name__ == '__main__':
    main()