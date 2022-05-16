import torch
import numpy as np
import librosa
import random


#lambda 값이 매번 random하게 변화
def mixup_random_lambda(x, alpha=0.4, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    if lam < 0.5:
        lam = 0.5

    batch_size = x.shape[0]
    
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]

    return mixed_x


#lambda 값 고정
def mixup_fixed_lambda(x, lam=0.6, use_cuda=True):
    batch_size = x.shape[0]
    
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]

    return mixed_x


def pitch_shift(batch, n_steps, sr=22050):
    batch_shifted = librosa.effects.pitch_shift(batch, sr=sr, n_steps=n_steps)
    
    return batch_shifted


def time_stretch(batch, rate):
    batch_stretched = librosa.effects.time_stretch(batch, rate=rate)
    padded_samples = np.zeros(shape=batch.shape)
    window = batch_stretched[..., : batch.shape[-1]]
    actual_window_length = window.shape[-1] 
    padded_samples[..., :actual_window_length] = window
    fixed_length_batch = padded_samples

    return fixed_length_batch


def train_aug(batch, lam=None, sr=22050, p=0.4):
    n_steps = random.randint(-4, 4) # pitch_shift hyperparameter
    rate = random.uniform(0.6, 1.5) # time_stretch hyperparameter

    mix_p = random.random()   # mixup 적용 확률
    pitch_p = random.random() # pitch_shift 적용 확률
    time_p =  random.random() # time_stretch 적용 확률
    
    #확률이 hyperparameter p를 넘으면 각 augmentation 적용되도록 함. 
    if mix_p > p:
        # batch = mixup_fixed_lambda(batch, lam=lam, use_cuda=True)
        batch = mixup_random_lambda(batch, use_cuda=True)
    
    if pitch_p > p:
        batch = pitch_shift(batch.cpu().numpy(), n_steps=n_steps, sr=sr)
        batch = torch.from_numpy(batch)
    
    if time_p > p:
        batch = time_stretch(batch.cpu().numpy(), rate=rate)
        batch = torch.from_numpy(batch)

    return batch