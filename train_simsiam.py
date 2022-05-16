from torch.utils.data import DataLoader
import torch
import argparse
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

# local
from utils import adjust_learning_rate, train_log, draw_curve
from dataset import FMA_medium, FMA_small
from utils import CosineLoss
from models.simsiam import Siamusic, Evaluator
from augmentation import train_aug


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--optim', default='SGD', type=str, help='SGD or Adam')
parser.add_argument('--lr', default=0.05, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--fix_pred_lr', action='store_true',
                    help='Fix learning rate for the predictor')

parser.add_argument('--backbone', default='resnet50', type=str, help='Select the model among resnet50, resnet101, resnet152')
parser.add_argument('--dim', default=2048, type=int)
parser.add_argument('--pred_dim', default=512, type=int)
parser.add_argument('--sr', default=22050, type=int)
parser.add_argument('--n_fft', default=512, type=int)
parser.add_argument('--f_min', default=0.0, type=float)
parser.add_argument('--f_max', default=8000.0, type=float)
parser.add_argument('--n_mels', default=80, type=int)

parser.add_argument('--aug', default='basic')
parser.add_argument('--lam', default=0.6, help='mixup hyperparameter')
parser.add_argument('--aug_p', default=0.6, help='augmenatation probability')

parser.add_argument('--save_model_path', default='./exp/pth/simsiam.pth', type=str)
parser.add_argument('--log_path', default='./exp/log')

parser.add_argument('--gpu_id', default='0, 1', type=str)

args = parser.parse_args()

if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)


def train(train_loader, model, criterion, optimizer):
    print('-'*10 + 'training' + '-'*10)
    model.cuda()
    model.train()
    train_epoch_loss = 0 

    for audio, label in train_loader:
        if args.aug == 'basic':
            audio_aug1 = train_aug(audio, p=args.aug_p)
            audio_aug2 = train_aug(audio, p=args.aug_p)
        audio_aug1, audio_aug2 = audio_aug1.type(torch.float32).cuda(), audio_aug2.type(torch.float32).cuda()

        optimizer.zero_grad() 
        p1, z2, p2, z1 = model(audio_aug1, audio_aug2)

        train_loss = criterion.cuda()(p1, z2, p2, z1)
        train_loss.backward()
        optimizer.step()
        print('train_loss: ', np.around(train_loss.item(), 3))
        
        train_epoch_loss += train_loss.item()
    
    return train_epoch_loss


def validation(valid_loader, model, criterion):
    print('-'*10 + 'validation' + '-'*10)
    with torch.no_grad():
        model.eval()
        valid_epoch_loss = 0
        
        for audio, label in valid_loader:
            audio, label = audio.type(torch.float32).cuda(), label.type(torch.LongTensor).cuda()
            
            valid_pred = model(audio)
            valid_loss = criterion(valid_pred, label) 
            print('valid_loss: ', np.around(valid_loss.item(), 3))

            valid_epoch_loss += valid_loss.item()
        
        return valid_epoch_loss


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    ####### dataloader setting ##############################################################################
    # train_data = FMA_medium(split='training')
    # valid_data = FMA_medium(split='validation')
    train_data = FMA_small(split='training')
    valid_data = FMA_small(split='validation')

    num_workers = 4 * torch.cuda.device_count()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    ####### model setting ####################################################################################
    train_model = Siamusic(backbone=args.backbone, dim=args.dim, pred_dim=args.pred_dim, sr=args.sr, 
                    n_fft=args.n_fft, f_min=args.f_min, f_max=args.f_max, n_mels=args.n_mels)
    valid_model = Evaluator(train_model.encoder, 16, args.backbone, args.dim).cuda()

    ####### loss setting #####################################################################################
    train_criterion = CosineLoss()
    valid_criterion = nn.CrossEntropyLoss()

    ####### optimizer setting #################################################################################
    init_lr = args.lr * args.batch_size / 256
    
    if args.fix_pred_lr:
        optim_params = [{'params': train_model.encoder.parameters(), 'fix_lr': False},
                        {'params': train_model.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = train_model.parameters()

    if args.optim == 'SGD':
        optimizer = optim.SGD(optim_params, lr=init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(optim_params, lr=init_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #######training & validation#################################################################################
    best_loss = 1.0
    epoch_loss_dic = {}
    
    for epoch in tqdm(range(args.num_epochs)):
        print('Epoch %d/%d' % (epoch+1, args.num_epochs))
        print('-'*10)
        
        train_epoch_loss = train(train_loader, train_model, train_criterion, optimizer)
        train_epoch_loss = train_epoch_loss/len(train_loader)

        valid_epoch_loss = validation(valid_loader, valid_model, valid_criterion)
        valid_epoch_loss = valid_epoch_loss/len(valid_loader)

        adjust_learning_rate(optimizer, init_lr, epoch, args)

        train_log(args.log_path, str(epoch+1), np.around(train_epoch_loss, 3))
        epoch_loss_dic[epoch+1] = train_epoch_loss

        print(f'Epoch {epoch+1:02}: | Train Cosine Loss: {train_epoch_loss:.5f}',
              f'Valid CE Loss: {valid_epoch_loss:.5f}')
        
        if train_epoch_loss < best_loss:
            best_loss = train_epoch_loss
            torch.save(train_model, args.save_model_path)
            print('Model saved!')
    
    draw_curve(args.log_path, epoch_loss_dic)

if __name__ == '__main__':
    main()
