import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import argparse
import os

#local
from dataset import FMA
from models.simsiam import Evaluator


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=0.0001, type=float)
parser.add_argument('--gpu_id', default='0, 1', type=str)
parser.add_argument('--simsiam_path', default='./exp/pretrain_pth/simsiam_res50.pth', type=str)
parser.add_argument('--save_model_path', default='./exp/finetune_pth/res50.pth', type=str)

args = parser.parse_args()


def finetuning(train_loader, model, criterion, optimizer):
    print('-'*10 + 'finetuning' + '-'*10)
    model.train()
    epoch_loss = 0

    for audio, label in train_loader:        
        audio = audio.type(torch.float32).cuda()
        label = label.type(torch.LongTensor).cuda()

        optimizer.zero_grad()
        pred = model(audio)

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        print('finetuning CE loss: ', np.around(loss.item(), 3))

        epoch_loss += loss.item()
    
    return epoch_loss


def validation(test_loader, model, criterion):
    print('-'*10 + 'validation' + '-'*10)
    with torch.no_grad():
        model.eval()
        epoch_loss = 0

        for audio, label in test_loader:
            audio, label = audio.type(torch.float32).cuda(), label.type(torch.LongTensor).cuda()
            
            pred = model(audio)
            loss = criterion(pred, label) 
            print('valid_loss: ', np.around(loss.item(), 3))

            epoch_loss += loss.item()

    return epoch_loss


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    train_data = FMA(split='validation')
    test_data = FMA(split='test')

    num_workers = 4 * torch.cuda.device_count()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    model = torch.load(args.simsiam_path)
    finetuned_model = Evaluator(model.encoder, num_classes=16)

    ### freeze all layers except last fc layer ###
    for name, param in finetuned_model.named_parameters():
        if name not in ['evaluator.0.weight', 'evaluator.0.bias', 'evaluator.2.weight', 'evaluator.2.bias',
        'evaluator.4.weight', 'evaluator.4.bias', 'evaluator.6.weight', 'evaluator.6.bias']:
            param.requires_grad = False
    
    # ### initialize the fc layer ###
    # for i in [0,2,4,6]:
    #     finetuned_model.evaluator[i].weight.data.normal_(mean=0.0, std=0.01)
    #     finetuned_model.evaluator[i].bias.data.zero_()

    finetuned_model.cuda()

    criterion = nn.CrossEntropyLoss()

    parameters = list(filter(lambda p: p.requires_grad, finetuned_model.parameters()))
    optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    milestones = [int(args.num_epochs/3), int(args.num_epochs/2)]
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.7)

    best_loss = 100
    for epoch in tqdm(range(args.num_epochs)):
        print('Epoch %d/%d' % (epoch+1, args.num_epochs))
        print('-'*10)
        
        train_epoch_loss = finetuning(train_loader, finetuned_model, criterion, optimizer)
        train_epoch_loss = train_epoch_loss/len(train_loader)

        valid_epoch_loss = validation(test_loader, finetuned_model, criterion)
        valid_epoch_loss = valid_epoch_loss/len(test_loader)

        print(f'Epoch {epoch+1:02}: | Finetuning CE Loss: {train_epoch_loss:.5f}',
              f'Valid CE Loss: {valid_epoch_loss:.5f}')
        
        if valid_epoch_loss < best_loss:
            best_loss = valid_epoch_loss
            torch.save(finetuned_model, args.save_model_path)
            print('Model saved!')
        
        scheduler.step()

if __name__ == '__main__':
    main()

        



