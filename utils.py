# library
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

### Loss ###
class CosineLoss(nn.Module):
    def __init__(self,dim=1):
        super().__init__()
        self.dim = dim

    def neg_cos_sim(self,p,z):
        z = z.detach()
        p = F.normalize(p,dim=self.dim) # default : L2 norm
        z = F.normalize(z,dim=self.dim)
        return -torch.mean(torch.sum(p*z,dim=self.dim))
    
    def forward(self,p1,z2,p2,z1):
        L = self.neg_cos_sim(p1,z2)/2 + self.neg_cos_sim(p2,z1)/2
        return L
    

### metrics ###
def recall_at_k(pred,target,k):
    recall = []
    for i in range(len(pred)):
        inter = 0
        query = pred[i]
        y_q = target[i]
        idx = torch.argsort(query,dim=0).tolist()
        idx.reverse()
        idx = idx[0:k] # k개의 top index
        for j in idx:
            if y_q.tolist()[j] == 1:
                inter += 1
        recall.append(inter/len(torch.where(y_q==1)[0]))
    
    return sum(recall)/len(recall)


### learning scheduler ###
def adjust_learning_rate(optimizer, init_lr, epoch, args):
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.num_epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


### logging ###
def train_log(save_path, epoch, loss):
    with open(save_path + '/loss_log.txt', 'a') as f:
        f.write(str(epoch) + '   ' + str(loss) + '\n')


def draw_curve(work_dir, epoch_loss_dic):
    epoch_list = []
    loss_list = []
    for epoch, loss in epoch_loss_dic.items():
        epoch_list.append(epoch)
        loss_list.append(loss)
    
    plt.plot(epoch_list, loss_list, color='red', label="Train Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(work_dir + '/loss_curve.png')
    plt.close()
