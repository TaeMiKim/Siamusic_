import torch
from torch.utils.data import DataLoader
import argparse
import os
from sklearn import metrics
from tqdm import tqdm

#local
from dataset import FMA_small, FMA_medium

parser = argparse.ArgumentParser()
parser.add_argument('--finetuned_model_path', default='./exp/finetune_pth/res50.pth', type=str)
parser.add_argument('--gpu_id', default='0, 1', type=str)

args = parser.parse_args()


def test(test_loader, model):
    print('-'*10 + 'test' + '-'*10)
    with torch.no_grad():
        model.eval()

        pred_list = []
        label_list = []
        for audio, label in tqdm(test_loader):
            label_list.append(label.item())
            audio, label = audio.type(torch.float32).cuda(), label.type(torch.LongTensor).cuda()
            
            pred = model(audio)
            pred = torch.argmax(torch.sigmoid(pred))
            pred_list.append(pred.cpu().item())

        print(metrics.confusion_matrix(label_list, pred_list))
        print(metrics.classification_report(label_list, pred_list, digits=3))

    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # test_data = FMA_medium(split='test')
    test_data = FMA_small(split='test')

    num_workers = 4 * torch.cuda.device_count()
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=True, num_workers=num_workers)

    model = torch.load(args.finetuned_model_path)
    model.cuda()
     
    test(test_loader, model)