import numpy as np
from lib.networks import make_network
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_network
import tqdm
import torch
from lib.visualizers import make_visualizer
from lib.config import cfg, args
from torch.autograd import Variable
import cv2
from lib.utils.pvnet import pvnet_config
from lib.utils import img_utils
import torch.nn as nn
from lib.utils import net_utils
import torch
import matplotlib.pyplot as plt
import os

# model loading
# P는 0으로 초기화 기존 이미지에 P를 더하고 이 2개를 더한 값을 input으로 넣는데 이때
# requires_grad=True 진행
# model 에 넣는다.
# model.zero_grad()
# loss.backward

mean = pvnet_config.mean
std = pvnet_config.std

eps = 0.2
alpha = 20.0 # 2.0, 3.0, 4.0, 5.0
iteration = 120

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

def to_cuda(batch):
    for k in batch:
        if k == 'meta':
            continue
        if k == 'path':
            continue
        if isinstance(batch[k], tuple):
            batch[k] = [b.cuda() for b in batch[k]]
        else:
            batch[k] = batch[k].cuda()
    return batch

def run_attack():

    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)

    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):

        batch = to_cuda(batch)
        batch_size = len(batch['inp'])

        path_list = []
        count = 0
        for list_idx in batch['path']:
            path = list_idx.split('/')
            path[3] = "JPEGImages_xyz"
            path_list[count] = '/'.join(path)
            lsit_check = '/'.join(path_list[:3])
            count += 1

        if not os.path.isdir(lsit_check):
            os.mkdir(lsit_check)

        input = Variable(batch['inp'], requires_grad=True)

        for i in range(iteration):
            network.zero_grad()
            output = network(input)

            # Loss init
            loss = 0

            weight = batch['mask'][:, None].float()
            vote_loss = torch.nn.functional.smooth_l1_loss(output['vertex'] * weight, batch['vertex'] * weight, reduction='sum')
            vote_loss = vote_loss / weight.sum() / batch['vertex'].size(1)
            loss += vote_loss * 1.4

            mask = batch['mask'].long()
            seg_loss = nn.CrossEntropyLoss()(output['seg'], mask)
            loss += seg_loss

            loss.backward()

            # input update Using gradient
            input_grad = input.grad
            input = input - alpha*input_grad
            input = where(input > batch['inp'] + eps, batch['inp'] + eps, input)
            input = where(input < batch['inp'] - eps, batch['inp'] - eps, input)

            input = Variable(input.data, requires_grad=True)

            # if i % 10 == 0 :
            #     print("* Step {}".format(i))
            #     Pertur = batch['inp'][0] - input[0]
            #     Pertur = Pertur.permute(1, 2, 0)
            #     Pertur_50 = Pertur.cpu().detach().numpy() * 50
            #     cv2.imwrite(lsit_check+"/Perturbation.jpg", Pertur_50*255.0)
        ###################################################################################

        for idx in range(batch_size):
            inp = img_utils.unnormalize_img(batch['inp'][0].cpu(), mean, std).permute(1, 2, 0)
            img = img_utils.unnormalize_img(input[0].cpu(), mean, std).permute(1, 2, 0)

            Pertur = batch['inp'][0] - input[0]
            Pertur = Pertur.permute(1,2,0)
            Pertur_50 = Pertur.cpu().detach().numpy() * 50

            inp = inp.detach().numpy()
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

            img = img.detach().numpy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            Add = cv2.add(Pertur_50, inp)

            vis = np.concatenate((inp, img, Pertur_50, Add), axis=1)

            cv2.imwrite(list, img * 255.0)

            cv2.imshow("original_image", vis)
            cv2.waitKey(10000)


            print("="*100)
            print("\n")


if __name__ =="__main__":
    globals()['run_' + args.type]()