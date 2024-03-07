from attacks.bim import BIM
from attacks.ssa import SSA
from attacks.naa import NAA
from attacks.pgd import PGD
from attacks.difgsm import DIFGSM
from attacks.tifgsm import TIFGSM
from attacks.mifgsm import MIFGSM
from attacks.sinifgsm import SINIFGSM
from attacks.gra import GRA
from attacks.igattack import IGAttack
from attacks.fsps import FSPS
from attacks.bim2 import BIMFX,BIMDBAORI,BIMDBAFX
from attacks.bim3 import BIM3
from attacks.pgn import PGN
from attacks.sia import SIA
from attacks.danaa import DANAA
from attacks.fgsra import FGSRA
from omegaconf import OmegaConf
import pretrainedmodels
import os
import torch
from torchvision import transforms as T
from loader import ImageNet,Normalize,TfNormalize
import torch.nn as nn
import copy
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image
from torch.autograd import Variable as V
from torchvision import transforms as T
from torchvision.models import maxvit_t,vit_b_16
from functools import partial
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
)
import sys

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    
setup_seed(2023)

class ReturnFirst(nn.Module):
    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            x = x[:,1:]
        return x

def get_model(net_name, model_dir):
    """Load converted model"""
    if isinstance(net_name, str):
        model_path = os.path.join(model_dir, net_name + '.npy')

        if net_name == 'tf_inception_v3':
            net = tf_inception_v3
        elif net_name == 'tf_inception_v4':
            net = tf_inception_v4
        elif net_name == 'tf_resnet_v2_50':
            net = tf_resnet_v2_50
        elif net_name == 'tf_resnet_v2_101':
            net = tf_resnet_v2_101
        elif net_name == 'tf_resnet_v2_152':
            net = tf_resnet_v2_152
        elif net_name == 'tf_inc_res_v2':
            net = tf_inc_res_v2
        elif net_name == 'tf_adv_inception_v3':
            net = tf_adv_inception_v3
        elif net_name == 'tf_ens3_adv_inc_v3':
            net = tf_ens3_adv_inc_v3
        elif net_name == 'tf_ens4_adv_inc_v3':
            net = tf_ens4_adv_inc_v3
        elif net_name == 'tf_ens_adv_inc_res_v2':
            net = tf_ens_adv_inc_res_v2
        else:
            print('Wrong model name!')
        rf = ReturnFirst().cuda().eval()
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),
            rf
        )
        return model
    else:
        return net_name

def verify(model_name, path, adv_dir, input_csv, batch_size=10,num_images=1000):

    model = get_model(model_name, path)

    X = ImageNet(adv_dir, input_csv, T.Compose([T.ToTensor()]), num_images=num_images)
    data_loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            try:
                sum += (model(images)[0].argmax(-1) != (gt+1)).detach().sum().cpu()
            except:
                sum += (model(images).argmax(-1) != (gt)).detach().sum().cpu()

    if isinstance(model_name, str):
        print(model_name + '  acu = {:.2%}'.format(sum / num_images))
    else:
        print("Torch Model" + '  acu = {:.2%}'.format(sum / num_images))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="configs/template.yaml")
parser.add_argument('--partition', type=int, default=0)
args = parser.parse_args()
config = OmegaConf.load(args.config)
PREFIX = os.path.basename(args.config)[:-5]
METHOD = eval(config.attack.method)
BATCH_SIZE = config.data.batch_size
EPS = config.attack.eps / 255
NUM_IMAGES = config.data.num_images
STEPS = config.attack.steps
MODEL = config.model.name

TRAIN_STEPS = config.model.train_config.train_steps
LEARING_RATE = config.model.train_config.learning_rate
SAVE_STEPS = config.attack.save_steps
ALPHA = EPS / STEPS
if "FGSRA" in args.config:
    RHO = config.attack.rho
    BETA = config.attack.beta
    MAX_ITER = config.attack.max_iter

# ALPHA = 1.6/255
OUTPUT_DIR = "output/"+PREFIX+"/"

# sys.stdout = open(f"logs/{PREFIX}.log", "w")
def partial_save_image(names):
    return partial(save_image,names=names)
transforms = T.Compose(
    [T.Resize(224 if MODEL in ['maxvit_t','vit_b_16'] else 299
            ), T.ToTensor()]
)

dataset = ImageNet("dataset/images","dataset/images.csv",transforms=transforms,num_images=NUM_IMAGES)


all_images = torch.load("all_images.pt") if not "vit" in MODEL else torch.load("all_images_224.pt")
all_labels = torch.load("all_labels.pt")
all_images_ID = torch.load("all_images_ID.pt")

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

if MODEL in ['maxvit_t','vit_b_16']:
    model = eval(MODEL)(pretrained=True).eval().to(device)
    model = torch.nn.Sequential(Normalize(mean, std),model).eval().to(device)
elif MODEL.startswith("tf_"):
    model = get_model(MODEL,"models").eval().to(device)
else:
    model = eval("pretrainedmodels."+MODEL)
    model = torch.nn.Sequential(Normalize(mean, std),model(num_classes=1000, pretrained='imagenet').eval().to(device)).eval().to(device)
# from torch.nn.parallel import DataParallel
# model = DataParallel(model)
if "FGSRA" in args.config:
    attack = METHOD(copy.deepcopy(model),eps=EPS,alpha=ALPHA,steps=STEPS,train_steps=TRAIN_STEPS,model_learning_rate=LEARING_RATE,rho=RHO,beta=BETA,max_iter=MAX_ITER)
else:
    attack = METHOD(copy.deepcopy(model),eps=EPS,alpha=ALPHA,steps=STEPS,train_steps=TRAIN_STEPS,model_learning_rate=LEARING_RATE)
if not "only" in args.config and not "bz10" in args.config:
    PARTATION = args.partition
    all_images_p = all_images[PARTATION*100:PARTATION*100+100]
    all_labels_p = all_labels[PARTATION*100:PARTATION*100+100]
    all_images_ID_p = all_images_ID[PARTATION*100:PARTATION*100+100]
    pbar = tqdm(total=100)
    for i in range(100):
    # for i in range(0,100,10):
        images = all_images_p[i].unsqueeze(0).to(device)
        gt = all_labels_p[i].unsqueeze(0).to(device)
        images_id = all_images_ID_p[i]
        
        if os.path.exists(OUTPUT_DIR[:-1]+"_9/"+images_id[0]):
            pbar.update(1)
            continue
        
        # images = all_images_p[i:i+10].to(device)
        # gt = all_labels_p[i:i+10].to(device)
        # images_id = [t[0] for t in all_images_ID_p[i:i+10]]

        adv_images = attack(images,gt,partial_save_image(images_id),save_steps=SAVE_STEPS,output_dir=OUTPUT_DIR)
        pbar.update(1)
    # for images, images_ID,  gt_cpu in dataloader:
    #     images = images.to(device)
    #     gt = gt_cpu.to(device)
    #     adv_images = attack(images,gt,partial_save_image(images_ID),save_steps=SAVE_STEPS,output_dir=OUTPUT_DIR)
    #     pbar.update(1)

    pbar.close()
else:
    pbar = tqdm(total=100)
    for i in range(100):
        all_images_p = all_images[i*10:i*10+10]
        all_labels_p = all_labels[i*10:i*10+10]
        all_images_ID_p = all_images_ID[i*10:i*10+10]
        all_images_ID_p = [t[0] for t in all_images_ID_p]
        skip = True
        for image_id in all_images_ID_p:
            if not os.path.exists(OUTPUT_DIR[:-1]+"_9/"+image_id):
                skip = False
                break
        if skip:
            pbar.update(1)
            continue
        images = all_images_p.to(device)
        gt = all_labels_p.to(device)
        adv_images = attack(images,gt,partial_save_image(all_images_ID_p),save_steps=SAVE_STEPS,output_dir=OUTPUT_DIR)
        pbar.update(1)
    pbar.close()