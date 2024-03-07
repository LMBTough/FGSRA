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
from attacks.fgsra import WXY
from torchvision.models import maxvit_t,vit_b_16
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

        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            T.Resize((299, 299)),
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
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
        print(model_name + ', {:.2%}'.format(sum / num_images))
    else:
        print("Torch Model" + ', {:.2%}'.format(sum / num_images))
        
        
def verify_vit(model_name, path, adv_dir, input_csv, batch_size=10,num_images=1000):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    model = eval(model_name)(pretrained=True).eval().to(device)
    model = torch.nn.Sequential(T.Resize((224, 224)),Normalize(mean, std),model).eval().to(device)
    X = ImageNet(adv_dir, input_csv, T.Compose([T.ToTensor()]), num_images=num_images)
    data_loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            sum += (model(images).argmax(-1) != (gt)).detach().sum().cpu()

    if isinstance(model_name, str):
        print(model_name + ', {:.2%}'.format(sum / num_images))
    else:
        print("Torch Model" + ', {:.2%}'.format(sum / num_images))

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
OUTPUT_DIR = "output/"+PREFIX+"/"

if not os.path.exists(f"logs/{PREFIX}.log"):
    sys.stdout = open(f"logs/{PREFIX}.log", "w")
    # 写入换行
    model_names = ['tf_inception_v3','tf_inception_v4','tf_inc_res_v2','tf_resnet_v2_50','tf_resnet_v2_101','tf_resnet_v2_152','tf_ens3_adv_inc_v3','tf_ens4_adv_inc_v3','tf_ens_adv_inc_res_v2','vit_b_16','maxvit_t']

    models_path = './models/'

    for step in SAVE_STEPS:
        for model_name in model_names:
            if not "vit" in model_name:
                verify(model_name, models_path, OUTPUT_DIR[:-1] + f"_{step}/", 'dataset/images.csv', batch_size=BATCH_SIZE, num_images=NUM_IMAGES)
            else:
                verify_vit(model_name, models_path, OUTPUT_DIR[:-1] + f"_{step}/", 'dataset/images.csv', batch_size=BATCH_SIZE, num_images=NUM_IMAGES)
else:
    f = open(f"logs/{PREFIX}.log", "r")
    lines = f.readlines()
    if len(lines) != 11:
        # print(1)
        # sys.stdout = open(f"logs/{PREFIX}.log", "w")
        # 追加写入
        sys.stdout = open(f"logs/{PREFIX}.log", "w")
        # 写入换行
        model_names = ['tf_inception_v3','tf_inception_v4','tf_inc_res_v2','tf_resnet_v2_50','tf_resnet_v2_101','tf_resnet_v2_152','tf_ens3_adv_inc_v3','tf_ens4_adv_inc_v3','tf_ens_adv_inc_res_v2','vit_b_16','maxvit_t']

        models_path = './models/'

        for step in SAVE_STEPS:
            for model_name in model_names:
                if not "vit" in model_name:
                    verify(model_name, models_path, OUTPUT_DIR[:-1] + f"_{step}/", 'dataset/images.csv', batch_size=BATCH_SIZE, num_images=NUM_IMAGES)
                else:
                    verify_vit(model_name, models_path, OUTPUT_DIR[:-1] + f"_{step}/", 'dataset/images.csv', batch_size=BATCH_SIZE, num_images=NUM_IMAGES)
# else:
#     print(0)