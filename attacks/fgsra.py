import torch
import torch.nn as nn
import scipy.stats as st
import torch.nn.functional as F
from attacks.attack import Attack
import numpy as np
import copy
from torch.autograd import Variable as V
from attacks.dct import *
import torch
import numpy as np
import scipy.stats as st
import torch.nn.functional as F

"""Translation-Invariant https://arxiv.org/abs/1904.02884"""
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
    return gaussian_kernel

"""Input diversity: https://arxiv.org/abs/1803.06978"""
def DI(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret

def DI_Resize(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    ret = padded if torch.rand(1) < diversity_prob else x
    ret = F.interpolate(ret, size=[img_size, img_size], mode='bilinear', align_corners=False)
    return ret

def clip_by_tensor(t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

T_kernel = gkern(7, 3)

class FGSRA(Attack):
    def __init__(self, model, eps=8/255,
                 alpha=2/255, steps=10,rho=0.5,beta=2.0, model_learning_rate=0.0001,train_steps=[0,2,4,6,8],max_iter=20):
        super().__init__("FGSRA", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_mode = ['default', 'targeted']
        self.momentum = 1.0
        self.beta = beta
        self.rho = rho
        self.max_iter = max_iter
        self.model_learning_rate = model_learning_rate
        self.train_steps = train_steps


    def forward(self, images, labels, save_func,save_steps,output_dir):
        m = torch.ones_like(images) * 10 / 9.4
        
        x = images.clone()
        images_min = clip_by_tensor(images - self.eps, 0.0, 1.0)
        images_max = clip_by_tensor(images + self.eps, 0.0, 1.0)
        model = copy.deepcopy(self.model)
        grad = torch.zeros_like(images)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.model_learning_rate)
        
        for i in range(self.steps):
            if i in self.train_steps:
                # The third line of pseudo code : Warm up
                optimizer.zero_grad()
                output_v3 = model(x)
                loss = F.cross_entropy(output_v3, labels)
                loss.backward()
                optimizer.step()
                
            x = V(x.detach(), requires_grad=True)
            output = model(x)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            current_gradient = x.grad.data
            avg_gradient = list()
            x_sims = list()
            for _ in range(self.max_iter):
                gauss = torch.rand_like(x) * 2 * (self.eps * self.beta) - self.eps * self.beta
                gauss = gauss.cuda()
                x_dct = dct_2d(x + gauss).cuda()
                mask = (torch.rand_like(x) * 2 * self.rho + 1 - self.rho).cuda()
                x_idct = idct_2d(x_dct * mask)
                x_idct = V(x_idct, requires_grad = True)

                # DI-FGSM https://arxiv.org/abs/1803.06978
                di_x_idct = DI_Resize(x_idct)
                output_v3 = model(di_x_idct)

                # output_v3 = model(x_idct)
                loss = F.cross_entropy(output_v3, labels)
                loss.backward()
                avg_gradient.append(x_idct.grad.data)
                # 计算x和di_x_idct的cosine similarity
                cossim = (x * di_x_idct).sum([1, 2, 3], keepdim=True) / (
                    torch.sqrt((x ** 2).sum([1, 2, 3], keepdim=True)) * torch.sqrt((di_x_idct ** 2).sum([1, 2, 3], keepdim=True)))
                x_sims.append(cossim)
            x_sims = torch.stack(x_sims, dim=1)
            x_sims = F.softmax(x_sims, dim=1)
            avg_gradient = torch.stack(avg_gradient, dim=1)
            avg_gradient = (avg_gradient * x_sims).sum(1)
            
            cossim = (current_gradient * avg_gradient).sum([1, 2, 3], keepdim=True) / (
                        torch.sqrt((current_gradient ** 2).sum([1, 2, 3], keepdim=True)) * torch.sqrt(
                    (avg_gradient ** 2).sum([1, 2, 3], keepdim=True)))
            current_gradient = cossim * current_gradient + (1 - cossim) * avg_gradient
            current_gradient = F.conv2d(current_gradient, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)
            current_gradient = current_gradient / torch.abs(current_gradient).mean([1, 2, 3], keepdim=True)
            current_gradient = self.momentum * grad + current_gradient
            eqm = (torch.sign(grad) == torch.sign(current_gradient)).float()
            grad = current_gradient
            dim = torch.ones_like(images)  - eqm
            m = m * (eqm + dim * 0.94)
            x = x + self.alpha * torch.sign(grad) * m
            x = clip_by_tensor(x, images_min, images_max)
            
            
            if i in save_steps:
                adv_img_np = x.detach().cpu().numpy()
                adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
                save_func(images=adv_img_np,output_dir=output_dir[:-1]+f"_{i}/")
                
        
        return x.detach()
