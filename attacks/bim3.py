import torch
import torch.nn as nn
import copy
from attacks.attack import Attack
import numpy as np
import torch.nn.functional as F
import os
class BIM3(Attack):
    r"""
    BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    .. note:: If steps set to 0, steps will be automatically decided following the paper.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, model_learning_rate=0.0001,train_steps=[0,2,4,6,8]):
        super().__init__("BIM3", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps
        self.supported_mode = ['default', 'targeted']
        self.train_steps = train_steps
        self.model_learning_rate = model_learning_rate

    def forward(self, images, labels, save_func,save_steps,output_dir):
        r"""
        Overridden.
        """
        self._check_inputs(images)
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        model = copy.deepcopy(self.model)
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.model_learning_rate,weight_decay=0.01)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.model_learning_rate)
        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        ori_images = images.clone().detach()

        for i in range(self.steps):
            if i in self.train_steps:
                # The third line of pseudo code : Warm up
                optimizer.zero_grad()
                output_v3 = model(images)
                cost = F.cross_entropy(output_v3, labels)
                cost.backward()
                optimizer.step()
            # images_near = images + torch.rand_like(images).uniform_(-self.eps*3, self.eps*3)
            # images_near.requires_grad = True
            # outputs = model(images_near)
            
            images.requires_grad = True
            outputs = model(images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False,
                                       create_graph=False)[0]
            # grad = torch.autograd.grad(cost, images_near,
            #                            retain_graph=False,
            #                            create_graph=False)[0]
            grad_flatten = grad.flatten()
            mask = torch.zeros_like(grad_flatten)
            mask[abs(grad_flatten).topk(largest=False,k=int(len(grad_flatten)*0.4)).indices]=1
            grad_flatten *= mask
            grad = grad_flatten.view_as(grad)
            adv_images = images + self.alpha*grad.sign()
            a = torch.clamp(ori_images - self.eps, min=0)
            b = (adv_images >= a).float()*adv_images \
                + (adv_images < a).float()*a
            c = (b > ori_images+self.eps).float()*(ori_images+self.eps) \
                + (b <= ori_images + self.eps).float()*b
            images = torch.clamp(c, max=1).detach()
            
            
                
            if i in save_steps:
                adv_img_np = images.detach().cpu().numpy()
                adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
                save_func(images=adv_img_np,output_dir=output_dir[:-1]+f"_{i}/")


        return images
