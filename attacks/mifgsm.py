import torch
import torch.nn as nn
import copy
from attacks.attack import Attack
import numpy as np
import torch.nn.functional as F


class MIFGSM(Attack):
    r"""
    MI-FGSM in the paper 'Boosting Adversarial Attacks with Momentum'
    [https://arxiv.org/abs/1710.06081]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.MIFGSM(model, eps=8/255, steps=10, decay=1.0)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0, model_learning_rate=0.0001,train_steps=[0,2,4,6,8]):
        super().__init__("MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
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
        optimizer = torch.optim.SGD(model.parameters(), lr=self.model_learning_rate)
        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for i in range(self.steps):
            if i in self.train_steps:
                # The third line of pseudo code : Warm up
                optimizer.zero_grad()
                output_v3 = model(adv_images)
                cost = F.cross_entropy(output_v3, labels)
                cost.backward()
                optimizer.step()
                
            adv_images.requires_grad = True
            outputs = model(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            
            
                
            if i in save_steps:
                adv_img_np = adv_images.detach().cpu().numpy()
                adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
                save_func(images=adv_img_np,output_dir=output_dir[:-1]+f"_{i}/")
                
        return adv_images

