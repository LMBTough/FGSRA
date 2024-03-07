# %%
import numpy as np
from attacks.attack import Attack
import torch.optim as optim
# %%
import torch
import torch.nn as nn
import copy
import numpy as np
import torch.nn.functional as F
from copy import deepcopy

class FISHER_OPERATION(object):
        def __init__(self, input_data, network, vector, epsilon = 1e-3):
                self.input = input_data
                self.network = network
                self.vector = vector
                self.epsilon = epsilon

        # Computes the fisher matrix quadratic form along the specific vector
        def fisher_quadratic_form(self):
                fisher_sum = 0
                network = deepcopy(self.network)
                for name,parameter in network.named_parameters():
                    parameter.data += self.epsilon * self.vector[name]
                log_softmax_output1 = network(self.input)
                softmax_output1 = F.softmax(log_softmax_output1, dim=1)
                for name,parameter in network.named_parameters():
                    parameter.data -= 2 * self.epsilon * self.vector[name]
                log_softmax_output2 = network(self.input)
                softmax_output2 = F.softmax(log_softmax_output2, dim=1)
                # fisher_sum += (((log_softmax_output1 - log_softmax_output2)/(2 * self.epsilon))*((softmax_output1 - softmax_output2)/(2 * self.epsilon))).sum()
                # fisher_sum += (((log_softmax_output1 - log_softmax_output2))*((softmax_output1 - softmax_output2))).sum()
                fisher_sum += (((log_softmax_output1 - log_softmax_output2) / (2 * self.epsilon))**2 * ((softmax_output1 - softmax_output2) / (2 * self.epsilon))**2).sum()

                return fisher_sum
            
            
class FISHER_OPERATION_OTHER_LABEL_DECREASE(object):
        def __init__(self, input_data, network, vector, epsilon = 1e-3):
                self.input = input_data
                self.network = network
                self.vector = vector
                self.epsilon = epsilon

        # Computes the fisher matrix quadratic form along the specific vector
        def fisher_quadratic_form(self,labels):
                fisher_sum = 0
                network = deepcopy(self.network)
                for name,parameter in network.named_parameters():
                    parameter.data += self.epsilon * self.vector[name]
                log_softmax_output1 = network(self.input)
                softmax_output1 = F.softmax(log_softmax_output1, dim=1)
                for name,parameter in network.named_parameters():
                    parameter.data -= 2 * self.epsilon * self.vector[name]
                log_softmax_output2 = network(self.input)
                softmax_output2 = F.softmax(log_softmax_output2, dim=1)
                mask = torch.zeros_like(softmax_output1)
                mask[:,labels] = 2
                mask -= 1
                fisher_sum += (((log_softmax_output1 - log_softmax_output2) * mask)*((softmax_output1 - softmax_output2))).sum()
                return fisher_sum

class FISHER_OPERATION_X(object):
        def __init__(self, input_data, network, vector, epsilon = 1e-3):
                self.input = input_data
                self.network = network
                self.vector = vector
                self.epsilon = epsilon

        # Computes the fisher matrix quadratic form along the specific vector
        def fisher_quadratic_form(self):
                fisher_sum = 0
                network = deepcopy(self.network)
                x = deepcopy(self.input)
                x = x + self.epsilon * self.vector
                log_softmax_output1 = network(x)
                softmax_output1 = F.softmax(log_softmax_output1, dim=1)
                x = x - 2 * self.epsilon * self.vector
                log_softmax_output2 = network(x)
                softmax_output2 = F.softmax(log_softmax_output2, dim=1)
                # fisher_sum += (((log_softmax_output1 - log_softmax_output2)/(2 * self.epsilon))*((softmax_output1 - softmax_output2)/(2 * self.epsilon))).sum()
                # fisher_sum += (((log_softmax_output1 - log_softmax_output2))*((softmax_output1 - softmax_output2))).sum()
                # (((log_softmax_output1 - log_softmax_output2) / (2 * self.epsilon))**2 * ((softmax_output1 - softmax_output2) / (2 * self.epsilon))**2).sum()
                fisher_sum += (((log_softmax_output1 - log_softmax_output2) / (2 * self.epsilon))**2 * ((softmax_output1 - softmax_output2) / (2 * self.epsilon))**2).sum()
                return fisher_sum
# %%
class BIMFX(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, model_learning_rate=0.00001,train_steps=[0,2,4,6,8]):
        super().__init__("BIMFX", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps
        self.supported_mode = ['default', 'targeted']
        self.train_steps = train_steps
        self.model_learning_rate = model_learning_rate
        
    def get_gd_model(self,model,images,labels):
        model = copy.deepcopy(model)
        model.eval()
        images = images.clone().detach()
        labels = labels.clone().detach()
        optimizer = optim.SGD(model.parameters(), lr=self.model_learning_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        return model
    
    def get_ga_model(self,model,images,labels):
        model = copy.deepcopy(model)
        model.eval()
        images = images.clone().detach()
        labels = labels.clone().detach()
        optimizer = optim.SGD(model.parameters(), lr=self.model_learning_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer.zero_grad()
        loss = -criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        return model
    
    def get_x_fx(self,model,images,labels):
        model = copy.deepcopy(model)
        epsilon = 1e-3
        images = images.clone().detach().requires_grad_(True)
        outputs = model(images)
        cost = outputs.max()
        vector = torch.autograd.grad(cost, images,create_graph=False, retain_graph=False)[0]
            
        FISHER = FISHER_OPERATION_X(images, model, vector, epsilon)
        return FISHER.fisher_quadratic_form().item()
    
    def get_fx(self,model,images,labels):
        model = copy.deepcopy(model)
        epsilon = 1e-3
        # criterion = nn.CrossEntropyLoss()
        outputs = model(images)
        # cost = criterion(outputs, labels)
        # cost.backward()
        cost = outputs.max()
        cost.backward()
        vector = {}
        for name,parameter in model.named_parameters():
            if parameter.grad is not None:
                vector[name] = parameter.grad.clone()
                parameter.grad.zero_()
            else:
                vector[name] = torch.zeros_like(parameter)
            
        FISHER = FISHER_OPERATION(images, model, vector, epsilon)
        return FISHER.fisher_quadratic_form().item()
    
    def get_fx_other_label_decrease(self,model,images,labels):
        model = copy.deepcopy(model)
        epsilon = 1e-3
        # criterion = nn.CrossEntropyLoss()
        outputs = model(images)
        # cost = criterion(outputs, labels)
        # cost.backward()
        cost = outputs.max()
        cost.backward()
        vector = {}
        for name,parameter in model.named_parameters():
            if parameter.grad is not None:
                vector[name] = parameter.grad.clone()
                parameter.grad.zero_()
            else:
                vector[name] = torch.zeros_like(parameter)
            
        FISHER = FISHER_OPERATION_OTHER_LABEL_DECREASE(images, model, vector, epsilon)
        return FISHER.fisher_quadratic_form(labels).item()

    def forward(self, images, labels, save_func,save_steps,output_dir):
        r"""
        Overridden.
        """
        self._check_inputs(images)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        model = copy.deepcopy(self.model)
        ori_model = copy.deepcopy(self.model)

        loss = nn.CrossEntropyLoss()
        ori_images = images.clone().detach()

        for i in range(self.steps):
            gd_model = self.get_gd_model(ori_model,images,labels)
            fx1 = self.get_fx_other_label_decrease(ori_model,images,labels)
            fx2 = self.get_fx_other_label_decrease(gd_model,images,labels)
            # fx1 = self.get_x_fx(ori_model,images,labels)
            # fx2 = self.get_x_fx(gd_model,images,labels)
            if fx1 > fx2:
                # model = self.get_ga_model(ori_model,images,labels)
                model = ori_model
            else:
                model = copy.deepcopy(gd_model)
            images.requires_grad = True
            outputs = model(images)

            # Calculate loss
            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False,
                                       create_graph=False)[0]
            
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


# %%
class BIMDBAORI(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, model_learning_rate=0.00001,train_steps=[0,2,4,6,8]):
        super().__init__("BIMDBAORI", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps
        self.supported_mode = ['default', 'targeted']
        self.train_steps = train_steps
        self.model_learning_rate = model_learning_rate
        
    def get_gd_model(self,model,images,labels):
        model = copy.deepcopy(model)
        model.eval()
        images = images.clone().detach()
        labels = labels.clone().detach()
        optimizer = optim.SGD(model.parameters(), lr=self.model_learning_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        return model
        

    def forward(self, images, labels, save_func,save_steps,output_dir):
        r"""
        Overridden.
        """
        self._check_inputs(images)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        model = copy.deepcopy(self.model)
        ori_model = copy.deepcopy(self.model)
        loss = nn.CrossEntropyLoss()
        ori_images = images.clone().detach()

        for i in range(self.steps):
            
            model = self.get_gd_model(ori_model,images,labels)
            
            images.requires_grad = True
            outputs = model(images)

            # Calculate loss
            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False,
                                       create_graph=False)[0]
            
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

# %%
class BIMDBAFX(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, model_learning_rate=0.00001,train_steps=[0,2,4,6,8]):
        super().__init__("BIMDBAFX", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps
        self.supported_mode = ['default', 'targeted']
        self.train_steps = train_steps
        self.model_learning_rate = model_learning_rate
        
    def get_gd_model(self,model,images,labels):
        model = copy.deepcopy(model)
        model.eval()
        images = images.clone().detach()
        labels = labels.clone().detach()
        optimizer = optim.SGD(model.parameters(), lr=self.model_learning_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        return model
    
    
    
    def get_ga_model(self,model,images,labels):
        model = copy.deepcopy(model)
        model.eval()
        images = images.clone().detach()
        labels = labels.clone().detach()
        optimizer = optim.SGD(model.parameters(), lr=self.model_learning_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer.zero_grad()
        loss = -criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        return model
    
    def get_x_fx(self,model,images,labels):
        model = copy.deepcopy(model)
        epsilon = 1e-3
        images = images.clone().detach().requires_grad_(True)
        outputs = model(images)
        cost = outputs.max()
        vector = torch.autograd.grad(cost, images,create_graph=False, retain_graph=False)[0]
            
        FISHER = FISHER_OPERATION_X(images, model, vector, epsilon)
        return FISHER.fisher_quadratic_form().item()
    
    def get_fx(self,model,images,labels):
        model = copy.deepcopy(model)
        epsilon = 1e-4
        # criterion = nn.CrossEntropyLoss()
        outputs = model(images)
        # cost = criterion(outputs, labels)
        # cost.backward()
        cost = outputs.max()
        cost.backward()
        vector = {}
        for name,parameter in model.named_parameters():
            if parameter.grad is not None:
                vector[name] = parameter.grad.clone()
                parameter.grad.zero_()
            else:
                vector[name] = torch.zeros_like(parameter)
            
        FISHER = FISHER_OPERATION(images, model, vector, epsilon)
        return FISHER.fisher_quadratic_form().item()

    def forward(self, images, labels, save_func,save_steps,output_dir):
        r"""
        Overridden.
        """
        self._check_inputs(images)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        model = copy.deepcopy(self.model)

        loss = nn.CrossEntropyLoss()
        ori_images = images.clone().detach()

        for i in range(self.steps):
            ga_model = self.get_ga_model(model,images,labels)
            gd_model = self.get_gd_model(model,images,labels)
            temp_image = images.clone().detach().requires_grad_(True)
            ga_outputs = ga_model(temp_image)
            cost = loss(ga_outputs, labels)
            grad = torch.autograd.grad(cost, temp_image,
                                        retain_graph=False,
                                        create_graph=False)[0]
            ga_grad_norm = grad.norm()
            gd_outputs = gd_model(temp_image)
            cost = loss(gd_outputs, labels)
            grad = torch.autograd.grad(cost, temp_image,
                                        retain_graph=False,
                                        create_graph=False)[0]
            gd_grad_norm = grad.norm()
            if ga_grad_norm < gd_grad_norm:
                model = ga_model
            else:
                model = gd_model
            # fx1 = self.get_fx(model,images,labels)
            # fx2 = self.get_fx(gd_model,images,labels)
            # fx1 = self.get_x_fx(model,images,labels)
            # fx2 = self.get_x_fx(gd_model,images,labels)
            # if fx1 < fx2:
            #     model = model
            #     # model = self.get_ga_model(model,images,labels)
            # else:
            #     model = copy.deepcopy(gd_model)
            images.requires_grad = True
            outputs = model(images)

            # Calculate loss
            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False,
                                       create_graph=False)[0]
            
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