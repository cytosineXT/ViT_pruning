"""### Importing Libraries"""
import timm
import torch
from torch import nn, einsum
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from torch.optim import Adam, lr_scheduler
from torchvision import transforms

#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pickle
import os
from thop import profile
import time
import re

from timm.models.layers import DropPath, PatchEmbed, trunc_normal_, lecun_normal_
from deit_modified import  DynaLinear
from code.deit_modified_ghost_init import VisionTransformer
from functools import partial

from tqdm import tqdm

#try:
    #from google.colab import drive
    #drive.mount("/content/gdrive")
    #colab = True
#except:
    #colab = False

#helpers
#def show_torch_img(image):
    #plt.imshow(transforms.ToPILImage()(image))

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def soft_cross_entropy(predicts, targets):
    student_likelihood = nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = nn.functional.softmax(targets, dim=-1)
    return -torch.sum(targets_prob * student_likelihood, dim=-1).mean()


"""### Importance reordering"""

def compute_neuron_head_importance(
    eval_dataloader, model, n_layers, n_heads,device, loss_fn=nn.CrossEntropyLoss()
    ):
    """ This method shows how to compute:
        - neuron importance scores based on loss according to http://arxiv.org/abs/1905.10650
    """
    
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    head_mask = torch.ones(n_layers, n_heads).to(device)
    head_mask.requires_grad_(requires_grad=True)

    intermediate_weight = []
    intermediate_bias = []
    output_weight = []
    for name, w in model.named_parameters(): #这里好像是筛选层 但是怎么最后三个list还是空的呢 这里是筛选神经元的
        if 'intermediate' in name:
            if w.dim() > 1:
                intermediate_weight.append(w)
            else:
                intermediate_bias.append(w)

        if 'output' in name and 'attention' not in name:
            if w.dim() > 1:
                output_weight.append(w)
    
    neuron_importance = []
    for w in intermediate_weight:
        neuron_importance.append(torch.zeros(w.shape[0]).to(device))
    
    model.to(device)

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, label_ids = batch

        '''
        # calculate head importance
        outputs = model(input_ids, head_mask=head_mask)
        loss = loss_fn(outputs, label_ids)
        loss.backward()
        head_importance += head_mask.grad.abs().detach()
        '''
        # calculate head importance using Hessian (second-order derivative)
        outputs = model(input_ids, head_mask=head_mask)
        loss = loss_fn(outputs, label_ids)
        grads = torch.autograd.grad(loss, head_mask, create_graph=True)[0]
        for layer in range(n_layers):
            for head in range(n_heads):
                hessian = torch.autograd.grad(grads[layer, head], head_mask, retain_graph=True)[0]
                head_importance[layer, head] += hessian.abs().detach()

        # calculate  neuron importance
        for w1, b1, w2, current_importance in zip(intermediate_weight, intermediate_bias, output_weight, neuron_importance):
            current_importance += ((w1 * w1.grad).sum(dim=1) + b1 * b1.grad).abs().detach()
            current_importance += ((w2 * w2.grad).sum(dim=0)).abs().detach()
        break
    return head_importance, neuron_importance #好吧 实际上neuron_importance根本就没做，都是空的，在reorder_head_neuron.py里也把neuron相关的删掉了，太逗了

def reorder_neuron_head(model, head_importance, neuron_importance):

    model = model.module if hasattr(model, 'module') else model

    # reorder heads and ffn neurons
    for layer, current_importance in enumerate(neuron_importance):
        # reorder heads --> [layer][0]: attention module
        idx = torch.sort(head_importance[layer], descending=True)[-1]
        model.transformer.layers[layer][0].fn.reorder_heads(idx)
        # reorder neurons --> [layer][1]: feed-forward module
        idx = torch.sort(current_importance, descending=True)[-1]
        model.transformer.layers[layer][1].fn.reorder_intermediate_neurons(idx)
        model.transformer.layers[layer][1].fn.reorder_output_neurons(idx)

"""### Training"""

def train(
    model, train_data, eval_data, device,
    mode = "finetuning", width_list = None,
    weights_file = None, model_path = "./",
    loss_fn=nn.CrossEntropyLoss(), epochs=10,depth_list=None,
        **args
    ):
    assert mode in ["finetuning", "width", "depth"], "Wrong mode input"

    model.to(device)

    if weights_file is not None:
        model.load_state_dict(torch.load(weights_file))

    optimizer = Adam(model.parameters(), lr=1e-4)#优化器和调度器
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=50)

    if mode == "finetuning":
        train_model(model, path=model_path,
                    train_data=train_data, eval_data=eval_data,
                    epochs=epochs, loss_fn=loss_fn,
                    optimizer=optimizer, scheduler=scheduler, device=device
                    )
    
    if mode == "width":
            print("Training Distillation")
            # teacher_model = timm.create_model(args['model_architecture'], pretrained=False)
            # torch.save(model.state_dict(), args.model_path)
            teacher_model = VisionTransformer(
                img_size=args["img_size"],
                patch_size=args["patch_size"],
                num_classes=args["num_classes"],
                embed_dim=args["embed_dim"],
                depth=args["depth"],
                num_heads=args["num_heads"],
                mlp_ratio=args["mlp_ratio"],
                qkv_bias = args["qkv_bias"],
                in_chans=args["in_chans"],
                mha_width=args["mha_width"],
                mlp_width=args["mlp_width"],
                no_ghost=args["no_ghost"],
                ghost_mode=args["ghost_mode"],
            )
            # 加载预训练权重
            pretrained_weights = torch.load(model_path)

            # 创建一个新的字典，只包含模型中存在的权重
            new_weights = {}
            for key in teacher_model.state_dict().keys():
                if key in pretrained_weights:
                    new_weights[key] = pretrained_weights[key]

            # 将新权重加载到模型中
            teacher_model.load_state_dict(new_weights, strict=False)
            teacher_model.to(device)
            model.load_state_dict(new_weights, strict=False)

            # print(teacher_model.state_dict().keys())
            # teacher_model.load_state_dict(torch.load(model_path))
            head_importance, neuron_importance = compute_neuron_head_importance( #这里是计算神经元和头重要性的
                eval_data, model, args["depth"], args["heads"],
                loss_fn=loss_fn, device=device
                )
            reorder_neuron_head(model, head_importance, neuron_importance) #这里是将其按重要性排序的
            #width_list = sorted(width_list, reverse=True)
            for i, width in enumerate(tqdm(width_list, desc="Width", leave=False)):
                print(f"\nWidth: {width}")
                model.apply(lambda m: setattr(m, 'width_mult', width))
                path = os.path.join("../models", f"Width{width}_model_width_distillation.pt")
                train_distillation(
                    model, teacher_model = teacher_model,path=path,
                    train_data = train_data, eval_data = eval_data,
                    epochs = 1, device=device,
                    optimizer=optimizer, scheduler=scheduler,loss_fn=loss_fn
                )
                model.load_state_dict(new_weights, strict=False)
            print("Fine tuning after distillation")
            for i, width in enumerate(tqdm(width_list, desc="Width", leave=False)):
                path = os.path.join("../models", f"Width{width}_model_width_distillation.pt")
                print(f"\nWidth: {width}")
                model.apply(lambda m: setattr(m, 'width_mult', width))
                train_model(
                    model, path = path,
                    train_data=train_data, eval_data = eval_data,
                    epochs=epochs, loss_fn =loss_fn,
                    optimizer=optimizer, scheduler=scheduler, device=device
                )
                model.load_state_dict(torch.load(path))

    if mode == "depth":
        width_list = sorted(width_list, reverse=True)
        for i, width in enumerate(tqdm(width_list, desc="Width", leave=False)):
            teacher_model = VisionTransformer(
                img_size=args["img_size"],
                patch_size=args["patch_size"],
                num_classes=args["num_classes"],
                embed_dim=args["embed_dim"],
                depth=args["depth"],
                num_heads=args["num_heads"],
                mlp_ratio=args["mlp_ratio"],
                qkv_bias=args["qkv_bias"],
                in_chans=args["in_chans"],
                mha_width=args["mha_width"],
                mlp_width=args["mlp_width"],
                no_ghost=args["no_ghost"],
                ghost_mode=args["ghost_mode"],
            )
            teacher_model.apply(lambda m: setattr(m, 'width_mult', width))
            teacher_model.to(device)
            path = os.path.join("../models", f"Width{width}_model_width_distillation.pt")
            teacher_model.load_state_dict(torch.load(path))
            for i, depth in enumerate(tqdm(depth_list, desc="Depth", leave=False)):
                model.apply(lambda m: setattr(m, 'width_mult', width))
                model.apply(lambda m: setattr(m, 'depth', depth))
                path = os.path.join("../models", f"Width{width}_Depth{depth}_model_width_distillation.pt")
                print("Training Distillation")
                train_distillation(
                    model, teacher_model=teacher_model, path=path,
                    train_data=train_data, eval_data=eval_data,
                    epochs=epochs, device=device,
                    optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn
                )
                model.load_state_dict(torch.load(path))#为啥会报错找不到文件夹。。


def train_model(model, train_data, eval_data, path, epochs, loss_fn, optimizer, scheduler, device='cuda', **args):
    model.train()
    best_eval_loss = 1e8
    model.to(device)
    
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        total_loss = 0.0
        print(f"\nEpoch: {epoch}")

        for i, data in enumerate(tqdm(train_data, desc="Training", leave=False)):
            inputs, labels = tuple(t.to(device) for t in data)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*inputs.size(0)

        print(f"Train loss = {total_loss/len(train_data.sampler)}")

        model.eval()
        eval_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                eval_loss += loss.item()*inputs.size(0)

        eval_loss = eval_loss/len(eval_data.sampler)
        scheduler.step(metrics=eval_loss)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), path)
            best_eval_loss = eval_loss

        print(f"Validation loss = {eval_loss}")

def train_naive(model, train_data, eval_data, path, epochs, loss_fn, 
                        width_list, optimizer, scheduler, layernorm=False, device='cuda', **args):
    model.train()
    best_eval_loss = 1e8

    model.to(device)
    
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        total_loss = 0.0
        print(f"\nEpoch: {epoch}")

        for i, data in enumerate(tqdm(train_data, desc="Training", leave=False)):

            inputs, labels = tuple(t.to(device) for t in data)
            optimizer.zero_grad()
            width_list_loss = 0.0
            for j, width in enumerate(width_list):
                model.apply(lambda m: setattr(m, 'width_mult', width))
                if layernorm:
                    outputs = model(inputs, width_n=j)
                else:
                    outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                width_list_loss += loss.item()
                loss.backward()
            optimizer.step()

            total_loss += (width_list_loss/len(width_list))*inputs.size(0)
        
        print(f"Train loss = {total_loss/len(train_data.sampler)}")

        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                width_list_loss = 0.0
                for j, width in enumerate(width_list):
                    model.apply(lambda m: setattr(m, 'width_mult', width))
                    if layernorm:
                        outputs = model(inputs, width_n=j)
                    else:
                        outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    width_list_loss += loss.item()

                eval_loss += (width_list_loss/len(width_list))*inputs.size(0)
        
        eval_loss = eval_loss/len(eval_data.sampler)
        scheduler.step(metrics=eval_loss)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), path)
            best_eval_loss = eval_loss

        print(f"Validation loss = {eval_loss}")

def train_sandwich(model, train_data, eval_data, path, epochs, loss_fn,  
                        optimizer, scheduler, layernorm=False, width_min = 0.25, width_max = 1, n_widths=5 , device='cuda',
                   **args
                   ):
    model.train()
    best_eval_loss = 1e8
    
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        total_loss = 0.0
        print(f"\nEpoch: {epoch}")

        for i, data in enumerate(tqdm(train_data, desc="Training", leave=False)):

            inputs, labels = tuple(t.to(device) for t in data)
            optimizer.zero_grad()
            width_list_loss = 0.0
            width_list = list(np.random.choice(np.arange(256*width_min, 256*width_max), n_widths-2))
            width_list = [width_min] + width_list + [width_max]
            for j, width in enumerate(width_list):
                model.apply(lambda m: setattr(m, 'width_mult', width))
                if layernorm:
                    outputs = model(inputs, width_n=j)
                else:
                    outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                width_list_loss += loss.item()
                loss.backward()
            optimizer.step()

            total_loss += (width_list_loss/len(width_list))*inputs.size(0)
        
        print(f"Train loss = {total_loss/len(train_data.sampler)}")

        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                width_list_loss = 0.0
                for j, width in enumerate(width_list):
                    model.apply(lambda m: setattr(m, 'width_mult', width))
                    if layernorm:
                        outputs = model(inputs, width_n=j)
                    else:
                        outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    width_list_loss += loss.item()

                eval_loss += (width_list_loss/len(width_list))*inputs.size(0)
        
        eval_loss = eval_loss/len(eval_data.sampler)
        scheduler.step(metrics=eval_loss)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), path)
            best_eval_loss = eval_loss

        print(f"Validation loss = {eval_loss}")

def train_incremental(model, train_data, eval_data, path, 
                      epochs, loss_fn, optimizer, scheduler, 
                      freeze_width=None, device='cuda', **args):
    model.train()
    best_eval_loss = 1e8

    def zero_grad_dyna_linear(x, width):
        if isinstance(x, DynaLinear):
            x.set_grad_to_zero(width)
    
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        total_loss = 0.0
        print(f"\nEpoch: {epoch}")

        for i, data in enumerate(tqdm(train_data, desc="Training", leave=False)):

            inputs, labels = tuple(t.to(device) for t in data)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            
            # important step to not destroy previously learned parameters
            if freeze_width:
                model.apply(lambda x: zero_grad_dyna_linear(x, freeze_width))

            optimizer.step()

            total_loss += loss.item()*inputs.size(0)

        print(f"Train loss = {total_loss/len(train_data.sampler)}")

        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                eval_loss += loss.item()*inputs.size(0)

        eval_loss = eval_loss/len(eval_data.sampler)
        scheduler.step(metrics=eval_loss)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), path)
            best_eval_loss = eval_loss

        print(f"Validation loss = {eval_loss}")

def train_distillation(model, teacher_model, train_data, eval_data, path,
                      epochs, optimizer, scheduler, loss_fn, device='cuda',lambda1=0.5, lambda2=0.5,
                      **args):
    model.train()
    best_eval_loss = 1e8

    loss_mse = nn.MSELoss()

    def zero_grad_dyna_linear(x, width):
        if isinstance(x, DynaLinear):
            x.set_grad_to_zero(width)

    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        total_loss = 0.0
        print(f"\nEpoch: {epoch}")

        for i, data in enumerate(tqdm(train_data, desc="Training", leave=False)):
            inputs, labels = tuple(t.to(device) for t in data)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_out, teacher_hidden = teacher_model(inputs, return_states=True)
            student_out, student_hidden = model(inputs, return_states=True)
            loss1 = soft_cross_entropy(student_out, teacher_out.detach())
            loss2 = loss_mse(student_hidden, teacher_hidden.detach())

            loss = loss1*lambda1 + loss2*lambda2
            loss.backward()

            optimizer.step()

            total_loss += loss.item()*inputs.size(0)

        print(f"Train loss = {total_loss/len(train_data.sampler)}")

        # model.eval()
        # eval_loss = 0.0
        # correct = 0
        # total = 0

        # with torch.no_grad():
        #     for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
        #         inputs, labels = tuple(t.to(device) for t in data)
        #         start_time = time.time()
        #         outputs = model(inputs)
        #         end_time = time.time()
        #         loss = loss_fn(outputs, labels)
        #         eval_loss += loss.item()*inputs.size(0)
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()

        # eval_loss = eval_loss/len(eval_data.sampler)
        # scheduler.step(metrics=eval_loss)
        
        # print(f"Validation loss = {eval_loss}")
        # accuracy = 100 * correct / total
        # print(f'Accuracy of the  network on the test images: %d %%' % accuracy)
        # num_params = sum(p.numel() for p in model.parameters())
        # print('Number of parameters: %d' % num_params)
            
        # inference_time = end_time - start_time
        # print('Inference time: %.2fms' % (inference_time * 1000))
        # flops, params = profile(model, inputs=(inputs,))
        # print('Number of parameters: %d' % params)
        # print('FLOPS: %.2fG' % (flops / 1e9))

        # if eval_loss < best_eval_loss:
        #     torch.save(model.state_dict(), path)
        #     best_eval_loss = eval_loss
        
        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                eval_loss += loss.item()*inputs.size(0)

        eval_loss = eval_loss/len(eval_data.sampler)
        scheduler.step(metrics=eval_loss)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), path)
            best_eval_loss = eval_loss

        print(f"Validation loss = {eval_loss}")

        

def print_metrics(model, test_data, metric_funcs, device, loss_fn=None, width_list=None, width_switch=False):
    model.eval()
    model.to(device)

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if width_list:
        print(f"Width | {'Loss':^20}", end = "")
        for metric, args in metric_funcs:
            print(f" | {metric.__name__:^20}", end = "")
        print()
        for k, width in enumerate(width_list):
            print(f"{width:^5}", end = "")
            model.apply(lambda m: setattr(m, 'width_mult', width))
            preds = []
            truths = []
            
            total_loss = 0
            with torch.no_grad():
                for i, data in enumerate(test_data):
                    inputs, labels = tuple(t.to(device) for t in data)
                    if width_switch:
                        outputs = model(inputs, width_n=k)
                    else:
                        outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    total_loss += loss.item()*inputs.size(0)
                    preds = preds + list(
                        torch.argmax(
                            nn.functional.softmax(outputs.cpu(), dim=1), 
                            dim=1
                            )
                        )
                    truths = truths + list(labels.cpu())
            test_loss = total_loss/len(test_data.sampler)
            print(f" | {test_loss:^20.4f}", end = "")
            for metric, args in metric_funcs:
                perf = metric(truths, preds, **args)
                print(f" | {perf:^20.4f}", end = "")
            print()
    else:
        preds = []
        truths = []
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_data):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()*inputs.size(0)
                preds = preds + list(
                    torch.argmax(
                        nn.functional.softmax(outputs.cpu(), dim=1), 
                        dim=1
                        )
                    )
                truths = truths + list(labels.cpu())
        test_loss = total_loss/len(test_data.sampler)
        print(f"Loss: {test_loss}")
        for metric, args  in metric_funcs:
            perf = metric(truths, preds, **args)
            print(f"{metric.__name__}: {perf:^.4f}")

def print_accuracy(model, test_data, loss_fn=None, width_list=None, width_switch=False, device=None):
    model.eval()
    model.to(device)
    metric_funcs = [(accuracy_score, {})]

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if width_list:
        print(f"Width | {'Loss':^20}", end = "")
        for metric, args in metric_funcs:
            print(f" | {metric.__name__:^20}", end = "")
        print()
        for k, width in enumerate(width_list):
            print(f"{width:^5}", end = "")
            model.apply(lambda m: setattr(m, 'width_mult', width))
            preds = []
            truths = []
            
            total_loss = 0
            with torch.no_grad():
                for i, data in enumerate(test_data):
                    inputs, labels = tuple(t.to(device) for t in data)
                    if width_switch:
                        outputs = model(inputs, width_n=k)
                    else:
                        outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    total_loss += loss.item()*inputs.size(0)
                    preds = preds + list(
                        torch.argmax(
                            nn.functional.softmax(outputs.cpu(), dim=1), 
                            dim=1
                            )
                        )
                    truths = truths + list(labels.cpu())
            test_loss = total_loss/len(test_data.sampler)
            print(f" | {test_loss:^20.4f}", end = "")
            for metric, args in metric_funcs:
                perf = metric(truths, preds, **args)
                print(f" | {perf:^20.4f}", end = "")
            print()
    else:
        preds = []
        truths = []
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_data):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()*inputs.size(0)
                preds = preds + list(
                    torch.argmax(
                        nn.functional.softmax(outputs.cpu(), dim=1), 
                        dim=1
                        )
                    )
                truths = truths + list(labels.cpu())
        test_loss = total_loss/len(test_data.sampler)
        print(f"Loss: {test_loss}")
        for metric, args  in metric_funcs:
            perf = metric(truths, preds, **args)
            print(f"{metric.__name__}: {perf:^.4f}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
