import torch
import os
# from torch.datasets import load_and_cache_examples, SequentialSampler, DataLoader
from tqdm import tqdm
from torch import nn

def compute_neuron_head_importance(model, eval_dataloader, num_layers=12, num_heads=6, device='cuda', loss_fn=nn.CrossEntropyLoss()):
    """This method shows how to compute:
    - neuron importance scores based on loss according to http://arxiv.org/abs/1905.10650
    """
    # prepare things for heads
    # model = model.module if hasattr(model, "module") else model
    # base_model = getattr(model, model.base_model_prefix, model)
    n_layers, n_heads = (num_layers, num_heads)
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    head_mask = torch.ones(n_layers, n_heads).to(device)
    head_mask.requires_grad_(requires_grad=True)

    # collect weights
    # TODO: neuron importance
    # intermediate_weight = []
    # intermediate_bias = []
    # output_weight = []
    # for name, w in model.named_parameters():
    #     if "intermediate" in name:
    #         if w.dim() > 1:
    #             intermediate_weight.append(w)
    #         else:
    #             intermediate_bias.append(w)

    #     if "output" in name and "attention" not in name:
    #         if w.dim() > 1:
    #             output_weight.append(w)

    # neuron_importance = []
    # for w in intermediate_weight:
    #     neuron_importance.append(torch.zeros(w.shape[0]).to(args.device))

    model.to(device)

    # eval_task_names = (
    #     ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    # )
    # eval_outputs_dirs = (
    #     (args.output_dir, args.output_dir + "MM")
    #     if args.task_name == "mnli"
    #     else (args.output_dir,)
    # )
    # eval_dataset = load_and_cache_examples(
    #     args, eval_task, tokenizer, evaluate=True
    # )

    # if not os.path.exists(eval_output_dir):
    #     os.makedirs(eval_output_dir)

    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # eval_sampler = SequentialSampler(eval_dataset)
    # eval_dataloader = DataLoader(
    #     eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    # )

    for data in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = tuple(t.to(device) for t in data)

        # calculate head importance
        outputs = model(
            inputs,
            head_mask=head_mask,
        )
        loss = loss_fn(outputs, labels)
        loss.backward()
        head_importance += head_mask.grad.abs().detach()

        # calculate  neuron importance
        # for w1, b1, w2, current_importance in zip(
        #     intermediate_weight, intermediate_bias, output_weight, neuron_importance
        # ):
        #     current_importance += (
        #         ((w1 * w1.grad).sum(dim=1) + b1 * b1.grad).abs().detach()
        #     )
        #     current_importance += ((w2 * w2.grad).sum(dim=0)).abs().detach()

    return head_importance

def reorder_neuron_head(model, head_importance):
    """ reorder neurons based on their importance.
        Arguments:
            model: bert model
            head_importance: 12*12 matrix for head importance in 12 layers
            neuron_importance: list for neuron importance in 12 layers.
    """
    # model = model.module if hasattr(model, 'module') else model
    # base_model = getattr(model, model.base_model_prefix, model)

    # reorder heads and ffn neurons
    for layer in range(head_importance.shape[0]):
        # reorder heads
        _, idx = torch.sort(head_importance[layer], descending=True)
        model.blocks[layer].attn.reorder_heads(idx)
        # # TODO: reorder neurons
        # idx = torch.sort(current_importance, descending=True)[-1]
        # base_model.encoder.layer[layer].intermediate.reorder_neurons(idx)
        # base_model.encoder.layer[layer].output.reorder_neurons(idx)
