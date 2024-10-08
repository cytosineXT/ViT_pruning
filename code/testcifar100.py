# python ./code/testcifar100.py --num_classes 100 --batch_size 4096 --device "cuda:0" --img_size 32 

from deit_modified_ghost import VisionTransformer
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, ImageFolder
from torchvision import transforms
from tqdm import tqdm
from utils import get_logger,increment_path
import os
from thop import profile
import time
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    "--test",
    action="store_true",
    help="To only test the model with the given weights file",
)
parser.add_argument(
    "--reorder",
    action="store_true",
    help="Load model, reorder and save it"
)
parser.add_argument(
    "--path_val",
    type=str,
    default="./tiny-imagenet-200/val",
    # default="/home/jxt/docworkspace/ViT/tiny-imagenet-200/val",
    #default="F:/Download/ImageNet/val",
    help="Path to validation dataset (default: '../data/ImageNet200FullSize/val')",
)
parser.add_argument(
    "--model_architecture",
    type=str,
    default="vit_small_patch16_224",
    help="Architecture of model (default: 'vit_small_patch16_224')",
)
parser.add_argument(
    "--mha_width",
    type=float,
    default=1.0,
    help="Width of multi head attention (default: 1.0)",
)
parser.add_argument(
    "--mlp_width",
    type=float,
    default=1.0,
    help="Width of feed forward layer (default: 1.0)",
)
parser.add_argument(
    "--epochs", type=int, default=50, help="Number of epochs (default: 50)"
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:1",
    help="Device to train (or test) on (default: 'cuda:0')",
)
parser.add_argument(
    "--batch_size", type=int, default=128, help="Batch size (default: 64)"
)
parser.add_argument(
    "--num_classes", type=int, default=1000, help="Number of classes (default: 1000)"
)
parser.add_argument(
    "--img_size",
    type=int,
    default=224,
    help="Image size (default: 224)",
)
parser.add_argument(
    "--patch_size", type=int, default=16, help="Patch size (default: 16)"
)
parser.add_argument(
    "--embed_dim", type=int, default=384, help="Embedding dimension (default: 384)"
)
parser.add_argument("--depth", type=int, default=12, help="Depth (default: 12)")
parser.add_argument(
    "--num_heads", type=int, default=6, help="Number of heads (default: 6)"
)
parser.add_argument("--mlp_ratio", type=int, default=4, help="MLP ratio (default: 4)")
parser.add_argument(
    "--in_chans", type=int, default=3, help="Input channels (default: 3)"
)
parser.add_argument("--qkv_bias", action="store_true", help="To use qkv bias")
parser.add_argument("--no_ghost", action="store_true", help="To not use ghost")
parser.add_argument(
    "--ghost_mode",
    type=str,
    default="dense",
    help="Mode for applying ghost module (default: 'simple')",
)
parser.add_argument("--init_scratch", action="store_false", help="To start from scratch model")
# parser.add_argument("--training_phase", default="width", type=str,
parser.add_argument("--training_phase", default="test", type=str,
                        help="can be finetuning, width, depth")
args = parser.parse_args()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
save_dir = str(increment_path(Path(ROOT / "output" / "test" /'0822cifar100-prun'), exist_ok=False))
logdir = os.path.join(save_dir,'log.txt')
logger = get_logger(logdir)

logger.info(f' device: {args.device}, batch_size: {args.batch_size}, num_classes: {args.num_classes}, img_size: {args.img_size}')

if torch.cuda.is_available():
    device = torch.device(args.device)
    logger.info(f"{torch.cuda.device_count()} GPU(s) available.")
    logger.info("Device name:", torch.cuda.get_device_name(0))
else:
    logger.info("No GPU available, using the CPU instead.")
    device = torch.device("cpu")
     
path_val = args.path_val

train_transforms = create_transform(
    input_size=args.img_size,
    is_training=True,
    color_jitter=0.4,
    auto_augment="rand-m9-mstd0.5-inc1",
    interpolation="bicubic",
    re_prob=0.25,
    re_mode="pixel",
    re_count=1,
)
val_transforms = transforms.Compose(
    [
        transforms.Resize(args.img_size + 32),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ]
)

val_dataset = CIFAR100(root = './', train = False, transform=val_transforms)
val_sampler = SequentialSampler(val_dataset)
test_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size)

args.no_ghost = True
model = VisionTransformer(
    img_size=args.img_size,
    patch_size=args.patch_size,
    num_classes=args.num_classes,
    embed_dim=args.embed_dim,
    depth=args.depth,
    num_heads=args.num_heads,
    mlp_ratio=args.mlp_ratio,
    in_chans=args.in_chans,
    qkv_bias=args.qkv_bias,
    mha_width=args.mha_width,
    mlp_width=args.mlp_width,
    no_ghost=args.no_ghost,
    ghost_mode=args.ghost_mode,
)


for i, width in enumerate(tqdm([0.25, 0.5, 0.75, 1], desc="Width", leave=False)):
    # for j, depth in enumerate(tqdm([0.25,0.5], desc="Depth", leave=False)):
    for j, depth in enumerate(tqdm([0.25, 0.5, 0.75, 1], desc="Depth", leave=False)):
        path = os.path.join("code/testmodelcifar100", f"Width{width}_model_width_distillation.pt")
        # path = os.path.join("code/testmodelcifar100", f"Width{width}_Depth{depth}_model_width_distillation.pt")
        try:
            model.apply(lambda m: setattr(m, 'width_mult', width))
            model.apply(lambda m: setattr(m, 'depth', depth))
            model.load_state_dict(torch.load(path,weights_only=True), strict=False)
        except:
            continue
        logger.info(f'width{width},depth{depth}')
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, desc="Evaluating", leave=False)):
                inputs, labels = tuple(t.to(device) for t in data)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f'Accuracy of the  network on the test images: %0.2f %%' % accuracy)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info('Number of parameters: %d' % num_params)
        inputs, _ = next(iter(test_loader))
        inputs = inputs.to(device)
        with torch.no_grad():
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
        inference_time = end_time - start_time
        logger.info('Inference time: %.2fms' % (inference_time * 1000))
        flops, params = profile(model, inputs=(inputs,))
        logger.info('Number of parameters: %d' % params)
        logger.info('FLOPS: %.2fG' % (flops / 1e9))


# path = os.path.join("/home/jxt/docworkspace/ViT_pruning/code/testmodelcifar100/vit-small-224-cifar100i-finetuned-1.0.pth")
# logger.info(path)
# model.load_state_dict(torch.load(path,weights_only=True), strict=False)

# model.to(device)
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for i, data in enumerate(tqdm(test_loader, desc="Evaluating", leave=False)):
#         inputs, labels = tuple(t.to(device) for t in data)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = 100 * correct / total
# logger.info(f'Accuracy of the  network on the test images: %0.2f %%' % accuracy)
# num_params = sum(p.numel() for p in model.parameters())
# logger.info('Number of parameters: %d' % num_params)
# inputs, _ = next(iter(test_loader))
# inputs = inputs.to(device)
# with torch.no_grad():
#     start_time = time.time()
#     outputs = model(inputs)
#     end_time = time.time()
# inference_time = end_time - start_time
# logger.info('Inference time: %.2fms' % (inference_time * 1000))
# flops, params = profile(model, inputs=(inputs,))
# logger.info('Number of parameters: %d' % params)
# logger.info('FLOPS: %.2fG' % (flops / 1e9))