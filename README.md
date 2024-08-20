# ViT
需下载tiny-ImageNet数据集到当前目录

Prune Vision Transformer

1、执行 python main.py --training_phase fintuning  就会训练stage1
2、执行 python main.py --training_phase width  就会训练stage2
3、执行 python main.py --training_phase depth  就会训练stage3

完成后models文件夹下会有一个finetune模型和12个不同宽度和深度的vit蒸馏模型

