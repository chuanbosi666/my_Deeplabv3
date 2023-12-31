import time
import os
import datetime

import torch

from src import deeplabv3_resnet50
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, init_distributed_mode, save_on_master, mkdir
from my_dataset import VOCSegmentation
import transforms as T


class SegmentationPresetTrain:  # 训练数据预处理,和之前的模型一样图像数据增广处理
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train):
    base_size = 520
    crop_size = 480

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)


def create_model(aux, num_classes):
    model = deeplabv3_resnet50(aux=aux, num_classes=num_classes)  # 使用resnet50作为我们backbone
    weights_dict = torch.load("./deeplabv3_resnet50_coco.pth", map_location='cpu')  #导入预训练权重

    if num_classes != 21:
        # 官方提供的预训练权重是21类(包括背景)
        # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
        for k in list(weights_dict.keys()):
            if "classifier.4" in k:    # 会删去分类器的最后一个卷积的通道数
                del weights_dict[k]

    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    return model


def main(args):
    init_distributed_mode(args)  # 启动多进程
    print(args)

    device = torch.device(args.device)
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    VOC_root = args.data_path  # 导入我们的数据集的路径
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # load train data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    train_dataset = VOCSegmentation(args.data_path,
                                    year="2012",
                                    transforms=get_transform(train=True),
                                    txt_name="train.txt")  #导入数据集
    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=get_transform(train=False),
                                  txt_name="val.txt") #导入验证集

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) # 这里我们直接将验证集作为了测试集，并且启动了多进程模式
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn, drop_last=True)  # 创建训练的dataloader,将drop_last=True后，在训练的时候，最后一个batch不足一个batch时，会将其丢弃

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)  # 创建验证的dataloader

    print("Creating model")
    # create model num_classes equal background + 20 classes
    model = create_model(aux=args.aux, num_classes=num_classes)  # 调用create_model函数，并传入参数
    model.to(device)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 是不是使用多进程的BN处理

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]  # 这个模型的backbone和分类器部分进行参数优化
    if args.aux:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})  # 对于使用辅助分类器的操作，进行参数优化，设置优化器，学习率
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  #使用SGD作为优化器，以及一些参数设定，像动量、权重衰减、学习率

    scaler = torch.cuda.amp.GradScaler() if args.amp else None  # 使用自动混合精度作为我们的梯度更新策略

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)，也就是一个batch更新一次
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs, warmup=True)

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:  # 测试要用到的模型，数据集
        confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        return

    print("Start training")
    start_time = time.time()  # 记录时间
    for epoch in range(args.start_epoch, args.epochs):  #开始循环
        if args.distributed:  # 是否启用多进程训练
            train_sampler.set_epoch(epoch)
            #从一个epoch中获取学习率和损失，这个train_one_epoch是导入官方的模块，传进去模型，优化器，dataloader等等
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes) #，验证部分如此
        val_info = str(confmat) #将验证的部分转化成字符串的形式，打印下来
        print(val_info)

        # 只在主进程上进行写操作
        if args.rank in [-1, 0]:  # 针对这一行代码，rank为0，代表的是主进程，而rank为-1时是特殊的角色，通常不承担任务或计算，更多的用于
            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n"
                f.write(train_info + val_info + "\n\n")

        if args.output_dir:
            # 只在主节点上执行保存权重操作
            save_file = {'model': model_without_ddp.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'lr_scheduler': lr_scheduler.state_dict(),
                         'args': args,
                         'epoch': epoch}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            save_on_master(save_file,
                           os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time # 计算总时间
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练文件的根目录(VOCdevkit)
    parser.add_argument('--data-path', default='/data/', help='dataset')
    # 训练设备类型
    parser.add_argument('--device', default='cuda', help='device')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=20, type=int, help='num_classes')
    # 每块GPU上的batch_size
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    # 是否使用同步BN(在多个GPU之间同步)，默认不开启，开启后训练速度会变慢
    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using SyncBatchNorm')
    # 数据加载以及预处理的线程数
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # 训练学习率，这里默认设置成0.0001，如果效果不好可以尝试加大学习率
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练过程打印信息的频率
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./multi_train', help='path where to save')
    # 基于上次的训练结果接着训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 不训练，仅测试
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # 分布式进程数
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
