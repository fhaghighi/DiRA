import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from data_loader import ChestX_ray14
import transformation
from DiRA_models import DiRA_UNet,DiRA_MoCo,MoCo,Discriminator,weights_init_normal
from trainer import train_dir,validate_dir,train_dira,validate_dira
from torch.autograd import Variable

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ChestX-ray14 Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=1200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-batch-size', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--disc-learning-rate', default=0.0002, type=float, metavar='LR',
                    help=' learning rate for discriminator')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.2)')

parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=str, help='path to checkpoint directory')
parser.add_argument('--train_list', default='dataset/Xray14_train_official.txt', type=str,
                     help='file for training list')
parser.add_argument('--val_list', default='dataset/Xray14_val_official.txt', type=str,
                     help='file for validation list')
parser.add_argument('--mode', default='dira', type=str,
                     help='di|dir|dira')
parser.add_argument('--encoder_weights', default=None, type=str,help='encoder pre-trained weights if available')
parser.add_argument('--activate', default="sigmoid", type=str,help='activation for reconstruction')
parser.add_argument('--contrastive_weight', default=1, type=float,help='weight of instance discrimination loss')
parser.add_argument('--mse_weight', default=10, type=float,help='weight of reconstruction loss')
parser.add_argument('--adv_weight', default=0.001, type=float,help='weight of adversarial loss')
parser.add_argument('--exp_name', default="DiRA_moco", type=str,help='experiment name')
parser.add_argument('--out_channels', default=1, type=str,help='number of channels in generator output')
parser.add_argument('--generator_pre_trained_weights', default=None, type=str,help='generator pre-trained weights')



def main():
    args = parser.parse_args()
    args.checkpoint_dir = os.path.join(args.checkpoint_dir,args.exp_name,args.mode)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    D_output_shape = (1, 28, 28)
    # create model
    if args.mode.lower() == "di": #discriminator only
        model = MoCo(models.__dict__[args.arch],args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    else:
        model = DiRA_MoCo(DiRA_UNet, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, backbone=args.arch, encoder_weights=args.encoder_weights, activation=args.activate)
    print(model)
    discriminator = Discriminator(args.out_channels)
    discriminator.apply(weights_init_normal)
    print(discriminator)

    if args.generator_pre_trained_weights is not None:
        print ("Loading pre-trained weights for generator...")
        ckpt = torch.load(args.generator_pre_trained_weights, map_location='cpu')
        if "state_dict" in ckpt:
            ckpt = ckpt['state_dict']
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        msg = model.load_state_dict(ckpt)
        print("=> loaded pre-trained model '{}'".format(args.generator_pre_trained_weights))
        print("missing keys:", msg.missing_keys)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            discriminator.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu])
        else:
            model.cuda()
            discriminator.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            discriminator = torch.nn.parallel.DistributedDataParallel(discriminator)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        discriminator = discriminator.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    nce_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    mse_criterion = nn.MSELoss().cuda(args.gpu)
    adversarial_criterion = nn.MSELoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_D = torch.optim.Adam(
                    params=discriminator.parameters(),
                    lr=args.disc_learning_rate,
                    betas=[0.5, 0.999]) #set from inpainting paper)


    best_loss = 10000000000
    cudnn.benchmark = True

    dataset_train = ChestX_ray14(pathImageDirectory=args.data, pathDatasetFile=args.train_list,
                                 augment=transformation.Transform(mode=args.mode))
    dataset_valid = ChestX_ray14(pathImageDirectory=args.data, pathDatasetFile=args.val_list,
                                 augment=transformation.Transform(mode=args.mode))


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset_valid)

    else:
        train_sampler = None
        valid_sampler = None


    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, sampler=train_sampler, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, sampler=valid_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        if args.mode.lower() == "di" or args.mode.lower() == "dir":
            train_dir(train_loader, model, nce_criterion, mse_criterion, optimizer, epoch, args)
            counter = validate_dir(valid_loader, model, nce_criterion, mse_criterion, epoch, args)
        elif args.mode.lower() =="dira":
            train_dira(train_loader, model, nce_criterion, mse_criterion, adversarial_criterion, optimizer, epoch,args, discriminator, optimizer_D, D_output_shape)
            counter = validate_dira(valid_loader, model, nce_criterion, mse_criterion, adversarial_criterion, epoch,args, discriminator, D_output_shape)

        torch.distributed.reduce(counter, 0)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):

            valid_loss = counter[0]/counter[1]
            print ("validation loss: ",valid_loss)
            if valid_loss < best_loss:
                print("Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}".format(epoch, best_loss, valid_loss))
                best_loss = valid_loss
                if args.mode.lower() == "di":
                    torch.save(model.module.encoder_q.state_dict(), os.path.join(args.checkpoint_dir, 'best_checkpoint.pth'))
                else:
                    torch.save(model.module.encoder_q.backbone.state_dict(), os.path.join(args.checkpoint_dir,'best_checkpoint.pth'))

            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_loss':best_loss
            }, os.path.join(args.checkpoint_dir,'checkpoint.pth'))

            if args.mode.lower() == "dira":
                torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': discriminator.state_dict(),
                    'optimizer' : optimizer_D.state_dict(),
                }, os.path.join(args.checkpoint_dir,'D_checkpoint.pth'))


    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        if args.mode.lower() == "di":
            torch.save(model.module.encoder_q.state_dict(), os.path.join(args.checkpoint_dir, 'resnet50.pth'))
        else:
            torch.save(model.module.encoder_q.backbone.state_dict(),
                   os.path.join(args.checkpoint_dir,'unet.pth'))


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
