import logging

import torch

import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


def get_loader(args):

    if args.dataset == 'CVUSA':
        from utils.dataloader_usa import TrainDataloader,TestDataloader
    elif args.dataset == 'CVACT':
        from utils.dataloader_act import TrainDataloader,TestDataloader

    trainset = TrainDataloader(args)
    testset = TestDataloader(args)


    train_loader = DataLoader(trainset,
                            batch_size=args.train_batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            num_workers=4)

    test_loader = DataLoader(testset,
                            batch_size=args.eval_batch_size,
                            shuffle=False,
                            pin_memory=True, 
                            num_workers=4,
                            drop_last=False)


    return train_loader, test_loader

