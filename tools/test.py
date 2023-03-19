#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import sys
sys.path.append(".")

from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
from test_net import test
from train_net import train
from visualization import visualize

"""Wrapper to train and test a video classification model."""


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    launch_job(cfg=cfg, init_method=args.init_method, func=test)



if __name__ == "__main__":
    main()
