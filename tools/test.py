#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import sys
sys.path.append(".")

from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
from test_net import test
from train_net import train
from visualization import visualize
from slowfast.models import build_model
import numpy as np
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
import torch
from fvcore.common.file_io import PathManager
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import TestMeter
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.
The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""

"""Wrapper to train and test a video classification model."""
logger = logging.get_logger(__name__)

def main():
    """
    Main function to spawn the train and test process.
    """

    args = parse_args()
    cfg = load_config(args)
    
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    print(model)
    # test model
    frames = torch.randn(1, 3, 8, 224, 224).cuda()
    label = torch.tensor([1])
    index = torch.tensor([1])
    input = [frames, label, index, {}]
    output = model(input)
    print(output.shape)


if __name__ == "__main__":
    main()
