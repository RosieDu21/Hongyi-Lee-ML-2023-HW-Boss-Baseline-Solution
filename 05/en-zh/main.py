import os
import math
from glob import glob
from shutil import copyfile

import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

from utils import train, validate, predict, save_prediction
from model import Model
from logger import Logger
from config import Config
from dataset import get_data_loader
from typing import Type


def set_random_seed(config: Type[Config]) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)


def set_file_system(config: Type[Config]) -> None:
    # set paths
    for path in [config.exp_dir, config.base_dir, config.log_dir, config.model_dir, config.pred_dir]:
        if not os.path.isdir(path):
            os.mkdir(path)

    # copy source files
    for filepath in glob(os.path.join(os.path.dirname(__file__), '*.py')):
        filename = os.path.split(filepath)[-1]
        copyfile(filepath, os.path.join(config.log_dir, filename))


def main(config: Type[Config]) -> None:
    # set random seed
    set_random_seed(config)

    # set file system
    set_file_system(config)

    # get file name
    filename = config.file_name_config

    # set tensorboard writer
    writer = SummaryWriter(os.path.join(config.tensorboard_dir, filename))

    # set logger (print on screen and in file at same time)
    logger = Logger(os.path.join(config.log_dir, filename+'.txt'))
    print('[info] Log on.', flush=True)

    # prepare data
    print('[info] Preparing data...', flush=True)
    train_data, valid_data, test_data = get_data_loader(config)
    print('[info] Data preparation finished.', flush=True)

    # set model, loss, optimizer and learning rate decay
    print('[info] Using '+config.device+'.', flush=True)
    print('[info] Preparing model...', flush=True)
    model = Model(config).to(config.device)
    model_save_name = ''
    criterion = config.criterion()
    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
    if config.use_ckpt is not None:
        model_stt, optim_stt, sched_stt = torch.load(config.use_ckpt, map_location=config.device)
        model.load_state_dict(model_stt)
        optimizer.load_state_dict(optim_stt)
        scheduler.load_state_dict(sched_stt)
    print('[info] Model preparation finished.', flush=True)

    # training preparation
    optimal_criterion = -math.inf if config.greater_better else math.inf
    early_stop_cnt = 0
    train_iter = iter(train_data)
    valid_iter = iter(valid_data)

    step = config.start_step
    while step < config.steps_n:
        # train
        train(train_data, train_iter, model, criterion, optimizer, scheduler, writer, step, config)
        step += config.valid_steps

        # validation
        metrics = validate(valid_data, valid_iter, model, writer, step, config)

        # save model and early stopping
        if (config.greater_better and metrics[config.save_criterion]>=optimal_criterion) or \
                (not config.greater_better and metrics[config.save_criterion]<=optimal_criterion):
            optimal_criterion = metrics[config.save_criterion]
            early_stop_cnt = 0
            print(f'[info] Optimal model has been found. {config.save_criterion}: {optimal_criterion:.4f}. Saving...',
                  flush=True)
            model_save_name = str(step) + '.' + filename
            torch.save((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()),
                       os.path.join(config.model_dir, model_save_name + '.ckpt'))
            print(f'[info] Saved. ({model_save_name:s})', flush=True)
        else:
            early_stop_cnt += config.valid_steps

        if early_stop_cnt > config.early_stop:
            print(f'[info] No improvement for {early_stop_cnt:d} steps. Early stop.', flush=True)
            break

    # predict
    if model_save_name != '':
        print('[info] Inferring...', flush=True)
        preds = predict(test_data, model, model_save_name, config)
        print('[info] Saving prediction...', flush=True)
        save_prediction(preds, filename, config)
        print(f'[info] Saved. ({filename:s})', flush=True)

    print('[info] Log off.', flush=True)
    logger.close()


if __name__ == '__main__':
    main(Config)
