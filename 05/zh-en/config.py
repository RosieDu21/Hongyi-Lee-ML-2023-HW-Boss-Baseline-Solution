import os
import time
from functools import partial
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torchtext.data.metrics as metrics
import sentencepiece as sp
import random

vocab = 10000
spm = sp.SentencePieceProcessor(model_file=os.path.join(os.path.dirname(__file__), f'../DATA/sp{vocab}.model'))
spm.SetEncodeExtraOptions('bos:eos')


def lr_lambda(step:int, d_model:int, warmup:int)->float:
    return d_model ** -0.5 * min((step+1) ** -0.5, (step+1) * warmup ** -1.5)


def bleu(pred:torch.Tensor, y:torch.Tensor, mask:torch.Tensor, y_mask:torch.Tensor)->float:
    pred = pred.detach().cpu().numpy().tolist()
    y = y.detach().cpu().numpy().tolist()
    for i in range(len(pred)):
        pred[i] = spm.Decode(pred[i][0:mask[i,:].sum().item()]).split(' ')
        y[i] = [spm.Decode(y[i][0:y_mask[i,:].sum().item()]).split(' ')]
        if random.randint(1,2000) <= 1:
            print('\nOutput sample:')
            print('Pred: ', ' '.join(pred[i]))
            print('GT: ', ' '.join(y[i][0]))
    return metrics.bleu_score(pred,y)


class Config:
    src_lang = 'zh'
    tgt_lang = 'en'

    file_dir = os.path.dirname(__file__)

    data_prefix = os.path.join(file_dir, '../DATA')
    train_prefix = os.path.join(data_prefix,'train_dev.clean.')
    test_prefix = os.path.join(data_prefix, 'mono_test.clean.')
    train_path = (train_prefix+src_lang, train_prefix+tgt_lang)
    test_path  = test_prefix+src_lang
    valid_path = None

    n_workers = 4  # data loader workers

    valid_ratio = 0.01

    seed = 666

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    use_ckpt = None

    model_name = 'Transformer-back'

    vocab_size = vocab

    hidden_width = 512
    hidden_depth = 6
    feedforward  = 2048
    dropout = 0.1
    n_head  = 8

    batch_size  = 32
    grad_accum  = 4
    start_step  = 0
    steps_n     = 750_000
    valid_steps = 20_000
    valid_batch = 64
    early_stop  = 750_000
    warmup      = 4_000

    learning_rate = 1
    weight_decay  = 1e-5

    max_norm = 1

    beam_size = 5

    loss_name = 'CE'
    criterion = partial(nn.CrossEntropyLoss, ignore_index=spm.pad_id())
    opt_name  = 'Adam'
    optimizer = partial(
        torch.optim.Adam,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9,0.98),
        eps=1e-9
    )
    lrs_name  = 'InvSqrt'
    scheduler = partial(lr_scheduler.LambdaLR,lr_lambda=partial(lr_lambda,d_model=hidden_width,warmup=warmup))
    other_metrics = {'BLEU':bleu}
    classification = False
    require_metrics_on_training = False
    save_criterion = 'BLEU'
    greater_better = True # if save criterion is "greater is better"

    file_name_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    file_name_config = '.'.join([
        model_name,
        loss_name,
        opt_name,
        lrs_name,
        'w' + str(hidden_width),
        'd' + str(hidden_depth),
        'h' + str(n_head),
        'bs' + str(batch_size),
        'reg' + str(weight_decay),
    ])
    file_name_both = file_name_time + '_' + file_name_config

    exp_dir   = os.path.join(file_dir, '../exps')
    base_dir  = os.path.join(exp_dir, file_name_both)
    log_dir   = os.path.join(base_dir, 'train_log')
    model_dir = os.path.join(base_dir, 'model')
    pred_dir  = os.path.join(base_dir, 'pred')
    tensorboard_dir = os.path.join(base_dir, 'runs')


if __name__ == '__main__':
    print(Config.file_name_both)
