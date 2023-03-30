import math
import os
import csv
import traceback
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import deque
from tqdm.auto import tqdm
from typing import Union, Iterator
from config import Config, spm
from model import Model
from dataset import get_data_loader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def beam_search(
        model:nn.Module,
        x:torch.Tensor,
        y:torch.Tensor,
        x_mask:torch.Tensor,
        y_mask:torch.Tensor,
        k:int,
        maxlen:int = 1000
)->tuple[torch.Tensor, torch.Tensor]:

    def _add_to_final(
            candidates:torch.Tensor,
            log_p:torch.Tensor,
            idx:torch.Tensor,
            finished:torch.Tensor,
            final_candidates:list[list[torch.Tensor]],
            final_log_p:list[list[float]]
    ) -> tuple[list[list[torch.Tensor]], list[list[float]]]:
        for i in range(candidates.size(0)):
            for j in range(candidates.size(1)):
                if not finished[i,j] and idx[i, j] == spm.eos_id():
                    final_candidates[i].append(candidates[i, j, :])
                    final_log_p[i].append(log_p[i, j].item()/candidates.size(2))
        return final_candidates, final_log_p

    # out : (batch, len, vocab), mem : (batch, len_x, d_model)
    out, mem = model(x,y,torch.logical_not(x_mask),torch.logical_not(y_mask),require_memory=True)
    out = out.detach()
    mem = mem.detach()
    # (batch*k, len_x, d_model)
    mem = mem.repeat_interleave(k,dim=0)
    mem_mask = torch.logical_not(x_mask).repeat_interleave(k,dim=0)

    batch = out.size(0)
    len = out.size(1)
    vocab = out.size(2)

    # (batch, vocab)
    out = out.gather(1,(y_mask.sum(dim=1).long()-1).view(batch,1,1).repeat((1,1,vocab))).squeeze()
    # (batch, vocab)
    log_p = F.softmax(out, dim=-1).log()
    # (batch, k)
    log_p, idx = log_p.topk(k, dim=-1)

    # (batch, k, len)
    candidates = torch.cat((y.view(batch,1,len).repeat((1,k,1)), idx.unsqueeze(-1)),dim=2)
    len += 1
    # (batch, k)
    final_candidates = [[] for _ in range(batch)]
    final_log_p = [[] for _ in range(batch)]
    final_candidates, final_log_p = _add_to_final(
        candidates,
        log_p,
        idx,
        torch.zeros((batch,k),dtype=torch.bool).to(y.device),
        final_candidates,
        final_log_p
    )
    finished = torch.Tensor(idx==spm.eos_id()).bool()


    while (k > finished.sum(dim=-1).long()).any() and len<maxlen:
        # (batch*k, len, vocab)
        out = model(
            x=mem,
            y=candidates.view(batch*k, len),
            src_padding_mask=mem_mask,
            tgt_padding_mask=None,
            is_x_memory=True
        ).detach()
        # (batch*k, vocab)
        out = out[:,-1:]
        # (batch, k, vocab)
        out = out.view(batch, k, vocab)
        # (batch, k, vocab)
        log_p = F.softmax(out, dim=-1).log() + log_p.unsqueeze(-1).repeat((1,1,vocab))
        # set finished log_p to -inf which is less than all other log(p)
        log_p = log_p * (torch.logical_not(finished).unsqueeze(-1).repeat((1,1,vocab)) * -torch.inf)
        # (batch, k, vocab) -> (batch, k*vocab)
        log_p = log_p.view(batch, k*vocab)
        # (batch, k*vocab) -> (batch, k)
        log_p, idx = log_p.topk(k, dim=-1)
        pre_idx = idx // vocab
        idx = idx % vocab
        # (batch, k, len) -> (batch, k, len)
        candidates = candidates.gather(1,pre_idx.unsqueeze(-1).repeat((1,1,len)))
        # (batch, k) -> (batch, k)
        finished = finished.gather(1,pre_idx)
        # (batch, k, len) -> (batch, k, len+1)
        candidates = torch.cat((candidates, idx.unsqueeze(-1)),dim=2)
        len += 1
        # (batch, k) -> (batch, k)
        final_candidates, final_log_p = _add_to_final(candidates,log_p,idx,finished,final_candidates,final_log_p)
        finished = torch.logical_or(finished, torch.Tensor(idx==spm.eos_id()).bool())

    # (batch, len)
    ret_c = torch.ones((batch,len),dtype=torch.long).to(y.device) * spm.pad_id()
    ret_m = torch.zeros((batch,len),dtype=torch.bool).to(y.device)
    for i in range(batch):
        if not final_candidates[i]:
            ret_c[i,:2] = torch.Tensor([spm.bos_id(),spm.eos_id()]).long().to(y.device)
            ret_m[i,:2] = 1
        else:
            j = np.argmax(final_log_p[i])
            ret_c[i, :final_candidates[i][j].size(0)] = final_candidates[i][j]
            ret_m[i, :final_candidates[i][j].size(0)] = 1
    return ret_c.detach(), ret_m.detach()


def model_run(
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
        model: nn.Module,
        criterion: callable,
        config: Config,
        train: bool = False
) -> dict:
    x = x.to(config.device)
    y = y.to(config.device)
    x_mask = x_mask.to(config.device)
    y_mask = y_mask.to(config.device)

    pred = model(x, y[:,:-1], torch.logical_not(x_mask), torch.logical_not(y_mask[:,1:]))

    loss = criterion(pred.reshape(pred.size(0)*pred.size(1),pred.size(2)), y[:,1:].reshape(y.size(0)*(y.size(1)-1)))

    if train:
        loss.backward()

    metrics = {}
    if (config.require_metrics_on_training or not train) and isinstance(config.other_metrics, dict):
        metrics = {name:metric(pred, y, y_mask) for name, metric in config.other_metrics.items()}

    if config.classification:
        acc = torch.mean((pred.argmax(axis=1) == y.argmax(axis=1)).float())
        return {'loss':loss.item(), 'acc':acc.item(), **metrics}
    else:
        return {'loss':loss.item(), **metrics}


def train(
        train_data: DataLoader,
        train_iter: Iterator,
        model: nn.Module,
        criterion: callable,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        writer: SummaryWriter,
        start_step: int,
        config: Config
) -> None:
    model.train()
    bar = tqdm(total=min(config.valid_steps, config.steps_n-start_step), ncols=0, desc="Train", unit=" step")

    q = deque()
    norm = 0
    # training loop
    for step in range(start_step, min(config.steps_n, start_step+config.valid_steps)):
        # get data
        try:
            x, y, x_mask, y_mask = next(train_iter)
        except StopIteration:
            train_iter = iter(train_data)
            x, y, x_mask, y_mask = next(train_iter)

        # forward and backward
        try:
            metrics = model_run(x, y, x_mask, y_mask, model, criterion, config, train=True)
        except Exception:
            traceback.print_exc()
            print(f'\n[step:{step+1:d}] Runtime error, current batch skipped.\n')
            continue

        if math.isnan(metrics['loss']):
            print(f'nan loss! step: {step+1:d}.')

        q.append(metrics['loss'])
        if len(q)>100:
            q.popleft()

        if (step+1) % config.grad_accum == 0:
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            try:
                optimizer.step()
                scheduler.step()
            except:
                traceback.print_exc()
                print(f'\n[step:{step+1:d}] Runtime error, clear gradient.\n')
            optimizer.zero_grad()

        # tqdm log
        bar.update()
        bar.set_postfix(
            step=f'{step + 1:d}',
            norm=f'{norm:.4f}',
            ave_loss=f'{np.sum(q)/len(q):.4f}',
            lr=f'{scheduler.get_last_lr()[0]:.3e}',
            **{k:f'{v:.4f}' for k,v in metrics.items()}
        )

        # tensor board log
        for k, v in metrics.items():
            writer.add_scalar(f'{k}/training', v, step + 1)

    bar.close()


def validate(
        valid_data: DataLoader,
        valid_iter: Iterator,
        model: nn.Module,
        writer: SummaryWriter,
        step_cnt: int,
        config: Config
) -> dict:
    model.eval()
    bar = tqdm(total=config.valid_batch, ncols=0, desc="Valid", unit=" item")

    total_metrics = {}
    step = 0
    with torch.no_grad():
        while step < config.valid_batch:
            try:
                x, y, x_mask, y_mask = next(valid_iter)
            except StopIteration:
                valid_iter = iter(valid_data)
                x, y, x_mask, y_mask = next(valid_iter)

            try:
                x = x.to(config.device)
                y = y.to(config.device)
                x_mask = x_mask.to(config.device)
                y_mask = y_mask.to(config.device)
                input = torch.Tensor([spm.bos_id()]).to(config.device)
                input = input.unsqueeze(dim=0).long().repeat((x.size(0),1))
                pred, mask = beam_search(
                    model,
                    x,
                    input,
                    x_mask,
                    torch.ones_like(input,dtype=torch.bool).to(config.device),
                    config.beam_size,
                    y.size(1)*1.5
                )
                metrics = {name:metric(pred, y, mask, y_mask) for name, metric in config.other_metrics.items()}
            except:
                traceback.print_exc()
                print(f'\n[step:{step+1:d}] Runtime error, current batch skipped.\n')
                continue

            step += 1
            bar.update()
            bar.set_postfix(step=f'{step:d}', **{k:f'{v:.4f}' for k,v in metrics.items()})
            for k in metrics.keys():
                if k not in total_metrics.keys():
                    total_metrics[k] = 0
                total_metrics[k] += metrics[k]

    bar.close()

    # tensor board log
    for k in total_metrics.keys():
        total_metrics[k] /= step
        writer.add_scalar(f'{k}/validation', total_metrics[k], step_cnt)

    return total_metrics


def predict(test_data: DataLoader, model: nn.Module, filename: str, config: Config) -> Union[np.ndarray, list, tuple]:
    model_state, _, __ = torch.load(os.path.join(config.model_dir, filename+'.ckpt'))
    model.load_state_dict(model_state)
    model.eval()
    model.to(config.device)
    preds = []
    with torch.no_grad():
        for x, x_mask in tqdm(test_data):
            x = x.to(config.device)
            input = torch.Tensor([spm.bos_id()]).to(config.device).unsqueeze(dim=0).long()
            pred, mask = beam_search(
                model,
                x,
                input,
                x_mask,
                torch.ones_like(input).bool().to(config.device),
                config.beam_size
            )
            for i in range(x.size(0)):
                preds.append([
                    spm.DecodeIdsWithCheck(x[i,:x_mask[i,:].sum()].squeeze().detach().cpu().numpy().tolist()),
                    spm.DecodeIdsWithCheck(pred[i,:mask[i,:].sum()].squeeze().detach().cpu().numpy().tolist())
                ])
    return preds


def save_prediction(preds: Union[np.ndarray, list, tuple], filename: str, config: Config) -> None:
    file = os.path.join(config.pred_dir, filename + '.csv')
    with open(file, 'w') as fl:
        writer = csv.writer(fl)
        writer.writerow([config.src_lang, config.tgt_lang])
        for i, o in enumerate(preds):
            writer.writerow([i, o])


if __name__ == '__main__':
    config = Config()
    filename = input('Filename')
    model = Model(config).to(config.device)
    train_data, valid_data, test_data = get_data_loader(config)
    preds = predict(test_data, model, filename, config)
    save_prediction(preds, filename, config)