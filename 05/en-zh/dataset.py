import torch
from typing import Union, Type
from config import spm, Config
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from collections import deque


class Data(Dataset):
    def __init__(self, filepath: str) -> None:
        super(Data, self).__init__()
        if isinstance(filepath, list) or isinstance(filepath, tuple):
            with open(filepath[0], 'r') as fl:
                self.data = fl.readlines()
            with open(filepath[1], 'r') as fl:
                self.label = fl.readlines()
        else:
            with open(filepath, 'r') as fl:
                self.data = fl.readlines()
            self.label = None


    def __getitem__(self, i: int) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.label is None:
            return torch.Tensor(spm.EncodeAsIds(self.data[i])).long()
        else:
            return torch.Tensor(spm.EncodeAsIds(self.data[i])).long(), \
                   torch.Tensor(spm.EncodeAsIds(self.label[i])).long()

    def __len__(self) -> int:
        return len(self.data)


def collate(batch: list[Union[torch.Tensor, tuple[torch.Tensor,torch.Tensor]]]) -> tuple[torch.Tensor, ...]:
    if isinstance(batch[0], torch.Tensor):
        x = pad_sequence(batch, batch_first=True, padding_value=spm.pad_id())
        mask = torch.Tensor(x!=spm.pad_id()).bool()
        return x, mask
    else:
        batch = list(zip(*batch))
        x = pad_sequence(batch[0], batch_first=True, padding_value=spm.pad_id())
        y = pad_sequence(batch[1], batch_first=True, padding_value=spm.pad_id())
        x_mask = torch.Tensor(x!=spm.pad_id()).bool()
        y_mask = torch.Tensor(y!=spm.pad_id()).bool()
        return x, y, x_mask, y_mask


def sorted_collate(batch: list[Union[torch.Tensor, tuple[torch.Tensor,torch.Tensor]]]) -> tuple[torch.Tensor, ...]:
    if not isinstance(batch[0], torch.Tensor):
        batch.sort(key=lambda x:len(x[1]))
    return collate(batch)


class CachedDataLoader(DataLoader):
    def __init__(self, dataset:Dataset, batch_size:int, **kwargs):
        super(CachedDataLoader, self).__init__(dataset=dataset, batch_size=batch_size*20, **kwargs)
        self.queue = deque()
        self.iter = super(CachedDataLoader,self).__iter__()
        self.real_batch_size = batch_size

    def __iter__(self):
        self.iter = super(CachedDataLoader,self).__iter__()
        return self

    def __next__(self):
        if len(self.queue)==0:
            try:
                batch = next(self.iter)
            except StopIteration:
                raise StopIteration
            for b in zip(*[x.split(self.real_batch_size,dim=0) for x in batch]):
                x_len = b[2].sum(dim=1).max()
                y_len = b[3].sum(dim=1).max()
                self.queue.append((b[0][:,:x_len], b[1][:,:y_len], b[2][:,:x_len], b[3][:,:y_len]))
        return self.queue.popleft()

    def __len__(self):
        return super(CachedDataLoader,self).__len__()*20


def get_dataset(config: Type[Config]) -> tuple[Dataset, Dataset, Dataset]:
    test = Data(config.test_path)
    if config.valid_path is None:
        assert 1 > config.valid_ratio > 0
        train_valid = Data(config.train_path)
        n = len(train_valid)
        valid_size = int(n * config.valid_ratio)
        train_size = n - valid_size
        train, valid = random_split(train_valid, [train_size, valid_size],
                                    generator=torch.Generator().manual_seed(config.seed))
    else:
        train = Data(config.train_path)
        valid = Data(config.valid_path)
    return train, valid, test


def get_data_loader(config: Type[Config]) -> tuple[DataLoader, DataLoader, DataLoader]:
    train, valid, test = get_dataset(config)
    train_data = CachedDataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.n_workers,
        collate_fn=sorted_collate
    )
    valid_data = CachedDataLoader(
        valid,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.n_workers,
        collate_fn=sorted_collate
    )
    test_data = DataLoader(
        test,
        batch_size=config.batch_size//4,
        shuffle=False,
        drop_last=False,
        num_workers=config.n_workers,
        collate_fn=collate
    )
    return train_data, valid_data, test_data


if __name__ == '__main__':
    train, valid, test = get_dataset(Config())
    print(train[0][0].shape, train[0][1], len(train))
    print(valid[0][0].shape, valid[0][1], len(valid))
    print(test[0].shape, len(test))
