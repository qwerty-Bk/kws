from typing import Union, List, Callable, Optional
from torch.utils.data import Dataset
import pathlib
import pandas as pd
import torchaudio
import torch
from src.config import TaskConfig
from src.augs import AugsCreation
from src.sampler import get_sampler, Collator
from torch.utils.data import DataLoader
from src.melspec import LogMelspec


class SpeechCommandDataset(Dataset):

    def __init__(
            self,
            transform: Optional[Callable] = None,
            path2dir: str = None,
            keywords: Union[str, List[str]] = None,
            csv: Optional[pd.DataFrame] = None
    ):
        self.transform = transform

        if csv is None:
            path2dir = pathlib.Path(path2dir)
            keywords = keywords if isinstance(keywords, list) else [keywords]

            all_keywords = [
                p.stem for p in path2dir.glob('*')
                if p.is_dir() and not p.stem.startswith('_')
            ]

            triplets = []
            for keyword in all_keywords:
                paths = (path2dir / keyword).rglob('*.wav')
                if keyword in keywords:
                    for path2wav in paths:
                        triplets.append((path2wav.as_posix(), keyword, 1))
                else:
                    for path2wav in paths:
                        triplets.append((path2wav.as_posix(), keyword, 0))

            self.csv = pd.DataFrame(
                triplets,
                columns=['path', 'keyword', 'label']
            )

        else:
            self.csv = csv

    def __getitem__(self, index: int):
        instance = self.csv.iloc[index]

        path2wav = instance['path']
        wav, sr = torchaudio.load(path2wav)
        wav = wav.sum(dim=0)

        if self.transform:
            wav = self.transform(wav)

        return {
            'wav': wav,
            'keywors': instance['keyword'],
            'label': instance['label']
        }

    def __len__(self):
        return len(self.csv)


def get_dataloader(train: bool, valid: bool, prefix: str = ''):
    dataset = SpeechCommandDataset(
        path2dir=prefix + 'speech_commands', keywords=TaskConfig.keyword
    )

    indexes = torch.randperm(len(dataset))
    train_indexes = indexes[:int(len(dataset) * 0.8)]
    val_indexes = indexes[int(len(dataset) * 0.8):]

    train_df = dataset.csv.iloc[train_indexes].reset_index(drop=True)
    val_df = dataset.csv.iloc[val_indexes].reset_index(drop=True)

    train_set = SpeechCommandDataset(csv=train_df, transform=AugsCreation(prefix + 'speech_commands'))
    val_set = SpeechCommandDataset(csv=val_df)

    train_sampler = get_sampler(train_set.csv['label'].values)

    # Here we are obliged to use shuffle=False because of our sampler with randomness inside.

    train_loader = DataLoader(train_set, batch_size=TaskConfig.batch_size,
                              shuffle=False, collate_fn=Collator(),
                              sampler=train_sampler,
                              num_workers=2, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=TaskConfig.batch_size,
                            shuffle=False, collate_fn=Collator(),
                            num_workers=2, pin_memory=True)

    melspec_train = LogMelspec(is_train=True, config=TaskConfig)
    melspec_val = LogMelspec(is_train=False, config=TaskConfig)

    if train:
        if valid:
            return train_loader, melspec_train, val_loader, melspec_val
        else:
            return train_loader, melspec_train
    else:
        if valid:
            return val_loader, melspec_val
