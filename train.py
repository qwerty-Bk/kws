import torch
from src.dataset import SpeechCommandDataset
from src.augs import AugsCreation
from src.sampler import get_sampler, Collator
from torch.utils.data import DataLoader
from src.melspec import LogMelspec
from collections import defaultdict
from src.train_val import train_epoch, validation
from src.model import CRNN
from src.config import TaskConfig
import wandb
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "-w",
    type=str,
    help="wandb key",
)

if __name__ == '__main__':

    dataset = SpeechCommandDataset(
        path2dir='speech_commands', keywords=TaskConfig.keyword
    )

    indexes = torch.randperm(len(dataset))
    train_indexes = indexes[:int(len(dataset) * 0.8)]
    val_indexes = indexes[int(len(dataset) * 0.8):]

    train_df = dataset.csv.iloc[train_indexes].reset_index(drop=True)
    val_df = dataset.csv.iloc[val_indexes].reset_index(drop=True)

    train_set = SpeechCommandDataset(csv=train_df, transform=AugsCreation())
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

    config = TaskConfig()
    model = CRNN(config).to(config.device)

    history = defaultdict(list)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    args = parser.parse_args()

    if args.w is not None:
        wandb.login(key=args.w)
        wandb.init(project="dla2", name="basic")

    for n in tqdm(range(TaskConfig.num_epochs)):
        train_epoch(model, opt, train_loader,
                    melspec_train, config.device)

        au_fa_fr = validation(model, val_loader,
                              melspec_val, config.device)
        history['val_metric'].append(au_fa_fr)

        if args.w is not None:
            wandb.log({'val_metric': au_fa_fr})

        print(f'END OF EPOCH {n}\nMetric: {au_fa_fr}')

    torch.save(model.state_dict(), "model_basic")
