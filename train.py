import torch
from src.dataset import get_dataloader
from collections import defaultdict
from src.train_val import train_epoch, validation
from src.model import CRNN
from src.config import basic_config
import wandb
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "-w",
    type=str,
    help="wandb key",
)
parser.add_argument(
    "-c",
    type=str,
    help="config",
)

configs = {
    'basic': basic_config
}

if __name__ == '__main__':

    train_loader, melspec_train, val_loader, melspec_val = get_dataloader(True, True)

    args = parser.parse_args()

    config = configs[args.c]()
    model = CRNN(config).to(config.device)

    history = defaultdict(list)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    if args.w is not None:
        wandb.login(key=args.w)
        wandb.init(project="dla2", name="basic")

    for n in tqdm(range(config.num_epochs)):
        train_epoch(model, opt, train_loader,
                    melspec_train, config.device)

        au_fa_fr = validation(model, val_loader,
                              melspec_val, config.device)
        history['val_metric'].append(au_fa_fr)

        if args.w is not None:
            wandb.log({'val_metric': au_fa_fr})

        print(f'END OF EPOCH {n}\nMetric: {au_fa_fr}')

    torch.save(model.state_dict(), "model_basic")
