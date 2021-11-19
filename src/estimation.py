import numpy as np
import time
from thop import profile
from src.dataset import get_dataloader


class Timer:

    def __init__(self, name: str, verbose=False):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.t = time.time() - self.t

        if self.verbose:
            print(f"{self.name.capitalize()} | Elapsed time : {self.t:.2f}")


def get_size_in_megabytes(model):
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    param_size = next(model.parameters()).element_size()
    return (num_params * param_size) / (2 ** 20)


def count_pars(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def count_mac(model, config):
    element = None

    val_loader, melspec_val = get_dataloader(False, True)

    for batch, labels in val_loader:
        batch, labels = batch.to(config.device), labels.to(config.device)
        element = melspec_val(batch)

    return profile(model, (element, ))
