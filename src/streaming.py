import torch
import torch.nn.functional as F


class Streaming:
    def __init__(self, model, max_window):
        self.model = model
        self.max_window = max_window

    def __call__(self, x):
        self.model.eval()
        outputs = []
        with torch.no_grad():
            hidden = None
            for i in range(x.shape[-1] - self.max_window):
                output, hidden = self.model(x[..., i:i + self.max_window], hidden, True)
                output = F.softmax(output, dim=-1)
                outputs.append(output.unsqueeze(-1))

        return torch.cat(outputs, dim=-1)
