from torch import nn


class ClassifierHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, pooled):
        return self.fc(pooled)
