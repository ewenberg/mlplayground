from dl import NNEngine
import torch
from torch import nn

def test_nn_engine():
    model = nn.Sequential(
        nn.Linear(36, 36),
        nn.ReLU(),
        nn.Linear(12, 1)
    )

    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    classifier = NNEngine(model, loss_fn, optimizer)
