import torch
import torch.nn as nn


class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self._pass = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False),
            self._extract_feature(24, 6),
            self._pool_feature(96, 48),
            self._extract_feature(48, 12),
            self._pool_feature(192, 96),
            self._extract_feature(96, 24),
            self._pool_feature(384, 192),
            self._extract_feature(192, 16),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(4),
        )

        self.classifier = nn.Linear(384, 10)

    def _pool_feature(self, num_inputs, num_outputs):
        return nn.Sequential(
            nn.BatchNorm2d(num_inputs),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_inputs, num_outputs, kernel_size=1, bias=False),
            nn.AvgPool2d(2)
        )

    def _extract_feature(self, num_inputs, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(Mux(nn.Sequential(
                nn.BatchNorm2d(num_inputs),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_inputs, 48, kernel_size=1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.Conv2d(48, 12, kernel_size=3, padding=1, bias=False),
            )))
            num_inputs += 12
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self._pass(x)
        y = y.view(y.size(0), -1)
        logits = self.classifier(y)
        return logits


class Mux(nn.Module):

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.cat([self.m(x), x], 1)
