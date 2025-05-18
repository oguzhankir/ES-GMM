#src/model.py

import torch
import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, scale):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                               padding=(dilation * (kernel_size - 1)) // 2, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm1d(in_channels)
        self.se = SEModule(in_channels)

    def forward(self, x):
        residual = x
        out = self.bn1(self.relu(self.conv1(x)))
        out = self.bn2(self.relu(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + residual
        # The SE-Block is applied after the residual connection
        return self.se(out)


class EcapaTdnn(nn.Module):
    def __init__(self, in_channels=80, channels=1024, emb_dim=192):
        super(EcapaTdnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channels)

        self.layer1 = Bottleneck(channels, channels, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottleneck(channels, channels, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottleneck(channels, channels, kernel_size=3, dilation=4, scale=8)

        # This layer concatenates the outputs of the 3 SE-Blocks
        self.layer4 = nn.Conv1d(3 * channels, 1536, kernel_size=1)

        # Attentive Statistics Pooling
        self.attention = nn.Sequential(
            nn.Conv1d(1536, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # Use Tanh for attention score
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn_agg = nn.BatchNorm1d(3072)
        self.fc = nn.Linear(3072, emb_dim)
        self.bn_emb = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        out = self.bn1(self.relu(self.conv1(x)))

        out1 = self.layer1(out)
        out2 = self.layer2(out + out1)
        out3 = self.layer3(out + out1 + out2)

        # The cat annd conv layer are correct
        out = self.layer4(torch.cat((out1, out2, out3), dim=1))
        out = self.relu(out)

        # Attentive Statistics Pooling
        w = self.attention(out)
        mu = torch.sum(out * w, dim=2)
        sg = torch.sqrt((torch.sum((out ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-9))

        # Concatenate mean and standard deviation
        out = torch.cat((mu, sg), dim=1)

        out = self.bn_agg(out)
        out = self.fc(out)
        embedding = self.bn_emb(out)

        return embedding
