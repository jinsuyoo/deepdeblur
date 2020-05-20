import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, negative_slope=0.2):
        super(Discriminator, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, stride=4, padding=2, bias=False),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(256, 256, kernel_size=5, stride=4, padding=2, bias=False),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=4, padding=0, bias=False),
            nn.LeakyReLU(negative_slope, inplace=True)
        )
      
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out