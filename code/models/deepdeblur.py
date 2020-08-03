import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, indim=64, outdim=64):
        super(ResBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.C2 = nn.Conv2d(outdim, outdim, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        out = self.C1(x)
        out = self.relu1(out)
        out = self.C2(out)
        out = out + x
        return out
        
class UpConvBlock(nn.Module):
    def __init__(self, ratio=4):
        super(UpConvBlock, self).__init__()
        self.conv = nn.Conv2d(3, 3*ratio**2, kernel_size=5, stride=1, padding=2)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        out = self.conv(x)
        out = self.ps(out)
        return out

class PyramidBlock(nn.Module):
    def __init__(self, block, num_layers, indim=3):
        super(PyramidBlock, self).__init__()
        layers = []

        conv1 = nn.Conv2d(indim, 64, kernel_size=5, stride=1, padding=2)
        layers.append(conv1)

        for _ in range(num_layers):
            B = block(64, 64)
            layers.append(B)

        conv_final = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)
        layers.append(conv_final)

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.layers(x)
        return out

class DeepDeblur(nn.Module):
    def __init__(self, block, k=3):
        super(DeepDeblur, self).__init__()
        layers_per_scale = 60//k - 1

        self.pyramid3 = PyramidBlock(block, layers_per_scale)
        self.upconv3 = UpConvBlock(2)
        self.pyramid2 = PyramidBlock(block, layers_per_scale, indim=6)
        self.upconv2 = UpConvBlock(2)
        self.pyramid1 = PyramidBlock(block, layers_per_scale, indim=6)

    def forward(self, B1, B2, B3):
        L3 = self.pyramid3(B3)
        L2 = self.pyramid2(torch.cat((self.upconv3(L3), B2), axis=1))
        L1 = self.pyramid1(torch.cat((self.upconv2(L2), B1), axis=1))

        return L1, L2, L3

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def DeepDeblur_scale3():
    return DeepDeblur(ResBlock)


if __name__ == '__main__':
    model = DeepDeblur_scale3()
    print(model)
    string = ''
    string = string + '-' * 30 + '\n'
    string = string + 'Trainable params: ' + str(count_parameters(model)) + '\n'
    string = string + '-' * 30
    print(string)