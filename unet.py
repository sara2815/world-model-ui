
import torch
import torch.nn as nn

class BasicUNet(nn.Module):
    """A minimal UNet implementation."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([ 
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2), 
        ])
        self.act = nn.SiLU() # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)
        
    def forward(self, x):

        h = []

        # Down path
        for i, l in enumerate(self.down_layers):

            x = self.act(l(x))

            if i < 2:
                h.append(x)
                x = self.downscale(x)

        # Up path
        for i, l in enumerate(self.up_layers):

            if i > 0:
                x = self.upscale(x)
                x += h.pop()

            # NO activation on final layer
            if i < len(self.up_layers) - 1:
                x = self.act(l(x))
            else:
                x = l(x)

        return x