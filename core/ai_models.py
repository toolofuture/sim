import torch
import torch.nn as nn
import torchvision.models as models

class SiameseNetwork(nn.Module):
    """
    Siamese Network with ResNet18 backbone for similarity comparison.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Use ResNet18 as backbone
        self.resnet = models.resnet18(pretrained=True)
        
        # Remove the fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Add a new fully connected layer for embedding
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward_one(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2
    
    def forward_spatial(self, x):
        """Returns spatial features for heatmap generation."""
        return self.features(x)

class Autoencoder(nn.Module):
    """
    Convolutional Autoencoder for Anomaly Detection.
    Trained on authentic art; high reconstruction error = anomaly (fake).
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # [B, 16, 112, 112]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # [B, 32, 56, 56]
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),                      # [B, 64, 50, 50]
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output 0-1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
