import torch
from torch import nn


class DiscriminatorNN(torch.nn.Module):
    """
    Distriminative Neural Network: 4 Convolutional layers.
    Used to discriminate if generated image is fake or not for the text(embedding).
    """
    def __init__(self):
        super(DiscriminatorNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encode_text = nn.Sequential(
            nn.Linear(10, 100),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.concat_image_n_text = nn.Sequential(
            nn.Conv2d(
                in_channels=1124, out_channels=1024, kernel_size=1,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, caption_embedding):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        encoded_text = self.encode_text(caption_embedding)
        encoded_text = encoded_text.view(-1, 100, 1, 1)
        encoded_text = encoded_text.repeat(1, 1, 4, 4)
        concat = torch.cat((x, encoded_text), 1)
        output = self.concat_image_n_text(concat)
        return output.view(-1, 1).squeeze(1)




class GeneratorNN(torch.nn.Module):
    """
    Generative Neural Network: 5 deconvolutional layers to construct the image.
    Used to generate fake image for the given text(embedding).
    """
    def __init__(self):
        super(GeneratorNN, self).__init__()
        self.linear = torch.nn.Linear(200, 1024 * 4 * 4)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=200, out_channels=512, kernel_size=4,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=1, kernel_size=4,
                stride=2, padding=1, bias=False
            )
        )
        self.out = torch.nn.Tanh()
        self.encode_text = nn.Sequential(
            nn.Linear(10, 100), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x, caption_embedding):
        x = x.view(100, 100, 1, 1)
        encoded_text = self.encode_text(caption_embedding)
        encoded_text = encoded_text.view(-1, 100, 1, 1)
        x = torch.cat((x, encoded_text), 1)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.out(x)
        return x