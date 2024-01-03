from torch import Tensor, nn
from torchvision.models.resnet import ResNet, BasicBlock
class Resnet_model(ResNet):
    def __init__(self) -> None:
        super().__init__(BasicBlock, [2, 2, 2, 2])
        self.fc = nn.Linear(512, 10)


        # self.sp = dom.Linear(32, 10)
    def split(self):
        return nn.Sequential(
            self.conv1, 
            self.bn1, 
            self.relu,
            self.maxpool,
            self.layer1, self.layer2, self.layer3, self.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # self.linear
            ), nn.Sequential(
                self.fc
                )
    def forward(self, x: Tensor) -> Tensor:
        # refer split function
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(x, 1)
        x = self.fc(x)