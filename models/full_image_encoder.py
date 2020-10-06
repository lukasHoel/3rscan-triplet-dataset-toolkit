import torch
from models.backbone.vgg16 import VGG16

class FullImageEncoder(torch.nn.Module):
    def __init__(self, requires_grad=True, encoder_type="VGG16"):
        super(FullImageEncoder, self).__init__()
        if encoder_type == "VGG16":
            self.model = VGG16(requires_grad=requires_grad)
        else:
            raise ValueError("encoder_type {} not supported".format(encoder_type))

    def forward(self, x):
        return self.model(x["image"])

if __name__ == "__main__":
    model = FullImageEncoder()
    print(model)