import torch
from models.backbone.resnet import ResNet

from models.layers.reobj_conv_block import REObj_Conv_Block


class REObjEncoder(torch.nn.Module):
    def __init__(self, requires_grad=True):
        super(REObjEncoder, self).__init__()
        self.model = ResNet(requires_grad=requires_grad)
        self.conv_block = REObj_Conv_Block()

    def forward(self, batch):
        images = batch["image"]

        if images.shape[2] != 224 or images.shape[3] != 224:
            raise ValueError("Re-OBJ expects input in shape 224x224, but input shape was", images.shape)

        encodings = self.model(images)[-1] # only use last layer of encoding as this is what the re-obj baseline does
        embeddings = self.conv_block(encodings)

        return [embeddings] # return list of size 1 to have same API as the other encoders that return list of mulitple encodings

if __name__ == "__main__":
    model = REObjEncoder()
    print(model)
