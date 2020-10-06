import torch
from models.backbone.vgg16 import VGG16
from models.backbone.resnet import ResNet
import torchvision.transforms as tf

from models.layers.rmac_layer import RMACLayer

class BoundingBoxEncoder(torch.nn.Module):
    def __init__(self, requires_grad=True, encoder_type="VGG16", resize_shape=None, use_rmac_layer=False):
        super(BoundingBoxEncoder, self).__init__()
        if encoder_type == "VGG16":
            self.model = VGG16(requires_grad=requires_grad)
        elif encoder_type == "ResNet":
            self.model = ResNet(requires_grad=requires_grad)
        else:
            raise ValueError("encoder_type {} not supported".format(encoder_type))

        if resize_shape is not None:
            self.crop_transform = tf.Compose([
                tf.Lambda(lambda x: x.cpu() if torch.cuda.is_available() else x),
                tf.ToPILImage(),
                tf.Resize(resize_shape),
                tf.ToTensor(),
                tf.Lambda(lambda x: x.cuda() if torch.cuda.is_available() else x)
            ])
        else:
            self.crop_transform = None
        
        self.use_rmac_layer = use_rmac_layer
        self.rmac = RMACLayer()

    def forward(self, batch):
        # crop each image by its bounding-box
        # TODO: can I vectorize it?
        # Problem: if no crop_transform then images have not same size and cannot be cat into one tensor
        images = ()
        for i in range(batch["image"].shape[0]):
            # load bbox from batch
            x = batch["bbox"]["x"][i]
            y = batch["bbox"]["y"][i]
            w = batch["bbox"]["w"][i]
            h = batch["bbox"]["h"][i]

            # crop image along each channel
            img = batch["image"][i, :, y:y+h, x:x+w]

            # apply transform if it exists
            if self.crop_transform is not None:
                img = self.crop_transform(batch["image"][i])

            # add batch dimension for cat later
            img = img.unsqueeze(0)

            # add to tuple of images for cat later
            images += (img,)

        if self.crop_transform is not None:
            # cat all images into cropped tensor because all have same size here!
            images = torch.cat(images, dim=0)
            
            # encode
            encodings = self.model(images)

            # apply rmac if it is selected
            if self.use_rmac_layer:
                encodings = self.rmac(encodings)

            return encodings

        else:
            # loop through each image (non-vectorized!) because they have different sizes due to different bboxes
            encodings = []
            for image in images:
                # encode
                enc = self.model(image)

                # apply rmac if it is selected
                if self.use_rmac_layer:
                    enc = self.rmac(enc)

                encodings.append(enc)

            if self.use_rmac_layer:
                # convert list of encodings where each element is a list of output layer encodings into
                # a list of output layer encodings where each element is one tensor with batch_size = number of images (encodings)
                encodings = [torch.cat(tuple([encodings[i][layer] for i in range(len(encodings))]), dim=0) for layer in range(len(encodings[0]))]
            else:
                # we want to return the encodings as a tensor for each layer for the next steps to work more smoothly.
                # if this is a problem later, we need to fix the dataloader method "output_to_triplets" to also take in a list of all image encodings instead of tensors
                raise ValueError("use_rmac_layer must be true when resize_shape is None.")

            return encodings

if __name__ == "__main__":
    model = BoundingBoxEncoder()
    print(model)
