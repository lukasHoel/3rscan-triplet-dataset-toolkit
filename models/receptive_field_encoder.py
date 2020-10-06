import torch
from models.backbone.vgg16 import VGG16
import torchvision.transforms as tf

from models.layers.rmac_layer import RMACLayer

class ReceptiveFieldEncoder(torch.nn.Module):
    '''
    EXAMPLE:

    ----------------------------------------------------------------
    INPUT IMAGE: 960x960
    Bounding Box: [100:150, 100:150]
    ----------------------------------------------------------------

    Input --> Conv --> Conv
    FIRST LAYER: [1, 64, 960, 960]    --> 5x5    receptive field

    TWO POSSIBILITIES:
      a) Use [100:150, 100:150] in first layer
          --> corresponds to features of [98:152, 98:152] in input

      b) Use [102:148, 102:148] in first layer
          --> corresponds to features of [100:150, 100:150] in input

    -----------------------------------------------------------------

    Input --> Conv --> Conv --> Max-Pool --> Conv --> Conv
    SECOND LAYER: [1, 128, 480, 480]  --> 14x14  receptive field

    TWO POSSIBILITIES:
      a) Downscale [100:150, 100:150] to [50:75, 50:75] and use that in second layer
          --> corresponds to features of [48:77, 48:77] after Max-Pool layer
          --> corresponds to features of [96:155, 96:155] before Max-Pool layer
          --> corresponds to features of [94:157, 94:157] in input

      b) Downscale [100:150, 100:150] to [50:75, 50:75], but only use [53:71, 53:71]
          --> corresponds to features of [51:73, 51:73] after Max-Pool layer
          --> corresponds to features of [102:147, 102:147] before Max-Pool layer
          --> corresponds to features of [100:149, 100:149] in input

    -----------------------------------------------------------------

    Pros & Cons

    a) + Easy to produce new bounding-box --> just downscale
       + Will grow a larger "context" around actual bounding-box the deeper in the network
       - Not only bounding-box as input, but more

    b) + Possible to extract features that closer match the bounding-box
       - But not possible to do it 100% exact (also only an approximation)

    Decision: Use variant a) as we want to have a context around image anyways?

    -----------------------------------------------------------------

    Problem: Reference and Rescan bounding-boxes need not be of equal size!

    --> Need to downscale bbox or features to a fixed size fo comparison of feature vectors

    --> Option 1: Just choose min(x) and min(y) for the current pair of features to compare

    --> Option 2: Use PCA to downscale features to fixed size

    --> Option 3: Only use this model with R-MAC techniques
    '''
    def __init__(self, requires_grad=True, encoder_type="VGG16", use_rmac_layer=False):
        super(ReceptiveFieldEncoder, self).__init__()
        if encoder_type == "VGG16":
            self.model = VGG16(requires_grad=requires_grad)
        else:
            raise ValueError("encoder_type {} not supported".format(encoder_type))
        
        self.use_rmac_layer = use_rmac_layer
        self.rmac = RMACLayer()

    def forward(self, batch):

        # encode full images first
        encodings = self.model(batch["image"])

        # then only keep receptive field of encoding w.r.t bounding box for each image
        # TODO: can I vectorize it? do I need to? there is no heavy computation here! 
        cropped_encodings = [() for i in range(len(encodings))]
        for i in range(batch["image"].shape[0]):
            _, src_height, src_width = batch["image"][i].shape # TODO h/w in correct order?

            # load bbox from batch
            x = batch["bbox"]["x"][i]
            y = batch["bbox"]["y"][i]
            w = batch["bbox"]["w"][i]
            h = batch["bbox"]["h"][i]

            # crop encodings from all layers
            for k, enc in enumerate(encodings):
                target_height, target_width = enc.shape[2:] # TODO h/w in correct order?

                x_t = int(1.0 * x / src_width * target_width)
                w_t = int(1.0 * w / src_width * target_width)
                y_t = int(1.0 * y / src_height * target_height)
                h_t = int(1.0 * h / src_height * target_height)

                cropped_enc = encodings[k][i, :, y_t:y_t+h_t, x_t:x_t+w_t].unsqueeze(0)

                # apply rmac if it is selected
                if self.use_rmac_layer:
                    cropped_enc = self.rmac([cropped_enc])[0] # fix because rmac wants to have lists of encodings

                cropped_encodings[k] += (cropped_enc,) # TODO h/w in correct order?

        cropped_encodings = [torch.cat(enc, dim=0) for enc in cropped_encodings]

        return cropped_encodings

if __name__ == "__main__":
    model = ReceptiveFieldEncoder()
    print(model)