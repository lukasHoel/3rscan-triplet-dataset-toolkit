import torch
from models.backbone.vgg16 import VGG16

class RMACLayer(torch.nn.Module):
    def __init__(self):
        super(RMACLayer, self).__init__()

    def forward(self, encs, regions=None):
        """
        Forward pass

        :param encs: list of encodings
        :param regions: list of dict (each region-index corresponds to enc with that index) with format:
            {
                "x": int,
                "y": int,
                "w": int,
                "h": int
            }

        :return: max value per activation map in the whole spatial domain or in given region
        """

        max_outputs = []
        for i, act_map in enumerate(encs):
            if regions is not None:
                x = regions[i]["x"]
                y = regions[i]["y"]
                w = regions[i]["w"]
                h = regions[i]["h"]
                act_map = act_map[:, :, y:y+h, x:x+w] # TODO h/w in correct order?

            spatial_flattened_act_map = torch.reshape(act_map, (act_map.shape[0], act_map.shape[1], -1))
            # spatial_flattened_act_map = act_map.view(act_map.shape[0], act_map.shape[1], -1)

            spatial_max_act_map = spatial_flattened_act_map.max(dim=2)
            max_outputs.append(spatial_max_act_map.values)

        return max_outputs

if __name__ == "__main__":
    model = RMACLayer()
    print(model)