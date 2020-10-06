import torch
from models.backbone.vgg16 import VGG16

class REObj_Conv_Block(torch.nn.Module):
    def __init__(self):
        super(REObj_Conv_Block, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.max1 = torch.nn.MaxPool2d(kernel_size=2, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1)
        self.max2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv3 = torch.nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, padding=1)
        self.max3 = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.max3(x)

        return x

if __name__ == "__main__":
    from models.backbone.resnet import ResNet
    from torchvision import transforms
    from data.triplet_dataset import Triplet_Dataset

    model = ResNet(requires_grad=False)

    transform = transforms.Compose([
        Triplet_Dataset.rotate_vertical_transform,
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    d = Triplet_Dataset(root_path="/home/lukas/datasets/3RScan/3RScan-10/all_in_one",
                        transform=transform,
                        number_negative_samples=1,
                        positive_sample_probabilities=[0.25, 0.25, 0.25, 0.25],
                        negative_sample_probabilities=[1, 0, 0, 0, 0],
                        pos_minimum_visibility=[0, 0],
                        sample_treshold=9,
                        neg_background_overlap_minimum=128 * 128,
                        sample_subset_percentage=0.1,
                        sample_treshold_per_category=True,
                        bbox_data_aligned_vertically=False,
                        sample_fallback=True,
                        verbose=False)

    loader = torch.utils.data.DataLoader(d, batch_size=2)

    reobj_conv_block = REObj_Conv_Block()

    for batch in loader:
        batch = d.triplets_as_batches(batch)
        enc = model(batch["image"])
        enc = reobj_conv_block(enc[-1])

        print(enc.shape)

        break