import torch
import torchvision.models as models

class ResNet(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(ResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.start = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.slice1 = resnet.layer1
        self.slice2 = resnet.layer2
        self.slice3 = resnet.layer3
        self.slice4 = resnet.layer4
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.start(X)
        h = self.slice1(h)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]

if __name__ == "__main__":
    model = ResNet(requires_grad=True)
    print(model)

    def count_parameters(model):
        """Given a model return total number of parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(model))
    print(count_parameters(model.slice1))
    print(count_parameters(model.slice2))
    print(count_parameters(model.slice3))
    print(count_parameters(model.slice4))

    from torchvision import transforms
    from data.triplet_dataset import Triplet_Dataset

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

    for batch in loader:
        batch = d.triplets_as_batches(batch)
        enc = model(batch["image"])

        for e in enc:
            print(e.shape)

        break