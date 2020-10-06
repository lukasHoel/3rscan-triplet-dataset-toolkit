from data.abstract_dataset import Abstract_Dataset
from data.invariance_database import Invariance_Database
from data.triplet_dataset import Triplet_Dataset

import math

class Invariance_Dataset(Abstract_Dataset):
    """
    The Invariance_Dataset takes in an Invariance_Database and provides its anchor2positive pairs in a similar format as does the Triplet_Dataset (see for reference).
    Optionally, the Invariance_Dataset will also load the corresponding negatives for each anchor2positive pair based on the matching of a provided Triplet_Dataset.
    The Invariance_Dataset allows to filter all anchor2positive pairs via a transformation and illuminance range.

    The Invariance_Dataset provides each sample as a triplet (negative might be None):
    {
        "anchor": <entry>,
        "pos": <entry>,
        "neg": <list of entries>
    }

    Each entry of a triplet has the format:
    {
        "reference": reference scan id,
        "scan": scan id,
        "frame_nr": frame number,
        "instance_id": instance id,
        "label": semantic label,
        "bbox": {
          "x": upper-left corner,
          "y": upper-left corner,
          "w": width,
          "h": height
        },
        "visibility": visibility,
        "number_other_instances": how many other instances visible in this image,
        "other_instance_ids": which other instances visible in this image,
        "image": image defined by the above values
    }
    """
    def __init__(self,
                 invariance_database: Invariance_Database,
                 triplet_dataset: Triplet_Dataset=None,
                 load_negatives_from_triplet_dataset=True,
                 transformation_range=[0.0, 1.0],
                 illuminance_range=[0.0, 1.0],
                 transform=None,
                 bbox_data_aligned_vertically=True,
                 use_cache=False):
        """
        Constructs an Invariance_Dataset. It filterse all available anchor2positive pairs from the invariance_database given the concrete configuration
        and optionally loads the corresponding negative samples based on the triplet_dataset.

        :param invariance_database: from which all anchor2positive pairs will be filtered.
        :param triplet_dataset: will load the negatives of each anchor2positive pair based on this configuration (if an anchor does not appear in the triplet_dataset that anchor will be filtered out)
        :param load_negatives_from_triplet_dataset: if negatives should be loaded or not
        :param transformation_range: [min_ratio, max_ratio] that each anchor2positive pair must fulfill to be selected (based on the calculations of the invariance_Database)
        :param illuminance_range: [min_ratio, max_ratio] that each anchor2positive pair must fulfill to be selected (based on the calculations of the invariance_Database)
        :param transform: PyTorch transform object to apply to all images
        :param bbox_data_aligned_vertically: whether or not the bounding-box data in each row of 2DInstances.txt is aligned to a vertical image orientation (also see the Rio_Renderer code).
        :param use_cache: If true, will cache all selected samples on RAM once they were accessed.
        """

        Abstract_Dataset.__init__(self,
                                  root_path=invariance_database.root_path,
                                  number_negative_samples=0 if not load_negatives_from_triplet_dataset else triplet_dataset.number_negative_samples,
                                  seed=invariance_database.seed,
                                  bbox_data_aligned_vertically=bbox_data_aligned_vertically,
                                  transform=transform,
                                  cache=use_cache,
                                  verbose=False)

        # save all subtype specific constructor arguments
        self.invariance_database = invariance_database
        self.triplet_dataset = triplet_dataset
        self.load_negatives_from_triplet_dataset = load_negatives_from_triplet_dataset and triplet_dataset is not None
        self.transformation_range = transformation_range
        self.illuminance_range = illuminance_range

        self.filtered_pairs_same_scan = self.filter_pairs(self.invariance_database.pairs_same_scan)
        self.filtered_pairs_different_scan = self.filter_pairs(self.invariance_database.pairs_different_scan)

        self.size = len(self.filtered_pairs_same_scan) + len(self.filtered_pairs_different_scan)

    def filter_pairs(self, pairs):

        filtered_indices_by_invariance = self.filter_by_invariances(pairs)

        filtered_indices_by_triplet_dataset = self.filter_by_triplet_dataset(pairs)

        filtered_indices = [*filtered_indices_by_invariance, *filtered_indices_by_triplet_dataset]

        filtered_pairs = [p for i, p in enumerate(pairs) if i not in filtered_indices]

        return filtered_pairs

    def filter_by_triplet_dataset(self, pairs):
        filtered_indices = []

        if self.triplet_dataset is not None:
            for idx, pair in enumerate(pairs):
                a = str(pair["anchor"])
                p = str(pair["pos"])
                triplets = self.triplet_dataset.triplet_dict

                found = (a in triplets and str(triplets[a]["pos"]) == p) or (p in triplets and str(triplets[p]["pos"]) == a)

                if not found:
                    filtered_indices.append(idx)

        return filtered_indices

    def filter_by_invariances(self, pairs):
        filtered_indices = []

        for idx, pair in enumerate(pairs):
            transformation = pair.get("transformation", None)
            illuminance = pair.get("illuminance", None)

            if math.isnan(illuminance["ratio"]):
                illuminance["ratio"] = 0.0

            if transformation is None or illuminance is None:
                filtered_indices.append(idx)
            elif not self.ratio_in_range(transformation["ratio"], self.transformation_range)\
                or not self.ratio_in_range(illuminance["ratio"], self.illuminance_range):
                filtered_indices.append(idx)

        return filtered_indices

    def ratio_in_range(self, ratio, range):
        return range[0] <= ratio and ratio <= range[1]

    def __getitem__(self, idx):
        if idx < len(self.filtered_pairs_same_scan):
            # load from pairs_same_scan:
            pair = self.filtered_pairs_same_scan[idx]
        else:
            # load from pairs_different_scan
            pair = self.filtered_pairs_different_scan[idx]

        anchor = self.load_instance(pair["anchor"], self.invariance_database.triplet_database)
        pos = self.load_instance(pair["pos"], self.invariance_database.triplet_database)

        result = {
            "anchor": anchor,
            "pos": pos
        }

        if self.load_negatives_from_triplet_dataset:
            triplets = self.triplet_dataset.triplet_dict
            a = str(pair["anchor"])
            p = str(pair["pos"])
            if a in triplets:
                negs = triplets[a]["neg"]
                result["neg"] = [self.load_instance(neg, self.invariance_database.triplet_database) for neg in negs]
            elif p in triplets:
                negs = triplets[p]["neg"]
                result = {
                    "anchor": pos,
                    "pos": anchor,
                    "neg": [self.load_instance(neg, self.invariance_database.triplet_database) for neg in negs]
                }
            else:
                raise ValueError(f"Cannot find anchor2pos pair in triplets of triplet_dataset. {a}, {p}")

        return result

    def __len__(self):
        return self.size


def test():
    import torchvision
    from tqdm.auto import tqdm
    transform = torchvision.transforms.Compose([
        Triplet_Dataset.rotate_vertical_transform,
        torchvision.transforms.Resize(256),
        torchvision.transforms.ToTensor()
    ])

    triplet_dataset = Triplet_Dataset(root_path="/home/lukas/datasets/3RScan/3RScan-10/all_in_one",
                        transform=transform,
                        number_negative_samples=4,
                        positive_sample_probabilities=[1, 0, 0, 0],
                        negative_sample_probabilities=[0.5, 0.5, 0, 0, 0],
                        pos_minimum_visibility=[0, 0],
                        sample_treshold=9,
                        neg_background_overlap_minimum=128 * 128,
                        sample_subset_percentage=0.1,
                        sample_treshold_per_category=True,
                        bbox_data_aligned_vertically=False,
                        sample_fallback=True,
                        verbose=True)

    invariance_database = Invariance_Database(root_path="/home/lukas/datasets/3RScan/3RScan-10/all_in_one",
                                              triplet_database=triplet_dataset.database,
                                              sample_treshold_per_anchor=9,
                                              debug=False,
                                              render_size=64,
                                              verbose=True)

    invariance_dataset = Invariance_Dataset(invariance_database,
                                            triplet_dataset=triplet_dataset,
                                            transformation_range=[0.8, 1.0],
                                            bbox_data_aligned_vertically=False,
                                            transform=transform)

    print(len(invariance_dataset), len(triplet_dataset))

    import matplotlib.pyplot as plt

    save_dir = "/home/lukas/datasets/3RScan/3RScan-10/analysis_invariance/"
    save_dir += "transformation_filter_" + str(invariance_dataset.transformation_range) + "illuminance_filter" + str(invariance_dataset.illuminance_range)

    from pathlib import Path
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    title = "ANCHOR to POS PAIR"

    print("Will save {} anchor2pos pairs to {}".format(invariance_dataset.size, save_dir))
    for idx in tqdm(range(invariance_dataset.size)):
        item = invariance_dataset.__getitem__(idx)

        images = [item["anchor"]["image"], item["pos"]["image"]]

        labels = [item["anchor"]["label"], item["pos"]["label"]]

        bboxes = [item["anchor"]["bbox"], item["pos"]["bbox"]]

        i = 0
        fig, ax = plt.subplots(1, len(images), figsize=(2048 / 96.0, 2048 / 96.0))
        fig.suptitle(title, fontsize=16)
        for image, label, bbox in zip(images, labels, bboxes):
            if i == 0:
                ax[i].set_title("ANCHOR: " + label)
            elif i == 1:
                ax[i].set_title("POS: " + label)

            x = bbox["x"]
            y = bbox["y"]
            width = bbox["w"]
            height = bbox["h"]

            image_crop = image[:, y:y + height, x:x + width]
            ax[i].imshow(image_crop.permute(1, 2, 0).numpy())

            #rect = patches.Rectangle((x, y),
            #                         width,
            #                         height,
            #                         linewidth=2, edgecolor=np.random.rand(3, ), facecolor='none')
            # ax[i].imshow(image.permute(1, 2, 0).numpy())
            #ax[i].add_patch(rect)
            i += 1
        plt.show()
        #plt.savefig(save_dir + "/triplet-" + str(idx) + ".png")
        plt.cla()
        plt.close()


if __name__ == "__main__":
    # execute only if run as a script
    test()