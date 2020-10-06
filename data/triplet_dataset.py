import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms
import torchvision.transforms.functional as TF
import torch

from tqdm.auto import tqdm

from data.triplet_database import Triplet_Database
from data.abstract_dataset import Abstract_Dataset


class Triplet_Dataset(Abstract_Dataset):
    """
    Dataset selects a "triplet" pair of anchor, positive and <number_negative_samples> negative instances specified in the 2DInstances.txt file.
    The pairs are chosen at construction time of the dataset via a Triplet_Database.
    Total number of indices in the dataset: number of rows in 2DInstances.txt

    We sample 4 types of positives:
        -   (SSD): Same scan, same object, different view
        -   (OSD): Other scan, same object, different view
        -   (COA): Same or other scan, same class, other object, ambiguity between anchor and other
        -   (OSR): Other scan, same object, rigid movement happened

    We sample 5 types of negatives:
        -   (OAC): Other room, any class
        -   (SAC): Same room, any class
        -   (OSC): Other room, same class
        -   (SCA): Same room, other scan, same class, other instance, no ambiguity to anchor instance, rigid movement
        -   (AVB): Same room, same scan, anchor no longer visible, but something in the background of anchor still is

    Each entry in a triplet has the following structure:
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

    One sample has the structure:
    {
        "anchor": <entry>,
        "pos": <entry>,
        "neg": <list of entries>
    }

    See also: <REPO_ROOT>/util/FrameFilter for information about the file 2DInstances.txt
    """

    # transform object that rotates horizontal images to vertical images and pads them to be of square resolution.
    rotate_vertical_transform = transforms.Compose([
        transforms.Pad(padding=(0, 210), fill=0),  # padding needed for rotation to keep all information
        transforms.Lambda(lambda x: TF.rotate(x, -90)),
    ])

    # original image size from the 3RScan dataset (images from the sequences folder).
    orig_width = 960
    orig_height = 540

    def __init__(self,
                 root_path,
                 seed=42,
                 number_negative_samples=1,
                 positive_sample_probabilities=[0.25, 0.25, 0.25, 0.25],
                 pos_minimum_visibility=[0.0, 0.0],
                 pos_maximum_visibility=[1.0, 1.0],
                 negative_sample_probabilities=[0.5, 0.5, 0, 0, 0],
                 neg_background_overlap_minimum=128 * 128,
                 sample_treshold=1,
                 sample_treshold_per_category=True,
                 sample_subset_percentage=0.5,
                 sample_fallback=False,
                 bbox_data_aligned_vertically=True,
                 transform=None,
                 cache=False,
                 preload_all=False,
                 verbose=False):
        """
        Constructs a Triplet_Dataset

        :param root_path: path/to/2DInstances.txt and path/to/<all_scan_subdirectories>
        :param seed for reproducibility of sampling
        :param number_negative_samples: how many negative samples should be selected. default: 1
        :param positive_sample_probabilities: With which probability should a positive sample come from one of the 4 types of positives.
                If e.g. only the first type of positives should be sampled, then specify this with [1, 0, 0, 0].
                The list must always add up to 1 and contain only values v with 0<=v<=1.
                Default: uniform distribution [0.25, 0.25, 0.25, 0.25]
        :param pos_minimum_visibility: list with two entries. First: minimum truncation value for all positive samples from all types no matter the probabilities.
                Second: minimum occlusion value for all positive samples from all types no matter the probabilities. Used to specify the minimum for any positive sample no matter the type.
        :param negative_sample_probabilities: With which probability should a negative sample come from one of the 5 types of negatives.
                If e.g. only the first type of negatives should be sampled, then specify this with [1, 0, 0, 0, 0].
                The list must always add up to 1 and contain only values v with 0<=v<=1.
                Default: uniform distribution among the first two negative types [0.5, 0.5, 0, 0, 0]
        :param neg_background_overlap_minimum: For the negative sample type "AVB": how much bbox overlap in the anchor image does the negative instance need to have (in pixels).
        :param sample_treshold: after n positive and n negative samples are found for one anchor, we no longer iterate over the rest of available samples.
               Instead we randomly sample from the samples that we have found thus far. Default: -1, meaning we will always go through all available samples. Any value <1 will have that effect.
        :param sample_subset_percentage: how many percentage of all samples should be searched for a pos/neg sample at random. Default: 1.0. Must be 0<=p<=1
        :param sample_fallback: if pos/neg could not be found in the sample_subset defined by sample_subset_percentage, then search in all samples as fallback. Default: False
        :param transform: tranform that should be applied to the images and bboxes
        :param cache: if a sample should be cached in RAM once it was loaded. Default: false
        :param preload_all: if all samples should be preloaded (when cache=True this will store all samples in cache at construction time). Default: false
        :param verbose: If true, will tqdm the remaining time for precomputation of triplets and preloading of samples. Default: false
        :param bbox_data_aligned_vertically: whether or not the bounding-box data in each row of 2DInstances.txt is aligned to a vertical image orientation (also see the Rio_Renderer code).
        """

        Abstract_Dataset.__init__(self,
                                  root_path=root_path,
                                  number_negative_samples=number_negative_samples,
                                  seed=seed,
                                  bbox_data_aligned_vertically=bbox_data_aligned_vertically,
                                  transform=transform,
                                  cache=cache,
                                  verbose=verbose)

        # save all subtype specific constructor arguments
        self.positive_sample_probabilities = positive_sample_probabilities
        self.negative_sample_probabilities = negative_sample_probabilities
        self.preload_all = preload_all

        # add sanity check
        if self.number_negative_samples > sample_treshold:
            raise ValueError(
                f"Sampling more negatives than could be selected. number_negative_samples: {self.number_negative_samples}, sample_treshold: {sample_treshold}")

        # create database
        self.database = Triplet_Database(root_path=root_path,
                                         seed=seed,
                                         pos_minimum_visibility=pos_minimum_visibility,
                                         pos_maximum_visibility=pos_maximum_visibility,
                                         neg_background_overlap_minimum=neg_background_overlap_minimum,
                                         sample_treshold=sample_treshold,
                                         sample_treshold_per_category=sample_treshold_per_category,
                                         sample_subset_percentage=sample_subset_percentage,
                                         sample_fallback=sample_fallback,
                                         verbose=verbose)

        self.triplets_possibilities, self.instances = self.database.get()

        # create anchor, pos, neg list where the n-th entry is meant for the n-th sample of this dataset
        self.anchors = []
        self.positives = []
        self.negatives = []

        # precompute triplets
        self.precompute_triplets()

        # set size value to number of instances (now they only contain those instances for that we could find triplets)
        self.size = len(self.anchors)

        # make one run over complete dataset for preloading
        if self.preload_all:
            n = range(self.__len__())
            if self.verbose:
                print("Start preloading...")
                n = tqdm(n)
            for idx in n:
                self.__getitem__(idx)

    def precompute_triplets(self):
        # precompute the triplets at construction time. At __getitem__ we only need to load image data from disk.
        n = range(len(self.instances))
        if self.verbose:
            n = tqdm(n)
        self.removed_idx = []

        self.triplet_dict = {}

        # sample triplets
        if self.verbose:
            print("Start sampling triplets...")
        for idx in n:
            anchor = self.instances[idx]
            pos = self.triplets_possibilities[str(anchor)]["pos"]
            neg = self.triplets_possibilities[str(anchor)]["neg"]
            valid, positive, negatives = self.sample_triplets(pos, neg)

            if valid:
                # these lists are for the idx-based iteration of torch.util.dataset
                self.anchors.append(anchor)
                self.positives.append(positive)
                self.negatives.append(negatives)

                # this allows an O(1) access to a triplet by its key
                self.triplet_dict[str(anchor)] = {
                    "anchor": anchor,
                    "pos": positive,
                    "neg": negatives
                }
            else:
                self.removed_idx.append(idx)

        if self.verbose:
            print(f"Could not find triplet for {len(self.removed_idx)}/{len(self.instances)} anchors.")

        # remove all instances for which we could not find a valid triplet
        self.instances = [i for idx, i in enumerate(self.instances) if idx not in self.removed_idx]

    def sample_positives(self, positives):
        probs = self.calculate_probabilities(positives, self.positive_sample_probabilities)

        valid, samples = self.sample_from_types(positives, probs, 1)

        if valid:
            samples = samples[0]  # we do not need a list when only ever having one positive sample at a time

        return valid, samples

    def sample_negatives(self, negatives):
        probs = self.calculate_probabilities(negatives, self.negative_sample_probabilities)

        valid, samples = self.sample_from_types(negatives, probs, self.number_negative_samples)

        return valid, samples

    def calculate_probabilities(self, lists, probs):
        """
        For a given list of lists and probability to sample from each list, we redistribute the probability such that:

        - Only probabilities that are >0 are still >0
        - Only probabilities for a non-empty list are >0
          Will set probability to 0 for empty lists and redistribute among all non-empty lists with a probability >0
        - probabilities still sum up to the same number as before

        :param lists: list of lists
        :param probs: probabilities to sample from each list

        :return: altered probabilities to sample from each list satisfying the above criteria
        """

        # how much probability needs to be redistributed among remaining probabilities to sum to 1?
        zero_sum = np.sum([p if len(lists[i]) == 0 else 0 for i, p in enumerate(probs)])

        # set probability to 0 if no list entry present
        probs = [p if len(lists[i]) > 0 else 0 for i, p in enumerate(probs)]

        # how many probabilities are non-zero and can get an increment from the others
        non_zero_probs = np.count_nonzero(probs)

        if non_zero_probs > 0:
            # calculate the increment for each remaining probability
            increment = zero_sum / non_zero_probs

            # increment each probability that is not zero
            probs = [p + increment if p > 0 else 0 for p in probs]

        return probs

    def sample_from_types(self, types, probs, n):
        """
        Samples n items from types where types contains k lists and each list contains a different number of items.
        A type k is selected with probability probs[k] and the resulting item is sampled uniformly from types[k].

        :param types: list of lists where types contains k lists and each list contains a different number of items
        :param probs: list of floats with len(probs) == len(types) and probs[k] is sample probability for types[k]
        :param n: how many samples to take with above procedure

        :return: (True, samples): if at least one probability is greater than zero we can return all samples.
                    If n=1 then samples is just one sample, if n>1 then samples is a list of samples.
                 (False, None): if all sample probabilities are zero, then we return no samples.
        """
        try:
            # select from which type to sample from for each sample
            types_per_sample = np.random.choice(len(types), n, p=probs)

            # count how many samples to select from each type
            unique, counts = np.unique(types_per_sample, return_counts=True)
            sample_count_per_type = dict(zip(unique, counts))

            # sample indices per type at random without replacement
            sample_indices_per_type = {k: np.random.choice(len(types[k]), v, replace=False) for k, v in
                                       sample_count_per_type.items()}

            # create list of samples from indices
            samples = [types[k][index] for k, v in sample_indices_per_type.items() for index in v]

            return True, samples
        except:
            return False, None

    def sample_triplets(self, all_possible_positives, all_possible_negatives):
        # find positive instance: only one positive item
        valid_pos, positive_instance = self.sample_positives(all_possible_positives)

        # find negative instance(s): list of length self.number_negative_samples
        valid_neg, negative_instances = self.sample_negatives(all_possible_negatives)

        return valid_pos and valid_neg, positive_instance, negative_instances


    def __getitem__(self, idx):
        """
        Retrieve the idx-th instance from the dataset.

        :param idx: index in the dataset
        :return: Dict with format
        {
            "anchor": <entry>,
            "pos": <entry>,
            "neg": <list of entries>
        }

        where one entry has the format

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
            }
            "number_other_instances": how many other instances visible in this image,
            "other_instance_ids": which other instances visible in this image,
            "image": image defined by the above values
        }
        """

        anchor = self.load_instance(self.anchors[idx], self.database)
        p = self.load_instance(self.positives[idx], self.database)
        n = [self.load_instance(neg, self.database) for neg in self.negatives[idx]]

        return {
            "anchor": anchor,
            "pos": p,
            "neg": n
        }

    def __len__(self):
        return self.size


def test():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from torchvision import transforms
    import torchvision.transforms.functional as TF
    transform = transforms.Compose([
        Triplet_Dataset.rotate_vertical_transform,
        transforms.Resize(256),
        transforms.ToTensor()
    ])

    d = Triplet_Dataset(root_path="/home/lukas/datasets/3RScan/3RScan-10/all_in_one",
                        transform=transform,
                        number_negative_samples=4,
                        positive_sample_probabilities=[0.25, 0.25, 0.25, 0.25],
                        negative_sample_probabilities=[0, 0, 0, 0, 1],
                        pos_minimum_visibility=[0, 0],
                        sample_treshold=9,
                        neg_background_overlap_minimum=128 * 128,
                        sample_subset_percentage=0.1,
                        sample_treshold_per_category=True,
                        bbox_data_aligned_vertically=False,
                        sample_fallback=True,
                        verbose=True)

    loader = torch.utils.data.DataLoader(d, batch_size=2)

    for sample in loader:
        print(sample["anchor"]["instance_id"])

    save_dir = "/home/lukas/datasets/3RScan/3RScan-10/analysis/"
    save_dir += "seed_" + str(d.seed) + "_pos_" + str(d.positive_sample_probabilities) + "_neg_" + str(
        d.negative_sample_probabilities)

    from pathlib import Path
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    title = "TRIPLET PAIR"

    print("Will save {} triplets to {}".format(d.size, save_dir))
    for idx in tqdm(range(d.size)):
        item = d.__getitem__(idx)

        images = [item["anchor"]["image"], item["pos"]["image"]]
        images.extend([item["neg"][k]["image"] for k in range(len(item["neg"]))])

        labels = [item["anchor"]["label"], item["pos"]["label"]]
        labels.extend([item["neg"][k]["label"] for k in range(len(item["neg"]))])

        bboxes = [item["anchor"]["bbox"], item["pos"]["bbox"]]
        bboxes.extend([item["neg"][k]["bbox"] for k in range(len(item["neg"]))])

        i = 0
        fig, ax = plt.subplots(1, len(images), figsize=(2048 / 96.0, 2048 / 96.0))
        fig.suptitle(title, fontsize=16)
        for image, label, bbox in zip(images, labels, bboxes):
            if i == 0:
                ax[i].set_title("ANCHOR: " + label)
            elif i == 1:
                ax[i].set_title("POS: " + label)
            elif i >= 2:
                ax[i].set_title(f"NEG-{i - 2}: " + label)

            x = bbox["x"]
            y = bbox["y"]
            width = bbox["w"]
            height = bbox["h"]

            image_crop = image[:, y:y + height, x:x + width]
            ax[i].imshow(image_crop.permute(1, 2, 0).numpy())

            # rect = patches.Rectangle((x, y),
            #                         width,
            #                         height,
            #                         linewidth=2, edgecolor=np.random.rand(3, ), facecolor='none')
            # ax[i].imshow(image.permute(1, 2, 0).numpy())
            # ax[i].add_patch(rect)
            i += 1
        plt.show()
        # plt.savefig(save_dir + "/triplet-" + str(idx) + ".png")
        plt.cla()
        plt.close()


if __name__ == "__main__":
    # execute only if run as a script
    test()
