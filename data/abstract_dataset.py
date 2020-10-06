import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from os.path import join

from data.data_util import transform_bbox, apply_bbox_sanity_check

from data.triplet_database import Triplet_Database

class Abstract_Dataset(Dataset):
    """
    An Abstract_Dataset goes through instances of 2DInstances.txt found in the <root_path> and matches multiple instances as triplets (anchor, positive, negative).
    The number of negative samples to search for one triplet is configurable and can also be zero.
    The Abstract_Dataset parses all triplets and loads their corresponding images and other metadata (bounding-box in the image, scan-id, etc.) into a PyTorch datastructure.

    See also: <REPO_ROOT>/util/FrameFilter for information about the file 2DInstances.txt
    """
    def __init__(self,
                 root_path,
                 number_negative_samples,
                 seed=42,
                 bbox_data_aligned_vertically=True,
                 transform=None,
                 cache=False,
                 verbose=False):
        """
        Contructs an Abstract_Dataset.

        :param root_path: path/to/2DInstances.txt and path/to/<all_scan_subdirectories>
        :param number_negative_samples: how many negative samples to match per triplet (can also be zero).
        :param seed: random seed
        :param bbox_data_aligned_vertically: whether or not the bounding-box data in each row of 2DInstances.txt is aligned to a vertical image orientation (also see the Rio_Renderer code).
        :param transform: PyTorch transform object to apply to all images when loading them
        :param cache: whether or not to store all loaded images in RAM once they were accessed.
        :param verbose: If true, will print tqdm information on all processes.
        """

        # save all constructor arguments
        self.number_negative_samples = number_negative_samples
        self.transform = transform
        self.use_cache = cache
        self.verbose = verbose
        self.seed = seed
        self.root_path = root_path
        self.bbox_data_aligned_vertically = bbox_data_aligned_vertically

        # initialize empty cache for accessed items of this dataset
        self.cache = {}

        # random seed
        np.random.seed(seed)

    def load_instance(self, instance, triplet_database: Triplet_Database):
        """
        Loads a row of 2DInstances.txt by parsing its columns, loading the image from disk, loading the bbox and
        applying transformations. If specified, will store the loaded instance in cache.

        :param instance: a row of 2DInstances.txt (an entry in self.instances)
        :return: loaded instance
        """

        # check if available in cache
        key = str(instance)
        cached_instance = self.cache.get(key, None)
        if cached_instance is not None:
            return cached_instance

        # parse instance into dict
        parsed_instance = triplet_database.parse_instance(instance)

        # load image
        frame_name = "frame-{:06d}.color.jpg".format(parsed_instance["frame_nr"])
        image_path = join(self.root_path, parsed_instance["scan"], "sequence", frame_name)
        image = Image.open(image_path)
        bbox = parsed_instance["bbox"]

        # transform image and bbox
        if self.transform is not None:
            image = self.transform(image)
            bbox = transform_bbox(parsed_instance["bbox"],
                                       self.bbox_data_aligned_vertically,
                                       self.transform,
                                       self.cache)

        # make sure that bbox is valid in all cases
        bbox = apply_bbox_sanity_check(bbox, parsed_instance)

        # convert bbox into dict
        bbox_dict = {
            "x": bbox[0],
            "y": bbox[1],
            "w": bbox[2] - bbox[0],
            "h": bbox[3] - bbox[1]
        }

        # construct loaded dict
        loaded_instance = {
            "image": image,
            "bbox": bbox_dict,
            "label": parsed_instance["label"],
            "instance_id": parsed_instance["instance_id"],
            "reference": parsed_instance["reference"],
            "scan": parsed_instance["scan"],
            "frame_nr": parsed_instance["frame_nr"],
        }

        # store in cache
        if self.use_cache:
            self.cache[key] = loaded_instance

        return loaded_instance