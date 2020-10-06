import json
import csv

from os.path import join
import os

from tqdm.auto import tqdm

import uuid

from data.triplet_database import Triplet_Database

from torchvision.transforms import ToTensor, Resize, Compose, Lambda

from util.point_cloud_projection_and_rendering import Projector, RasterizePointsXYsBlending

from data.triplet_dataset import Triplet_Dataset
import torch
from PIL import Image
import numpy as np
import torchvision


class Invariance_Database():
    """
    The Invariance Database stores scores for transformation invariance and illuminance invariance between pairs of images of one specific instance.
    With the term "invariance" we reference the degree of difference w.r.t the categories of invariance between each of the images in a pair.
    For exmaple: transformation invariance captures how different image A is from image B in terms of view-point changes due to camera-viewpoint transformation between the two images.

    We find pairs of images by loading all "anchor to positive" instance pairs via a TripletDatabase for the following two positive types:
            -   (SSD): Same scan, same object, different view
            -   (OSD): Other scan, same object, different view

    TODO: Currently we only support the first type (SSD) because the second one requires to find the extrinsic matrices of target image wrt. the global mesh alignment of source image's scan

    The Invariances are characterized as follows:

    Transformation Invariance: {
        "target_pixels": number of pixels of instance i visible in target image
        "proj_pixels": number of pixels of instance i visible in target image when projected from source image
        "ratio": ratio between first and second
    }

    Illuminance Invariance: {
        "target_illuminance": mean illuminance of all pixels of instance i visible in target image,
        "proj_illuminance": mean illuminance of all pixels of instance i in target image when projected from source image,
        "ratio": difference between first and second
    }
    """

    # We use the depth intrinsics to project from one view to the next. These are the width and height values of the depth images.
    orig_width = 224
    orig_height = 172

    def __init__(self,
                 root_path,
                 triplet_database: Triplet_Database=None,
                 sample_treshold_per_anchor=10,
                 seed=42,
                 render_size=512,
                 debug=False,
                 verbose=False):
        """
        Constructs an Invariance_Database.
        Either the samples are found via a specified Triplet_Database or a default Triplet_Database is constructed with minimum filtering configuration (will try to find as many pairs as possible).

        :param triplet_database: from which to collect all anchor2positive pairs.
        :param sample_treshold_per_anchor: if no triplet_database is provided, this will be passed to the Triplet_Database as the <sample_treshold> argument
        :param render_size: image size of renderings from which we will count the pixels for the two invariance ratios (is expected to be Integer because we only support square image sizes).
        :param debug: if true, will show each rendering result after its calculation via matplotlib
        :param root_path: path/to/2DInstances.txt and path/to/<all_scan_subdirectories>
        :param seed: random seed
        :param verbose: If true, will print tqdm information on all processes.
        """

        # save arguments in object
        self.root_path = root_path
        self.render_size = render_size
        self.sample_treshold_per_anchor = sample_treshold_per_anchor
        self.seed = seed
        self.debug = debug
        self.verbose = verbose

        # create Triplet Database from which the invariances will be calculated
        if triplet_database is not None and isinstance(triplet_database, Triplet_Database):
            self.triplet_database = triplet_database
        else:
            # this default triplet_database is not restrictive in terms of negatives.
            # the negatives are not relevant for calculating the anchor2pos invariances and thus we want to
            # select as many anchor2pos pairs as possible, which will happen if negatives are not restrictive at all
            # when contructing the triplet_database.
            self.triplet_database = Triplet_Database(root_path=root_path,
                                                     seed=seed,
                                                     pos_minimum_visibility=[0.0, 0.0],
                                                     pos_maximum_visibility=[1.0, 1.0],
                                                     neg_background_overlap_minimum=1*1,
                                                     sample_treshold=sample_treshold_per_anchor,
                                                     sample_treshold_per_category=True,
                                                     sample_subset_percentage=0.01,
                                                     sample_fallback=True,
                                                     verbose=verbose)

        # create config of this invariance database's triplet database
        self.triplet_database_config = self.triplet_database.get_config()

        # create empty list of triplet_databases that are contained in this invariance database
        self.triplet_databases_contained = []
        self.loaded_files = []

        # try to load existing invariance databases from disk
        self.pairs_same_scan, self.pairs_different_scan = self.load_invariance_database()

        # loaded pairs might be more than what we need using this self.triplet_database, so filter the pairs
        filtered_pairs_same_scan, filtered_pairs_different_scan, new_pairs_rendered = self.create_invariance_database()

        # update the file to contain the new pairs and also still all existing pairs
        if new_pairs_rendered:
            # save all pairs (also the ones that do not come from this version of the self.triplet_database)
            self.save_invariance_database()

        # now assign the pairs to only the filtered ones accessible from self.triplet_database
        self.pairs_same_scan = filtered_pairs_same_scan
        self.pairs_different_scan = filtered_pairs_different_scan

        self.size = len(self.pairs_same_scan) + len(self.pairs_different_scan)

    def create_invariance_database(self):
        """
        Creates the invariance database like this:
        - load objects.json
        - retrieve all anchor2pos pairs from self.triplet_database
        - create "to pytorch" transformations
        - create Point-Cloud projector and renderer
        - calculate invariances for all anchor2pos pairs
        """
        # load objects.json for lookup of instance-id to hex_color in .ply mesh file
        self.objects = json.load(open(join(self.root_path, "objects.json"), "r"))

        # collect all anchor to positive instances --> we will calculate the invariances for each of these
        if self.verbose:
            print("Collecting all anchor2pos pairs...")
        new_pairs_same_scan, new_pairs_different_scan, reused_pairs_same_scan, reused_pairs_different_scan = self.compute_pairs()

        if self.verbose:
            print(f"Same Scan: Need to calculate {len(new_pairs_same_scan)} pairs and reuse {len(reused_pairs_same_scan)}/{len(self.pairs_same_scan)} pairs")
            print(f"Different Scan: Need to calculate {len(new_pairs_different_scan)} pairs and reuse {len(reused_pairs_different_scan)}/{len(self.pairs_different_scan)} pairs")

        # create the "to pytorch" transformations
        self.transform_images, self.transform_matrices = create_transforms((self.render_size, self.render_size))

        # create Point-Cloud projector and Point-Cloud renderer. Currently only square render_size is supported in renderer.
        self.projector = Projector(W=self.render_size, H=self.render_size)
        self.renderer = RasterizePointsXYsBlending(C=3, size=self.render_size)
        if torch.cuda.is_available():
            self.projector = self.projector.cuda()
            self.renderer = self.renderer.cuda()

        # calculate invariances for all anchor to positives instances which come from the same scan
        # print this always even if verbose=False because this operation takes very long and we want to inform anyways about its duration
        print("Calculating invariances in same scan...")
        self.calculate_invariances(new_pairs_same_scan)

        # calculate invariances for all anchor to positives instances which come from a different scan of the same room
        #print("Calculating invariances in different scans...")
        #self.calculate_invariances(new_pairs_different_scan)
        if self.verbose:
            print("Calculating invariances in different scans is currently skipped. Need to find the extrinsic matrices aligned to anchor scan for this to work (see 3RScan github FAQ)")

        # merge existing (loaded) pairs with new pairs
        self.pairs_same_scan.extend(new_pairs_same_scan)
        self.pairs_different_scan.extend(new_pairs_different_scan)

        # create list of the pairs that are available using self.triplet_database
        # we might have loaded more pairs from file than are accessible here
        # we still want to have all pairs to save all of them to disk again, but we also want to keep a list of pairs
        # only for this instance using self.triplet_database and the pairs available from that triplet_database.
        filtered_pairs_same_scan = [*new_pairs_same_scan, *reused_pairs_same_scan]
        filtered_pairs_different_scan = [*new_pairs_different_scan, *reused_pairs_different_scan]

        if self.verbose:
            print(f"Same Scan: From {len(self.pairs_same_scan)} pairs, {len(filtered_pairs_same_scan)} are for this instance")
            print(f"Different Scan: From {len(self.pairs_different_scan)} pairs, {len(filtered_pairs_different_scan)} are for this instance")

        if self.triplet_database_config not in self.triplet_databases_contained:
            self.triplet_databases_contained.append(self.triplet_database_config)

        return filtered_pairs_same_scan, filtered_pairs_different_scan, len(new_pairs_same_scan) > 0 or len(new_pairs_different_scan) > 0

    def save_invariance_database(self, delete_old_databases=True):
        """
        Saves this database to disk with a unique uuid as filename suffix.

        :return:
        """
        database = {
            "config": self.get_config(),
            "pairs_same_scan": self.pairs_same_scan,
            "pairs_different_scan": self.pairs_different_scan
        }

        id = uuid.uuid4()

        with open(join(self.root_path, f"invariance_database_{id}.json"), "w") as f:
            json.dump(database, f)
            if self.verbose:
                print("Saved invariance_database_ as: " + f.name)

        if delete_old_databases:
            for file in self.loaded_files:
                print(f"Remove old database, it is contained in new one --> {file}")
                os.remove(join(self.root_path, file))

    def load_invariance_database(self):
        """
        Tries to load a precomputed database from disk.

        :return: found_database_on_disk (bool), pairs_same_scan, pairs_different_scan
        """
        result = {
            "pairs_same_scan": [],
            "pairs_different_scan": []
        }
        files = os.listdir(self.root_path)
        if self.verbose:
            print("Searching for existing databases...")
            files = tqdm(files)
        for file in files:
            try:
                if "invariance_database" in file:
                    with open(join(self.root_path, file), "r") as f:
                        invariance_database = json.load(f)
                        file_config = invariance_database["config"]
                        if file_config["root_path"] == self.root_path and file_config["render_size"] == self.render_size:

                            result["pairs_same_scan"].extend(invariance_database["pairs_same_scan"])
                            result["pairs_different_scan"].extend(invariance_database["pairs_different_scan"])

                            # this retrieval is a fallback to previous version of the code where we had "triplet_database" as key in the file_config
                            file_triplet_databases = file_config.get("triplet_databases_contained", [file_config.get("triplet_database", None)])

                            self.triplet_databases_contained.extend(file_triplet_databases)
                            self.loaded_files.append(file)

                            if self.triplet_database_config in self.triplet_databases_contained:
                                if self.verbose:
                                    print(f"Database {file} completely contains the triplet database.")
                                break

                            if self.verbose:
                                print(f"Use existing database: {file}")
                        else:
                            if self.verbose:
                                print(f"Cannot use {file} because the config did not match")
            except:
                pass

        return result["pairs_same_scan"], result["pairs_different_scan"]

    def get_config(self):
        """
        Create the config object for this invariance database. It uniquely identifies this database and it can be assumed
        that identical config objects store identical databases.

        :return: the config
        """
        return {
            "root_path": self.root_path,
            "render_size": self.render_size,
            "triplet_databases_contained": self.triplet_databases_contained
        }

    def calculate_invariances(self, pairs):
        """
        Calculates transformation and illuminance invariance for all pairs like this:
        - Load data for anchor, pos
        - render image from target view given source view and rgb/seg image
        - calculate transformation and illuminance invariance scores
        - store scores in pairs dictionary

        :param pairs:
        :return:
        """
        for pair in tqdm(pairs):
            # load images, depth, extrinsics
            anchor_seg, anchor_rgb, anchor_depth, anchor_extrinsics, anchor_intrinsics = self.load_instance(pair["anchor"])
            pos_seg, pos_rgb, _, pos_extrinsics, pos_intrinsics = self.load_instance(pair["pos"])

            # we need to assume that anchor and pos instance share the same intrinsic matrix, otherwise calculations could be wrong
            # instead of failing the process, we log this as a warning and continue the calculations with the anchor intrinsics, hoping that we are not wrong.
            if not torch.equal(anchor_intrinsics, pos_intrinsics):
                print(f"Intrinsics not equal in scans {pair['anchor'][1]} and {pair['pos'][1]}: {anchor_intrinsics} vs {pos_intrinsics}")

            # render rgb and seg image from pos view given anchor view
            proj_rgb, proj_seg = self.render_from_target(anchor_depth,
                                                         anchor_extrinsics,
                                                         pos_extrinsics,
                                                         anchor_intrinsics,
                                                         anchor_rgb,
                                                         anchor_seg)

            # visualize with matplotlib
            if self.debug:
                show_rendered_images(anchor_rgb, pos_rgb, proj_rgb, proj_seg)

            # calculate transformation invariance with seg image
            target_pixels, proj_pixels, pixel_ratio = calculate_transformation_invariance(pos_seg, proj_seg)

            # calculate illuminance invariance with rgb image
            target_illuminance, proj_illuminance, illuminance_ratio = calculate_illuminance_invariance(pos_rgb, proj_rgb)

            # store scores in pairs dict
            pair["transformation"] = {
                "target_pixels": target_pixels,
                "proj_pixels": proj_pixels,
                "ratio": pixel_ratio
            }

            pair["illuminance"] = {
                "target_illuminance": target_illuminance,
                "proj_illuminance": proj_illuminance,
                "ratio": illuminance_ratio
            }

    def render_from_target(self, source_depth, source_extrinsics, target_extrinsics, intrinsics, source_rgb_colors, source_seg_colors):
        """
        Renders an image from the target view given the source view, depth and extrinsics/intrinsics.

        :param source_depth: depth in source image
        :param source_extrinsics: extrinsics of source view
        :param target_extrinsics: extrinsics of target view
        :param intrinsics: intrinsics of source view (assumed identical to target view)
        :param source_rgb_colors: rgb colors from source image
        :param source_seg_colors: seg colors from source image

        :return: projected_rgb, projected_seg
        """
        projected_output_image, flow, _ = self.projector(source_depth,
                                                         intrinsics,
                                                         intrinsics.inverse(),
                                                         source_extrinsics,
                                                         target_extrinsics.inverse())

        projected_output_image = projected_output_image.permute(0, 2, 1).contiguous()

        rgb_output = self.renderer(projected_output_image.detach().clone(), source_rgb_colors.view(1, 3, -1))
        seg_output = self.renderer(projected_output_image.detach().clone(), source_seg_colors.view(1, 3, -1))

        return rgb_output, seg_output

    def load_instance(self, instance):
        """
        Parses and loads one instance (see self.load_data)

        :param instance: unparsed instance directly retrieved from self.triplet_database
        :return: output of self.load_data
        """
        parsed_instance = self.triplet_database.parse_instance(instance)
        return self.load_data(parsed_instance)

    def load_data(self, instance):
        """
        Loads all image and matrix data for one instance.
        - rgb image: loads the camera image, masked to only show pixels of this instance
        - seg image: loads the rendered segmentation image, masked to only show pixels of this instance
        - depth: loads the rendered depth, unmasked
        - extrinsics: loads the extrinsics for this frame, scaled with *1000.0 to convert T from meters to millimeters
        - intrinsics: loads the intrinsics for this scan, scaled to the self.render_size

        The images and matrices are converted to pytorch via self.transform_images and self.transform_matrices

        :param instance: instance for which to load the data
        :return: seg_image, rgb_image, depth, extrinsics, intrinsics
        """
        base_path = join(self.root_path, instance["scan"])

        # load image as seg
        seg_frame_name = "frame-{:06d}.rendered.labels.png".format(instance["frame_nr"])
        seg_image_path = join(base_path, "rendered", seg_frame_name)
        seg_image = Image.open(seg_image_path).rotate(90, expand=True)

        # load image as rgb
        rgb_frame_name = "frame-{:06d}.color.jpg".format(instance["frame_nr"])
        rgb_image_path = join(base_path, "sequence", rgb_frame_name)
        rgb_image = Image.open(rgb_image_path)

        # mask seg and rgb images
        self.mask_images(seg_image, rgb_image, instance)

        # load rendered depth
        depth_frame_name = "frame-{:06d}.rendered.depth.png".format(instance["frame_nr"])
        depth_path = join(base_path, "rendered", depth_frame_name)
        depth = Image.open(depth_path).rotate(90, expand=True)
        # PIL loads the depth image in int mode, so manually convert to float
        depth = Image.fromarray(np.asarray(depth) * 1.0)

        # load pose
        extrinsics_frame_name = "frame-{:06d}.pose.txt".format(instance["frame_nr"])
        extrinsics = np.loadtxt(join(base_path, "sequence", extrinsics_frame_name)).astype(np.float32)
        extrinsics[:, 3] *= 1000.0 # convert from meter to millimeter because the depth is stored in millimeter and the RT in meter (see 3RScan github FAQ)

        # load intrinsics and its inverse, rescaled to the render_size
        intrinsics = load_intrinsics(base_path, self.render_size)

        # apply transformations
        seg_image = self.transform_images(seg_image)
        rgb_image = self.transform_images(rgb_image)
        depth = self.transform_images(depth)
        extrinsics = self.transform_matrices(extrinsics)
        intrinsics = self.transform_matrices(intrinsics)

        return seg_image, rgb_image, depth, extrinsics, intrinsics

    def mask_images(self, seg_image, rgb_image, instance):
        """
        Masks the segmentation and rgb image according to the instance. Only the pixels of that instance will be visible.

        :param seg_image: segmentation image to mask
        :param rgb_image: rgb image to mask
        :param instance: instance that specifies which pixels shall be masked
        """
        # get color of the instance in the seg image as defined in objects.json for its instance id
        instance_color = self.get_instance_color(instance)

        # mask the seg image by setting everything to black that does not have this color. Save those pixels
        masked_pixels = mask_image_by_color(seg_image, instance_color)

        # set the same pixels to black in the rgb image
        mask_image_by_pixels(rgb_image, masked_pixels)

    def get_instance_color(self, instance):
        """
        Looks up the color of the instance given its instance_id and retrieving the ply_color attribute in objects.json.

        :param instance: parsed instance for which to look up the color
        :return: color as (r, g, b) tuple
        """
        instance_id = instance["instance_id"]
        scan_id = instance["scan"]

        for scan in self.objects["scans"]:
            if scan["scan"] == scan_id:
                for obj in scan["objects"]:
                    if obj["id"] == str(instance_id):
                        hex = obj["ply_color"].lstrip('#')
                        color = tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))
                        return color

        raise ValueError("No color found for instance", instance)

    def compute_pairs(self):
        """
        Computes all possible anchor to positive pairs that are available in self.triplet_database.
        It calculates all pairs for the same scan and for different scans of the same room.
        This corresponds to the positive types "SSD" and "OSD" of self.triplet_database.

        :return: pairs_same_scan, pairs_different_scan --> dictionaries where the key is (anchor, pos) and the entry is the values of it.
        """

        # retrieve instances from database
        triplets_possibilities, instances = self.triplet_database.get()

        # create empty pairs
        new_pairs_same_scan = {}
        new_pairs_different_scan = {}

        # look up positives for each anchor in instances
        if self.verbose:
            instances = tqdm(instances)

        for anchor in instances:
            # find all positives from the same scan (type SSD)
            same_scan_pos = triplets_possibilities[str(anchor)]["pos"][0]
            add_positives(anchor, same_scan_pos, new_pairs_same_scan)

            # find all positives from different scan in same room (type OSD)
            different_scan_pos = triplets_possibilities[str(anchor)]["pos"][1]
            add_positives(anchor, different_scan_pos, new_pairs_different_scan)

            new_pairs_different_scan = {} # todo we currently do not support this (see class documentation)

        # filter out those that already exist
        new_pairs_same_scan, reused_pairs_same_scan = filter_pairs(self.pairs_same_scan, new_pairs_same_scan)
        new_pairs_different_scan, reused_pairs_different_scan = filter_pairs(self.pairs_different_scan, new_pairs_different_scan)

        # convert the keys to strings for serialization to work later
        # now it is no longer important to have a set as key because we went through all pairs already.
        new_pairs_same_scan = convert_to_list(new_pairs_same_scan)
        new_pairs_different_scan = convert_to_list(new_pairs_different_scan)

        return new_pairs_same_scan, new_pairs_different_scan, reused_pairs_same_scan, reused_pairs_different_scan


def filter_pairs(existing_pairs, new_pairs):
    reused_pairs = []

    for idx, p in enumerate(existing_pairs):
        key = frozenset({str(p["anchor"]), str(p["pos"])})
        if key in new_pairs:
            del new_pairs[key]
            reused_pairs.append(p)

    return new_pairs, reused_pairs


def convert_to_list(pairs):
    """
    Converts the pairs dictionary to list by discarding the keys.
    The keys are constructed as frozenset (see add_positives) but are no longer needed once the whole pairs are constructed.
    The keys were just needed to ensure that we do not add pairs more than once, but after that they serve no purpose and are
    redundant information.

    :param pairs: pairs dictionary for which the keys should be removed
    :return: list of all values in the order of their appearance when calling pairs.items()
    """
    return [v for k,v in pairs.items()]


def calculate_transformation_invariance(target_seg_image, projected_seg_image):
    """
    Calculates the transformation invariance scores between target and projected seg images.
    We calculate all non-black pixels and compare them.

    :param target_seg_image:
    :param projected_seg_image:
    :return: target_pixels, proj_pixels, pixel_ratio
    """
    gt_pixels = (target_seg_image != 0).sum() // 3
    proj_pixels = (projected_seg_image != 0).sum() // 3
    if proj_pixels < gt_pixels:
        identical_pixels = 1.0 * proj_pixels / gt_pixels
    else:
        identical_pixels = 1.0 * gt_pixels / proj_pixels

    return gt_pixels.cpu().numpy().item(),\
           proj_pixels.cpu().numpy().item(),\
           identical_pixels.cpu().numpy().item()


def calculate_illuminance_invariance(target_rgb_image, projected_rgb_image):
    """
    Calculates the illuminance invariance scores between target and projected rgb images.
    We retrieve the mean illuminance of all non-black pixels and compare them

    :param target_rgb_image:
    :param projected_rgb_image:
    :return: target_illuminance, proj_illuminance, illuminance_difference
    """
    gt_mean_illuminance = target_rgb_image[target_rgb_image != 0].mean()
    proj_mean_illuminance = projected_rgb_image[projected_rgb_image != 0].mean()
    illuminance_difference = torch.abs(gt_mean_illuminance - proj_mean_illuminance)

    return gt_mean_illuminance.cpu().numpy().item(),\
           proj_mean_illuminance.cpu().numpy().item(),\
           illuminance_difference.cpu().numpy().item()


def show_rendered_images(source_rgb, target_rgb, proj_rgb, proj_seg):
    """
    Debugging call to show the source, target and projected images.

    :param source_rgb:
    :param target_rgb:
    :param proj_rgb:
    :param proj_seg:
    :return:
    """
    input_image = source_rgb.squeeze().permute(1, 2, 0).cpu().numpy()
    seg_output_plt = proj_seg.squeeze().permute(1, 2, 0).cpu().numpy()
    rgb_output_plt = proj_rgb.squeeze().permute(1, 2, 0).cpu().numpy()
    output_image = target_rgb.squeeze().permute(1, 2, 0).cpu().numpy()

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(224, 224))
    fig.add_subplot(1, 4, 1)
    plt.imshow(input_image)
    fig.add_subplot(1, 4, 2)
    plt.imshow(seg_output_plt)
    fig.add_subplot(1, 4, 3)
    plt.imshow(rgb_output_plt)
    fig.add_subplot(1, 4, 4)
    plt.imshow(output_image)
    plt.show()


def mask_image_by_color(image, color):
    """
    Mask the input image according to the rgb color.
    ALl pixels that do not have this color will be set to black.

    :param image: the PIL image to mask
    :param color: the RGB color as (r, g, b) tuple
    """

    pixels = image.load()  # create the pixel map

    masked_pixels = []

    for w in range(image.size[0]):  # for every col:
        for h in range(image.size[1]):  # For every row
            if pixels[w, h] != color:
                pixels[w, h] = (0, 0, 0)  # set the colour accordingly
                masked_pixels.append((w,h))

    return masked_pixels


def mask_image_by_pixels(image, masked_pixels):
    """
    Mask the input image according to the list of (w, h) pixels in masked_pixels.
    All pixels that are present in the list will be set to black.

    :param image: the PIL image to mask
    :param masked_pixels: list of (w, h) tuples specifying all pixels that shall be masked.
    """
    pixels = image.load()  # create the pixel map
    for w, h in masked_pixels:
        pixels[w, h] = (0, 0, 0)  # set the colour accordingly


def add_positives(anchor, positives, pairs):
    """
    Creates one "anchor to positive" entry for each positive in the list of positives.
    Adds all these entries to the given list of pairs by constructing a unique key between anchor and pos.
    If anchor and pos are added later in reverse order, that key will be similar and thus, not duplicate entries will be added.

    :param anchor: the anchor from which to create entries
    :param positives: list of positives. for each positive we create one entry
    :param pairs: the list of pairs where to add all the entries
    """
    for pos in positives:
        key = frozenset({str(anchor), str(pos)})
        entry = {
            "anchor": anchor,
            "pos": pos
        }

        pairs[key] = entry


def load_intrinsics(scan_path, render_size):
    """
    Loads the intrinsic camera parameters from the file "_info.txt" in the given path and rescales it using the given render_size.

    :param render_size: new square size for the images. Intrinsic parameters need to be rescaled accordingly.
    :param scan_path: path to a scan folder
    :return:
    """
    intrinsics_path = join(scan_path, "sequence", "_info.txt")
    intrinsics_file = csv.reader(open(intrinsics_path), delimiter=" ")
    intrinsics = np.eye(4, dtype=np.float32)

    for line in intrinsics_file:
        if line[0] == "m_calibrationDepthIntrinsic":
            intrinsics_dict = {
                "fx": line[2],
                "fy": line[7],
                "cx": line[4],
                "cy": line[8],
            }
            intrinsics[0, 0] = float(intrinsics_dict["fx"])
            intrinsics[1, 1] = intrinsics_dict["fy"]
            intrinsics[0, 2] = float(intrinsics_dict["cx"])
            intrinsics[1, 2] = intrinsics_dict["cy"]
            break

    # when we change the size of the images, the intrinsics need to change as well. They are resized accordingly.
    intrinsics[0, 0] *= 1.0 * render_size / Invariance_Database.orig_width
    intrinsics[0, 2] *= 1.0 * render_size / Invariance_Database.orig_width
    intrinsics[1, 1] *= 1.0 * render_size / Invariance_Database.orig_height
    intrinsics[1, 2] *= 1.0 * render_size / Invariance_Database.orig_height

    return intrinsics


def create_transforms(shape):
    """
    Creates the "to pytorch" transformations for images and extrinsic/intrinsic camera matrices.
    The images are resized to given shape.
    Both images and matrices get appended a batch-dimension and are moved to cuda if available.

    :param shape: the resize shape for the images

    :return: transform_images, transform_matrices --> torchvision.transform objects that can transform the inputs.
    """
    to_cuda = Lambda(lambda x: x.cuda() if torch.cuda.is_available() else x)

    transform_images = Compose([
        Resize(shape),
        ToTensor(),
        Lambda(lambda x: x.unsqueeze(0)),
        to_cuda
    ])

    transform_matrices = Compose([
        ToTensor(),
        to_cuda
    ])

    return transform_images, transform_matrices


def test():
    from data.invariance_dataset import Invariance_Dataset
    invariance_database = Invariance_Database(root_path="/home/lukas/datasets/3RScan/3RScan-10/all_in_one",
                                              sample_treshold_per_anchor=10,
                                              debug=False,
                                              render_size=64,
                                              verbose=True)

    transform = torchvision.transforms.Compose([
        Triplet_Dataset.rotate_vertical_transform,
        torchvision.transforms.Resize(256),
        torchvision.transforms.ToTensor()
    ])

    invariance_dataset = Invariance_Dataset(invariance_database,
                                            bbox_data_aligned_vertically=False,
                                            transform=transform)

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
