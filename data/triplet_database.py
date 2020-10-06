import numpy as np
import json
import csv

from os.path import join
import os

from tqdm.auto import tqdm

import random

import uuid

class Triplet_Database():
    """
    The Triplet Database looks for the file 2DInstances.txt in the provided <root_path> and samples triplets (anchor, positive, negative) from it.
    For each instance (=row) in the 2DInstances.txt file, it selects possible positive and negative matches of the following categories.
    Up to <sample_treshold> matches are searched for each instance that allow for random sampling of concrete triplets from a list of available matches.
    All triplets are saved into a file database <triplet_possibilities_<uuid>.json>.

    We sample 4 types of positives (explanations are positive sample w.r.t anchor sample):
            -   (SSD): Same scan, same object, different view
            -   (OSD): Other scan, same object, different view
            -   (COA): Same or other scan, same class, other object, ambiguity between anchor and other
            -   (OSR): Other scan, same object, rigid movement happened

    We sample 5 types of negatives (explanations are negative sample w.r.t anchor sample):
            -   (OAC): Other room, any class
            -   (SAC): Same room, any class
            -   (OSC): Other room, same class
            -   (SCA): Same room, other scan, same class, other instance, no ambiguity to anchor instance, rigid movement happened
            -   (AVB): Same room, same scan, anchor no longer visible, but something in the background of anchor still is

    Each instance of a triplet has the following structure:
    {
        "reference": ref_scan_id,
        "scan": scan_id,
        "frame_nr": frame_id,
        "instance_id": instance_id,
        "label": label,
        "bbox": bbox,
        "visibility": visibility,
        "number_other_instances": number_other_instances,
        "other_instance_ids": other_instance_ids,
        "other_bboxes": other_bboxes
    }

    See also: <REPO_ROOT>/util/FrameFilter for information about the file 2DInstances.txt
    """

    # TODO: change pos_minimum_visibility to "minimum_visibility" and apply it also to anchor filtering!
    default_pos_minimum_visibility = [0.0, 0.0]
    default_pos_maximum_visibility = [1.0, 1.0]

    def __init__(self,
                 root_path,
                 seed=42,
                 pos_minimum_visibility=default_pos_minimum_visibility,
                 pos_maximum_visibility=default_pos_maximum_visibility,
                 neg_background_overlap_minimum=128*128,
                 sample_treshold=-1,
                 sample_treshold_per_category=True,
                 sample_subset_percentage=1.0,
                 sample_fallback=False,
                 verbose=False):
        """
        Creates a Triplet_Database. It will look for already saved triplet_possibilities_<uuid>.json files in the root_path.
        If any of those files matches the concrete configuration of this instance, it will load that file instead of going through the
        2DInstances.txt file again. If the configuration is not matched, a new database will be created.

        :param root_path: path/to/2DInstances.txt and path/to/<all_scan_subdirectories>
        :param seed: random seed
        :param pos_minimum_visibility: [<min_truncation_score>, <min_occlusion_score>] for all instances to be valid for a triplet (scores as defined in the 2DInstances.txt file)
        :param pos_maximum_visibility: [<max_truncation_score>, <max_occlusion_score>] for all instances to be valid for a triplet (scores as defined in the 2DInstances.txt file)
        :param neg_background_overlap_minimum: For the negative sample type "AVB": how much bbox overlap in the anchor image does the negative instance need to have (in pixels).
        :param sample_treshold: after n positive and n negative samples are found for one anchor, we no longer iterate over the rest of available samples.
               Instead we randomly sample from the samples that we have found thus far. Default: -1, meaning we will always go through all available samples. Any value <1 will have that effect.
        :param sample_treshold_per_category: if <sample_treshold> shall be counted per category or over all categories.
        :param sample_subset_percentage: how many percentage of all samples should be searched for a pos/neg sample at random. Default: 1.0. Must be 0<=p<=1
        :param sample_fallback: if pos/neg could not be found in the sample_subset defined by sample_subset_percentage, then search in all samples as fallback. Default: False
        :param verbose: If true, will tqdm the remaining time for all actions.
        """

        # save all constructor arguments
        self.pos_minimum_visibility = pos_minimum_visibility if pos_minimum_visibility is not None else Triplet_Database.default_pos_minimum_visibility
        self.pos_maximum_visibility = pos_maximum_visibility if pos_maximum_visibility is not None else Triplet_Database.default_pos_maximum_visibility
        self.neg_background_overlap_minimum = neg_background_overlap_minimum
        self.root_path = root_path
        self.seed = seed
        self.verbose = verbose
        self.sample_treshold = sample_treshold
        self.sample_subset_percentage = sample_subset_percentage
        self.sample_fallback = sample_fallback
        self.sample_treshold_per_category = sample_treshold_per_category

        # random seed
        np.random.seed(seed)

        # precompute triplets
        self.precompute_triplets()

    def get(self):
        return self.triplets_possibilities, self.instances

    def precompute_triplets(self):
        # initialize empty 3RScan.json cache for all scans that are accessed during pre-computation of triplets
        self.scans = {}

        # initialize empty cache for all instances that are parsed into their columns during pre-computation of triplets
        self.parsed_instances = {}

        # load all instances defined in 2DInstances.txt
        with open(join(self.root_path, "2DInstances.txt")) as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            self.instances = [i for i in reader]

        # create config
        self.config = self.get_config()

        # initialize triplets cache for saving to disk. For each anchor contains a list of all possible positives and negatives.
        self.triplets_possibilities = self.load_triplet_possibilities()

        # TODO Multiprocess this?
        #pool = multiprocessing.Pool(12)
        #triplets = pool.map(self.find_triplet, self.instances)

        # precompute triplets possibilities
        if not self.triplets_possibilities:
            # load 3RScan.json file: is the central metadata file
            self.metadata = json.load(open(join(self.root_path, "3RScan.json")))

            # precompute the triplets at construction time. At __getitem__ we only need to load image data from disk.
            n = range(len(self.instances))
            if self.verbose:
                n = tqdm(n)
                print("Start precomputing triplets...")

            # start precomputation
            for idx in n:
                anchor = self.instances[idx]
                if self.sample_subset_percentage < 1:
                    # sample k random instances
                    instances = random.sample(self.instances, k=int(self.sample_subset_percentage*len(self.instances)))

                    # add all instances from same room
                    self.add_all_instances_of_same_scan(anchor, idx, instances)

                    # could have selected an instance of same room in k random instances before --> only keep distinct values
                    instances = set(tuple(instance) for instance in instances)
                else:
                    instances = self.instances
                all_possible_positives, all_possible_negatives, triplet_found = self.find_triplet(anchor, instances)

                if self.sample_fallback and not triplet_found and self.sample_subset_percentage < 1:
                    all_possible_positives, all_possible_negatives, _ = self.find_triplet(anchor, self.instances)

                self.triplets_possibilities[str(anchor)] = {
                    "pos": all_possible_positives,
                    "neg": all_possible_negatives
                }

            # save calculated triplets to disk
            self.save_triplet_possibilities()

    def add_all_instances_of_same_scan(self, instance, idx, instances_list):

        # ref scan id of the instance for which adding should be performed
        ref_scan_id = self.parse_instance(instance)["reference"]

        # search backwards from instance
        for i in range(idx-1, -1, -1):
            scan_id = self.parse_instance(self.instances[i])["reference"]
            if scan_id == ref_scan_id:
                instances_list.append(self.instances[i])
            else:
                break # since 2DInstances.txt is ordered alphabetically w.r.t. scan id's we know that now no other scan will have the same scan_id again

        # search forwards from instance
        for i in range(idx+1, len(self.instances), 1):
            scan_id = self.parse_instance(self.instances[i])["reference"]
            if scan_id == ref_scan_id:
                instances_list.append(self.instances[i])
            else:
                break # since 2DInstances.txt is ordered alphabetically w.r.t. scan id's we know that now no other scan will have the same scan_id again

    def save_triplet_possibilities(self):
        self.triplets_possibilities["config"] = self.config

        id = uuid.uuid4()

        with open(join(self.root_path, f"triplets_possibilities_{id}.json"), "w") as f:
            json.dump(self.triplets_possibilities, f)
            if self.verbose:
                print("Saved triplets_possibilities as: " + f.name)

    def get_config(self):
        config = {
            "seed": self.seed,
            "sample_treshold": self.sample_treshold,
            "sample_treshold_per_category": self.sample_treshold_per_category,
            "sample_subset_percentage": self.sample_subset_percentage,
            "sample_fallback": self.sample_fallback,
            "neg_background_overlap_minimum": self.neg_background_overlap_minimum,
            "number_instances": len(self.instances),
            "root_path": self.root_path
        }

        if self.pos_maximum_visibility != Triplet_Database.default_pos_maximum_visibility:
            if self.verbose:
                print(f"Config will include pos_maximum_visibility of"
                      f" {self.pos_maximum_visibility} because it is different from the default "
                      f"{Triplet_Database.default_pos_maximum_visibility}")
            config["pos_maximum_visibility"] = self.pos_maximum_visibility

        if self.pos_minimum_visibility != Triplet_Database.default_pos_minimum_visibility:
            if self.verbose:
                print(f"Config will include pos_minimum_visibility of"
                      f" {self.pos_minimum_visibility} because it is different from the default "
                      f"{Triplet_Database.default_pos_minimum_visibility}")
            config["pos_minimum_visibility"] = self.pos_minimum_visibility

        return config

    def load_triplet_possibilities(self):
        result = {}
        try:
            files = os.listdir(self.root_path)
            if self.verbose:
                print("Searching for existing databases...")
                files = tqdm(files)
            for file in files:
                if "triplets_possibilities" in file:
                    with open(join(self.root_path, file), "r") as f:
                        triplets_possibilities = json.load(f)
                        file_config = triplets_possibilities["config"]
                        if file_config == self.config:
                            result = triplets_possibilities
                            if self.verbose:
                                print(f"Use existing database: {file}")
                            break
                        else:
                            if self.verbose:
                                print(f"Cannot use {file} because the config did not match")
        except:
            pass
        return result

    def get_scan(self, scan):
        """
        Return the json object for the scan as defined in 3RScan.json.
        If it is a reference scan, it will return the whole reference json object with all rescans as sub-object.
        If it is a rescan scan, it will also return the whole reference json object with this rescan being a sub-object.

        :param scan: scan_id of the scan to load

        :return: json object or ValueError if scan cannot be found in 3RScan.json
        """

        # try to find in scans cache if accessed before
        result = self.scans.get(scan, None)
        if result is not None:
            return result

        # search for scan in 3RScan.json
        for reference in self.metadata:

            # search in reference
            if scan == reference["reference"]:
                result = reference
                break # early stopping in reference

            # search in rescans
            found = False # early stopping in rescans
            for rescan in reference["scans"]:
                if scan == rescan["reference"]:
                    result = reference
                    found = True
                    break # early stopping in rescans
            if found:
                break # early stopping in rescans

        # store in cache or throw error
        if result is None:
            raise ValueError("Scan not found: ", scan)
        else:
            self.scans[scan] = result

        return result

    def parse_instance(self, instance):
        """
        Parse a row of 2DInstances.txt into its columns as defined in 2DInstances.txt

        :param instance: a row of 2DInstances.txt (an entry in self.instances)
        :return: dict with format
        {
            "reference": ref_scan_id,
            "scan": scan_id,
            "frame_nr": frame_id,
            "instance_id": instance_id,
            "label": label,
            "bbox": [x1, y1, x2, y2],
            "visibility": [truncation_number_pixels_original_image, truncation_number_pixels_larger_fov_image, truncation, occlusion_number_pixels_original_image, occlusion_number_pixels_only_with_that_instance, occlusion],
            "number_other_instances": number_other_instances,
            "other_instance_ids": other_instance_ids,
            "other_bboxes": other_bboxes (list of [x1, y1, x2, y2] where i-th bbox belongs to other_instance_id[i])
        }
        """
        # try to find in instances cache if accessed before
        result = self.parsed_instances.get(str(instance), None)
        if result is not None:
            return result

        ref_scan_id = instance[0]
        scan_id = instance[1]
        frame_id = int(instance[2])
        instance_id = int(instance[3])
        label = instance[4]
        bbox = [int(x) for x in instance[5:9]]
        visibility = [float(x) for x in instance[9:15]]
        number_other_instances = int(instance[15])
        other_instance_ids = []
        other_bboxes = []
        if number_other_instances > 0:
            for i in range(number_other_instances):
                start = 16 + i * 5  # we have 5 values in the file for each other instance: <instance_id> <bbox>
                other_instance_ids.append(int(instance[start]))  # every 5-th value is a new instance id
                other_bboxes.append([int(x) for x in instance[start + 1:start + 5]])  # next 4 values are the bbox

        result =  {
            "reference": ref_scan_id,
            "scan": scan_id,
            "frame_nr": frame_id,
            "instance_id": instance_id,
            "label": label,
            "bbox": bbox,
            "visibility": visibility,
            "number_other_instances": number_other_instances,
            "other_instance_ids": other_instance_ids,
            "other_bboxes": other_bboxes
        }

        # store in cache
        self.parsed_instances[str(instance)] = result

        return result

    def is_same_room(self, first_scan_id, second_scan_id):
        """
        Return if the two scan_ids reference the same room.

        :param first_scan_id:
        :param second_scan_id:
        :return: true or false
        """

        ref1_id = self.get_scan(first_scan_id)["reference"]
        ref2_id = self.get_scan(second_scan_id)["reference"]
        return ref1_id == ref2_id

    def has_ambiguity(self, first_scan, first_instance, second_scan, second_instance):
        """
        Check if the two instances share an ambiguity that is defined in 3RScan.json

        :param first: a row of 2DInstances.txt (an entry in self.instances)
        :param second: a row of 2DInstances.txt (an entry in self.instances)
        :return: true or false
        """

        # if not same room: immediately return false: is a sanity check
        if not self.is_same_room(first_scan, second_scan):
            return False

        # find reference
        ref = self.get_scan(first_scan)

        # check if ambiguity present for this reference id and if instance numbers are defined in it
        for ambiguity in ref["ambiguity"]:
            for entry in ambiguity:
                source = int(entry["instance_source"])
                target = int(entry["instance_target"])
                if (first_instance == source and second_instance == target) or (
                        second_instance == source and first_instance == target):
                    return True
        return False

    def has_rigid_movement(self, first_scan_id, second_scan_id, instance_id):
        """
        Check if the two scan_ids have a different rigid movement w.r.t the specified instance.
        Either one of them has no rigid movement and the other one does (i.e. one is a reference scan),
        or both have rigid movements (i.e. both are rescans).

        :param first_scan_id:
        :param second_scan_id:
        :param instance_id: for which instance to check the rigid movement defined
        :return: true or false
        """

        # if not same room: immediately return false: is a sanity check
        if not self.is_same_room(first_scan_id, second_scan_id):
            return False

        # there can be no movement when it is the same scan: is a sanity check
        if first_scan_id == second_scan_id:
            return False

        # find reference
        ref = self.get_scan(first_scan_id)

        # retrieve rigid movement for instance for each scan
        rigid_first = None
        rigid_second = None
        for rescan in ref["scans"]:
            if rescan["reference"] == first_scan_id:
                for rigid in rescan["rigid"]:
                    if rigid["instance_reference"] == instance_id or rigid["instance_rescan"] == instance_id:
                        rigid_first = rigid
            if rescan["reference"] == second_scan_id:
                for rigid in rescan["rigid"]:
                    if rigid["instance_reference"] == instance_id or rigid["instance_rescan"] == instance_id:
                        rigid_second = rigid

        # evaluate rigid movements to judge whether there is rigid movement for this instance
        if rigid_first is None and rigid_second is None:
            return False

        if rigid_first is not None:
            if rigid_second is None:
                return True
            else:
                return rigid_first["transform"] != rigid_second["transform"]

        if rigid_second is not None:
            if rigid_first is None:
                return True
            else:
                return rigid_first["transform"] != rigid_second["transform"]

    def contains_instance(self, instance, query_id):
        """
        Return if an object with given instance_id from the "other_instances" in a frame represented by a parsed instance
        (see self.parse_instance) is overlapping with the parsed instance (w.r.t. their bboxes).

        :param instance: instance in the format defined in self.parse_instance
        :param query_id: instance_id for which to compare to the bbox. If this instance_id is not even part of the "other_instances" of this parsed instance, return False as well.
        :return: (True, area_of_overlap) if overlapping, (False, 0) otherwise.
        """
        idx_in_other_instances = [idx for idx, id in enumerate(instance["other_instance_ids"]) if id == query_id]
        # if that instance id is contained in the frame represented by this parsed instance
        if len(idx_in_other_instances) > 0:
            other_bbox = instance["other_bboxes"][idx_in_other_instances[0]]
            return self.bboxes_overlapping(other_bbox, instance["bbox"])
        return False, 0

    def bboxes_overlapping(self, first_bbox, second_bbox):
        """
        Return if two axis-aligned-bboxes in format [x1, y1, x2, y2] are overlapping where
        (x1, y1): upper left corner
        (x2, y2): lower right corner
        (0, 0): origin at upper left

        :param first_bbox: first bbox to check
        :param second_bbox: second bbox to check
        :return: (True, area_of_overlap) if overlapping, (False, 0) otherwise.
        """
        first_min_x = first_bbox[0]
        first_min_y = first_bbox[1]
        first_max_x = first_bbox[2]
        first_max_y = first_bbox[3]

        second_min_x = second_bbox[0]
        second_min_y = second_bbox[1]
        second_max_x = second_bbox[2]
        second_max_y = second_bbox[3]

        dx = min(first_max_x, second_max_x) - max(first_min_x, second_min_x)
        dy = min(first_max_y, second_max_y) - max(first_min_y, second_min_y)

        if (dx >= 0) and (dy >= 0):
            return True, dx * dy
        else:
            return False, 0

    def visibility_in_range(self, visibility):
        """
        Check if a given visibility satisfies the range of [self.pos_minimum_visibility, self.pos_maximum_visibility].

        :param visibility: in the format as loaded from parse_instance
        :return: True or False
        """
        truncation = visibility[2]
        occlusion = visibility[5]

        return truncation >= self.pos_minimum_visibility[0] \
           and truncation <= self.pos_maximum_visibility[0] \
           and occlusion >= self.pos_minimum_visibility[1] \
           and occlusion <= self.pos_maximum_visibility[1]

    def sample_treshold_reached(self, list_of_lists):
        if self.sample_treshold < 1:
            return False

        if self.sample_treshold_per_category:
            return all(len(l) >= self.sample_treshold for l in list_of_lists)

        else:
            # count all samples in list_of_lists. The i-th list only counts when probs[i] > 0, i.e. it would also be selected later.
            samples = sum([len(l) for idx, l in enumerate(list_of_lists)])

            return samples >= self.sample_treshold

    def add_instance(self, instance, list, instances_counter):
        if not self.sample_treshold_reached([list]):
            list.append(instance)
            instances_counter += 1

        return instances_counter

    def find_positive_pair(self, anchor, instances):
        """
        Sample 4 types of positives:
            -   (SSD): Same scan, same object, different view
            -   (OSD): Other scan, same object, different view
            -   (COA): Same or other scan, same class, other object, ambiguity between anchor and other
            -   (OSR): Other scan, same object, rigid movement happened

        Then choose with a probability between the 4 types and sample from chosen type at random.

        :param anchor: a row of 2DInstances.txt (an entry in self.instances)
        :return: a row of 2DInstances.txt (an entry in self.instances) which is a positive match for anchor
        """
        # create lists as defined in documentation
        ssd = []
        osd = []
        coa = []
        osr = []

        # retrieve anchor attributes
        a = self.parse_instance(anchor)

        # save overall number of found instances
        instances_counter = 0

        # go through all instances and select the ones that match any of the 4 types of positives
        for instance in instances:

            # have we collected enough samples to randomly sample from?
            if self.sample_treshold_reached([ssd, osd, coa, osr]):
                break

            # retrieve instance attributes
            i = self.parse_instance(instance)

            # if this instance is not that much visible, we do not want to have it as a positive sample at all (no matter the case).
            if not self.visibility_in_range(i["visibility"]):
                continue

            # do some common checks helpful for all type checks
            same_room = self.is_same_room(a["scan"], i["scan"])
            same_label = i["label"] == a["label"]
            same_scan = i["scan"] == a["scan"]
            same_instance = i["instance_id"] == a["instance_id"]
            same_frame = i["frame_nr"] == a["frame_nr"]
            ambiguous = self.has_ambiguity(a["scan"], a["instance_id"], i["scan"], i["instance_id"])
            rigid_movement = self.has_rigid_movement(a["scan"], i["scan"], i["instance_id"])

            # do type checks
            if same_scan:
                # check SSD
                if same_instance and not same_frame:
                    # matches SSD
                    instances_counter = self.add_instance(instance, ssd, instances_counter)

                # check COA
                if same_label and not same_instance and ambiguous:
                    # matches COA
                    instances_counter = self.add_instance(instance, coa, instances_counter)

            elif same_room:
                # check OSD & OSR
                if same_instance:
                    # matches OSD
                    instances_counter = self.add_instance(instance, osd, instances_counter)
                    if rigid_movement:
                        # matches OSR
                        instances_counter = self.add_instance(instance, osr, instances_counter)

                # check COA
                if same_label and not same_instance and ambiguous:
                    # matches COA
                    instances_counter = self.add_instance(instance, coa, instances_counter)
            # else: matches no criteria, so ignore it

        # create positives list and sample probabilities
        positives = [ssd, osd, coa, osr]

        return positives, instances_counter

    def find_negative_pairs(self, anchor, instances):
        """
        Sample 5 types of negatives:
            -   (OAC): Other room, any other class
            -   (SAC): Same room, any other class
            -   (OSC): Other room, same class
            -   (SCA): Same room, other scan, same class, other instance, no ambiguity to anchor instance, rigid movement happened (due to bad ambiguity-annotation-rate in dataset for non-moved objects)
            -   (AVB): Same room, same scan, anchor no longer visible, but something in the background of anchor still is, that something is not ambiguous to anchor

        Then choose with a probability between the 5 types and sample from chosen type at random.

        :param anchor: a row of 2DInstances.txt (an entry in self.instances)
        :return: list of rows of 2DInstances.txt (an entry in self.instances) that are negative matches for anchor.
        """
        # create lists as defined in documentation
        oac = []
        sac = []
        osc = []
        sca = []
        avb = []

        # retrieve anchor attributes
        a = self.parse_instance(anchor)

        # save overall number of found instances
        instances_counter = 0

        # go through all instances and select the ones that match any of the 5 types of negatives
        for instance in instances:

            # have we collected enough samples to randomly sample from?
            if self.sample_treshold_reached([oac, sac, osc, sca, avb]):
                break

            # retrieve instance attributes
            i = self.parse_instance(instance)

            # do some common checks helpful for all type checks
            same_room = self.is_same_room(a["scan"], i["scan"])
            same_label = i["label"] == a["label"]
            same_scan = i["scan"] == a["scan"]
            same_instance = i["instance_id"] == a["instance_id"]
            anchor_in_instance, _ = self.contains_instance(i, a["instance_id"])
            instance_in_anchor, instance_in_anchor_overlap_area = self.contains_instance(a, i["instance_id"])
            instances_overlapping = anchor_in_instance or instance_in_anchor
            ambiguous = self.has_ambiguity(a["scan"], a["instance_id"], i["scan"], i["instance_id"])
            rigid_movement = self.has_rigid_movement(a["scan"], i["scan"], i["instance_id"])

            # do type checks
            if same_room:
                # check SAC
                if not same_label and not instances_overlapping:
                    # matches SAC
                    instances_counter = self.add_instance(instance, sac, instances_counter)

                # check SCA
                if not same_scan and same_label and not same_instance and not instances_overlapping and not ambiguous and rigid_movement:
                    # matches SCA
                    instances_counter = self.add_instance(instance, sca, instances_counter)

                # check AVB
                if same_scan and not anchor_in_instance and instance_in_anchor and instance_in_anchor_overlap_area > self.neg_background_overlap_minimum and not ambiguous:
                    # matches AVB
                    instances_counter = self.add_instance(instance, avb, instances_counter)
            else:
                # check OAC & OSC
                if not same_label:
                    # matches OAC
                    instances_counter = self.add_instance(instance, oac, instances_counter)
                else:
                    # matches OSC
                    instances_counter = self.add_instance(instance, osc, instances_counter)
            # else: matches no criteria, so ignore it

        # create negatives list
        negatives = [oac, sac, osc, sca, avb]

        return negatives, instances_counter

    def find_triplet(self, anchor_instance, instances):
        """
        Sample positive and negative for this anchor instance.
        If no positive or negative could be found, we signal this via the first object in tuple, which is then false.

        :param anchor_instance: a row of 2DInstances.txt (an entry in self.instances)
        :return: (valid, pos, neg) tuple
        """

        all_possible_positives, pos_counter = self.find_positive_pair(anchor_instance, instances)

        all_possible_negatives, neg_counter = self.find_negative_pairs(anchor_instance, instances)

        return all_possible_positives, all_possible_negatives, pos_counter > 0 and neg_counter > 0

def test():
    database = Triplet_Database(root_path="/home/lukas/datasets/3RScan/3RScan-10/all_in_one",
                                pos_minimum_visibility=[0.7, 0.7],
                                pos_maximum_visibility=[0.9, 0.9],
                                neg_background_overlap_minimum=111*128,
                                sample_treshold=1,
                                sample_treshold_per_category=True,
                                sample_subset_percentage=0.9,
                                sample_fallback=True,
                                verbose=True)

    triplet_possibilities, instances = database.get()

    print(triplet_possibilities["config"])

if __name__ == "__main__":
    # execute only if run as a script
    test()
