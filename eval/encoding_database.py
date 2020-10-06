import json
from tqdm.auto import tqdm

import torch
import uuid

from os.path import join
import os
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Dataset

from data.data_util import triplets_as_batches, outputs_as_triplets

from data.abstract_dataset import Abstract_Dataset

def convert_sample_dict(sample):
    sample_meta = {
        **sample
    }
    sample_meta.pop('image', None)
    sample_meta.pop('bbox', None)

    for k, v in sample_meta.items():
        if isinstance(v, list):
            sample_meta[k] = v[0]
        if isinstance(v, torch.Tensor):
            sample_meta[k] = v.detach().cpu().numpy().item()
    return sample_meta


class Encoding_Dataset(Dataset):
    """
    Loads triplet-encodings from a Encoding_Database and returns them in the same idx-based order as the Abstract_Dataset would.
    """
    def __init__(self, encodings, dataset: Abstract_Dataset):
        self.encodings = encodings
        self.dataset = dataset
        self.size = self.dataset.__len__()

    def __getitem__(self, idx):
        anchor, pos, negs = get_instance_metadata(self.dataset[idx])

        result = {
            "anchor": self.encodings[str(anchor)],
            "pos": self.encodings[str(pos)]
        }
        if negs is not None:
            result["neg"] = [self.encodings[str(neg)] for neg in negs]

        return result

    def __len__(self):
        return self.size


def get_instance_metadata(batch):
    anchor = convert_sample_dict(batch["anchor"])
    pos = convert_sample_dict(batch["pos"])
    negs = batch.get("neg", None)
    if negs is not None:
        negs = [convert_sample_dict(neg) for neg in negs]

    return anchor, pos, negs


class Encoding_Database():
    """
    Saves encodings from triplets to a file such that we do not need to re-evaluate each triplet multiple times.
    """

    def __init__(self,
                 dataset: Abstract_Dataset,
                 dataset_hparams,
                 data_path,
                 transform_str,
                 out_dir,
                 verbose=False):
        self.dataset = dataset
        self.dataset_hparams = dataset_hparams
        self.datasets_contained = []
        self.loaded_files = []

        # these are the only parameters defining differences in databases of encodings (along with the model used...)
        self.hparams = {
            "data_path": data_path,
            "transform": transform_str,
        }
        self.out_dir = out_dir
        self.verbose = verbose

        # create dataloader (for adding the batch dimension when iterating)
        # otherwise it is completely deterministic and the i-th item in the dataloader is dataset[i]
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=1,
                                                 num_workers=0,
                                                 shuffle=False)

    def get_dataset(self, model, name):
        encodings, _ = self.create(model, name)
        return Encoding_Dataset(encodings, self.dataset)

    def save(self, encodings, metadata, delete_old_encodings=True):
        # unique id for this encoding database
        id = uuid.uuid4()

        # save the encodings as pytorch .pt file
        enc_path = join(self.out_dir, f"encodings_{metadata['model']}_{id}.pt")
        torch.save(encodings, enc_path)

        # save the metadata with same id as .json file
        meta_path = join(self.out_dir, f"encodings_meta_{metadata['model']}_{id}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

        print("Saved encodings as:", enc_path, meta_path)

        if delete_old_encodings:
            for meta, enc in self.loaded_files:
                print(f"Remove old encodings, they are contained in new one --> {meta}, {enc}")
                os.remove(join(self.out_dir, meta))
                os.remove(join(self.out_dir, enc))

    def load(self, metadata):
        encodings = {} # all encodings from all valid files
        try:
            files = os.listdir(self.out_dir)
            if self.verbose:
                print("Searching for existing databases...")
                files = tqdm(files)
            for file in files:
                if "encodings_meta" in file:
                    with open(join(self.out_dir, file), "r") as metadata_file:
                        metadata_on_disk = json.load(metadata_file)
                        if metadata_on_disk["model"] == metadata["model"] \
                        and metadata_on_disk["data_params"] == metadata["data_params"]:
                            enc_file_name = file.replace("_meta", "")
                            enc_file_name = enc_file_name.replace(".json", ".pt")
                            encodings_on_disk = torch.load(join(self.out_dir, enc_file_name))

                            if self.verbose:
                                print(f"Use existing database: {file}, {enc_file_name}")

                            encodings.update(encodings_on_disk)
                            self.datasets_contained.extend(metadata_on_disk["datasets_contained"])
                            self.loaded_files.append((file, enc_file_name))

                            if self.dataset_hparams in self.datasets_contained:
                                if self.verbose:
                                    print(f"Database {file} completely contains the dataset.")
                                break

                        else:
                            if self.verbose:
                                print(f"Cannot use {file} because the metadata did not match")
        except:
            pass

        return encodings

    def get_metadata(self, modelname):
        return {
            "model": modelname,
            "data_params": self.hparams,
            "datasets_contained": self.datasets_contained
        }

    def add_encodings(self, database, encodings, meta):
        database[str(meta)] = {
            "encodings": encodings,
            "meta": meta
        }

    def has_encodings(self, database, anchor, pos, negs):
        if not str(anchor) in database:
            return False

        if not str(pos) in database:
            return False

        if negs is not None:
            for neg in negs:
                if not str(neg) in database:
                    return False

        return True

    def calculate(self, metadata, model, database):
        if self.verbose:
            print(f"Calculate encodings for {json.dumps(metadata, sort_keys=True, indent=4)} with data from {json.dumps(self.dataset_hparams, sort_keys=True, indent=4)}")
        else:
            # since this can take a while, we always want to see a print statement for it to not look at the screen and nothing happens for a long time
            print(f"Calculate encodings...")

        if torch.cuda.is_available():
            model = model.to("cuda:0")

        calculate_counter = 0

        for idx, batch in enumerate(tqdm(self.dataloader)):
            # retrieve anchor, pos, negs for saving the further metadata attributes like scanID, instanceID, ...
            anchor, pos, negs = get_instance_metadata(batch)

            # if already everything present in database --> no need to calculate again, continue to next sample
            if self.has_encodings(database, anchor, pos, negs):
                continue
            else:
                calculate_counter += 1

            # encode triplets in one batch via the dataset conversion method
            with torch.no_grad():
                batch = triplets_as_batches(batch, self.dataset.number_negative_samples)
                if torch.cuda.is_available():
                    batch["image"] = batch["image"].to("cuda:0")
                encodings = model(batch)
                encodings = outputs_as_triplets(encodings, self.dataset.number_negative_samples)

                # back to cpu to not overflow gpu memory
                for i, enc in enumerate(encodings):
                    for k, v in enc.items():
                        if k == "neg":
                            encodings[i][k] = [v1.cpu() for v1 in v]
                        else:
                            encodings[i][k] = v.cpu()

                # back to cpu to not overflow gpu memory
                # (because we save the images in the database dict in RAM until written to file in the very end)
                batch["image"] = batch["image"].cpu()

            # save encodings along with metadata per instance in database dict
            anchor_enc = [encodings[i]["anchor"] for i in range(len(encodings))]
            self.add_encodings(database, anchor_enc, anchor)

            pos_enc = [encodings[i]["pos"] for i in range(len(encodings))]
            self.add_encodings(database, pos_enc, pos)

            if negs is not None:
                negs_enc = [encodings[i]["neg"] for i in range(len(encodings))]
                for i in range(len(negs_enc[0])):
                    negs_i_enc = [negs_enc[k][i] for k in range(len(negs_enc))]
                    self.add_encodings(database, negs_i_enc, negs[i])

        return database, calculate_counter

    def create(self, model, name):
        # create metadata (same for loading from disk or calculating)
        metadata = self.get_metadata(name)

        # try to load existing encodings
        # don't need to use loaded metadata because they are similar if loaded successfully
        encodings = self.load(self.get_metadata(name))

        if self.dataset_hparams not in self.datasets_contained:
            # calculate encodings for those that are not already loaoded from previous files
            encodings, new_encodings_counter = self.calculate(metadata, model, encodings)

            # now this dataset is completely represented in these encodings
            self.datasets_contained.append(self.dataset_hparams)

            # update metadata to contain the updated datasets_contained value
            metadata = self.get_metadata(name)

            # always update the encodings (will delete old files)
            # even if new_encodings_counter = 0, we need to update the metadata
            # we cannot simply only update the metadata file because we do not know from how many encoding files we collected everything for this dataset
            self.save(encodings, metadata, True)

        return encodings, metadata

if __name__ == "__main__":
    from data.triplet_dataset import Triplet_Dataset
    from models.bounding_box_encoder import BoundingBoxEncoder
    from torchvision import transforms
    transform = transforms.Compose([
        Triplet_Dataset.rotate_vertical_transform,
        transforms.Resize(256),
        transforms.ToTensor()
    ])

    hparams = {
        "root_path": "/home/lukas/datasets/3RScan/3RScan-10/all_in_one",
        "bbox_data_aligned_vertically": False,
        "transform": str(transform)
    }

    eval_dataset = Triplet_Dataset(
        root_path=hparams["root_path"],
        transform=transform,
        bbox_data_aligned_vertically=hparams["bbox_data_aligned_vertically"],
        preload_all=False,
        cache=False,
        verbose=True)

    models = {
        "bounding_box_encoder_foo": BoundingBoxEncoder(requires_grad=False, use_rmac_layer=True)
    }

    database = Encoding_Database(eval_dataset, hparams, hparams["root_path"], str(transform), "../runs/Evaluation", verbose=True)
    for name, model in models.items():
        dataset = database.get_dataset(model, name)

