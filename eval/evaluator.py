from data.triplet_dataset import Triplet_Dataset
from data.invariance_database import Invariance_Database
from data.invariance_dataset import Invariance_Dataset
from data.triplet_database import Triplet_Database
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SubsetRandomSampler

from eval.encoding_database import Encoding_Database

from eval.visualizer import Visualizer

import torch

from datetime import datetime
from os.path import join
import uuid

from eval.top_k_accuracy import TopK_Accuracy
from eval.feature_distance import Feature_Distance

from tqdm.auto import tqdm

import numpy as np

default_dataset_hparams = {
    "neg_background_overlap_minimum": 128*128,
    "pos_minimum_visibility": Triplet_Database.default_pos_minimum_visibility,
    "pos_maximum_visibility": Triplet_Database.default_pos_maximum_visibility,
    "sample_treshold_per_category": True,
    "sample_subset_percentage": 0.01,
    "sample_fallback": True,
    "sample_treshold": 99,
    "seed": 42,
    "bbox_data_aligned_vertically": True,
}

default_eval_hparams = {

    "positive_types": {
        "all": [0.25, 0.25, 0.25, 0.25],
        "same": [1, 0, 0, 0],
        "other": [0, 1, 0, 0],
        "ambiguity": [0, 0, 1, 0],
        "movement": [0, 0, 0, 1]
    },

    "negative_types": {
        "easy": {
            "prob": [0.5, 0.5, 0, 0, 0],
            "samples": 99,
            "topk": [1,2,3,5,10,20,30,50,70,90]
        },
        "hard": {
            "prob": [0, 0, 0, 0.5, 0.5],
            "samples": 19,
            "topk": [1,2,3,5,10,15]
        },
        "mixed": {
            "prob": [0.2, 0.2, 0.2, 0.2, 0.2],
            "samples": 9,
            "topk": [1,2,3,5]
        }
    }
}

default_visibility_invariance_hparams = {
    "start": 0.2,
    "stop": 1.0,
    "step": 0.1,
}

default_transformation_illuminance_invariance_hparams = {
    "transformation_start": 0.0,
    "transformation_stop": 1.0,
    "transformation_step": 1.0,
    "illuminance_start": 0.0,
    "illuminance_stop": 50.0,
    "illuminance_step": 50.0,
    "sample_treshold_per_anchor": 10,
    "render_size": 128*128,
    "bbox_data_aligned_vertically": default_dataset_hparams["bbox_data_aligned_vertically"]
}

default_visualization_hparams = {
    "n": 2,
    "dim": -1,
    "p": [50],
    "color_mode": Visualizer.Color_Modes.image,
}

def create_id_suffix():
    now = datetime.now()  # current date and time
    return now.strftime("%Y-%b-%d_%H-%M-%S")


def run_one_evaluation(models,
                       dataset_hparams,
                       transform,
                       data_path,
                       out_dir,
                       id_suffix,
                       topk,
                       verbose,
                       invariance_hparams={},
                       visualize_hparams={}):

    # create dataset
    # --> might create a new dataset once if this configuration was never used before
    triplet_dataset = Triplet_Dataset(
        root_path=data_path,
        seed=dataset_hparams["seed"],
        transform=transform,
        number_negative_samples=dataset_hparams["number_negative_samples"],
        positive_sample_probabilities=dataset_hparams["positive_sample_probabilities"],
        negative_sample_probabilities=dataset_hparams["negative_sample_probabilities"],
        neg_background_overlap_minimum=dataset_hparams["neg_background_overlap_minimum"],
        pos_minimum_visibility=dataset_hparams["pos_minimum_visibility"],
        pos_maximum_visibility=dataset_hparams["pos_maximum_visibility"],
        sample_treshold=dataset_hparams["sample_treshold"],
        sample_treshold_per_category=dataset_hparams["sample_treshold_per_category"],
        sample_subset_percentage=dataset_hparams["sample_subset_percentage"],
        sample_fallback=dataset_hparams["sample_fallback"],
        bbox_data_aligned_vertically=dataset_hparams["bbox_data_aligned_vertically"],
        preload_all=False,
        cache=False,
        verbose=verbose)

    if invariance_hparams and \
            (invariance_hparams["transformation_range"] != [0.0, 1.0] or
             invariance_hparams["illuminance_range"] != [0.0, 50.0]):
        invariance_database = Invariance_Database(root_path=data_path,
                                                  triplet_database=triplet_dataset.database,
                                                  render_size=invariance_hparams["render_size"],
                                                  verbose=verbose)

        invariance_dataset = Invariance_Dataset(invariance_database,
                                                triplet_dataset=triplet_dataset,
                                                load_negatives_from_triplet_dataset=True,
                                                transformation_range=invariance_hparams["transformation_range"],
                                                illuminance_range=invariance_hparams["illuminance_range"],
                                                bbox_data_aligned_vertically=dataset_hparams["bbox_data_aligned_vertically"],
                                                transform=transform)

        triplet_dataset = invariance_dataset
        dataset_hparams["transformation_range"] = invariance_hparams["transformation_range"]
        dataset_hparams["illuminance_range"] = invariance_hparams["illuminance_range"]

    dataset_size = len(triplet_dataset)

    if dataset_size < 1:
        print(f"Returning without evaluation because no triplets were found for {dataset_hparams}, {invariance_hparams}")
        return

    # create encodings database + dataset for each model
    # --> might calculate encodings for each model once if they were never calculated before
    encoding_database = Encoding_Database(triplet_dataset,
                                          dataset_hparams,
                                          data_path,
                                          str(transform),
                                          out_dir,
                                          verbose=verbose)
    encoding_datasets = {name: encoding_database.get_dataset(model, name) for name, model in models.items()}

    # create unique id for this run
    eval_id = str(uuid.uuid4())
    log_dir = join(out_dir, id_suffix, eval_id)

    # create top_k_accuracy metric
    log_images_nth = dataset_size // 50 if dataset_size > 50 else 1
    writer = SummaryWriter(log_dir=log_dir)
    top_k_accuracy = TopK_Accuracy(writer=writer,
                                   minK=topk,
                                   hparams=dataset_hparams,
                                   log_images_nth=log_images_nth)

    feature_distance = Feature_Distance(writer=writer,
                                        hparams=dataset_hparams,
                                        log_images_nth=log_images_nth)

    # run top_k_accuracy metric
    visualizers = {}

    with torch.no_grad():
        for model_name, model in models.items():
            print(f"Evaluate model {model_name}")

            # retrieve the encoding_dataset for this model
            encoding_dataset = encoding_datasets[model_name]

            # create dataloader out of the encoding_dataset
            # we could also just iterate the dataset directly since we typically use batch_size of 1 here
            encoding_loader = torch.utils.data.DataLoader(encoding_dataset,
                                                        batch_size=1,
                                                        num_workers=4,
                                                        shuffle=False)

            # iterate dataloader --> evaluate top_k_accuracy
            for idx, encoding in enumerate(tqdm(encoding_loader)):
                batch = None
                if idx % log_images_nth == 0:
                    batch = triplet_dataset[idx]

                # evaluate encoding
                top_k_accuracy.evaluate(encoding,
                                        batch,
                                        model_name)

                feature_distance.evaluate(encoding,
                                          batch,
                                          model_name)

            # show plots if hparams are provided. if not: do not show the visualization
            if visualize_hparams:
                visualizers[model_name] = create_visualizer(encoding_dataset, visualize_hparams, out_dir)

        feature_distance.finish()
        top_k_accuracy.finish()

        print(f"Finished evaluation: {log_dir}")

        return visualizers

def evaluate(models,
             transform,
             data_path,
             custom_dataset_hparams={},
             custom_eval_hparams={},
             custom_invariance_hparams={},
             out_dir="../runs/Evaluation",
             id_suffix=None,
             verbose=False,
             visualize_hparams={}):

    # contains all hparams for the dataset
    dataset_hparams = default_dataset_hparams.copy()
    dataset_hparams.update(custom_dataset_hparams)

    # contains the eval specific hparams for the dataset
    eval_hparams = default_eval_hparams.copy()
    eval_hparams.update(custom_eval_hparams)

    # contains all hparams for the visibility invariance
    invariance_hparams = default_transformation_illuminance_invariance_hparams.copy()
    invariance_hparams.update(custom_invariance_hparams)

    transformation_start = invariance_hparams["transformation_start"]
    transformation_stop = invariance_hparams["transformation_stop"]
    transformation_step = invariance_hparams["transformation_step"]

    illuminance_start = invariance_hparams["illuminance_start"]
    illuminance_stop = invariance_hparams["illuminance_stop"]
    illuminance_step = invariance_hparams["illuminance_step"]

    # unique id for this evaluation
    if id_suffix is None:
        id_suffix = create_id_suffix()

    # save all visualizers that are constructed
    visualizers = {}

    # evaluate different negative types...
    for neg_type_name, neg_type_config in eval_hparams["negative_types"].items():
        print(f"\n----------------------\nEvaluate on {neg_type_name} negatives\n----------------------\n")

        visualizers[neg_type_name] = {}
        # for each negative type evaluate all different positive types...
        for pos_type_name, pos_sample_probabilities in eval_hparams["positive_types"].items():
            print(f"\n----------------------\nEvaluate with positive type {pos_type_name}: {pos_sample_probabilities}\n----------------------\n")

            for t_min in np.arange(transformation_start, transformation_stop, transformation_step):
                for i_min in np.arange(illuminance_start, illuminance_stop, illuminance_step):
                    t_max = t_min + transformation_step
                    i_max = i_min + illuminance_step

                    t_range_str = f"[{t_min}, {t_max}]"
                    i_range_str = f"[{i_min}, {i_max}]"

                    print(
                        f"\n----------------------\nEvaluate in transformation range {t_range_str} and illuminance range {i_range_str}\n----------------------\n")

                    invariance_hparams["transformation_range"] = [t_min, t_max]
                    invariance_hparams["illuminance_range"] = [i_min, i_max]

                    # construct the current dataset_hparams by setting the neg/pos config to the current iteration values
                    dataset_hparams = {
                        **dataset_hparams,
                        "number_negative_samples": neg_type_config["samples"],
                        "negative_sample_probabilities": neg_type_config["prob"],
                        "positive_sample_probabilities": pos_sample_probabilities
                    }

                    # run the evaluation with the current values and save the results
                    visualizers[neg_type_name][pos_type_name] = run_one_evaluation(models,
                                                                                   dataset_hparams,
                                                                                   transform,
                                                                                   data_path,
                                                                                   out_dir,
                                                                                   id_suffix,
                                                                                   neg_type_config["topk"],
                                                                                   verbose,
                                                                                   invariance_hparams=invariance_hparams,
                                                                                   visualize_hparams=visualize_hparams)

    return visualizers

def eval_visibility_invariance(models,
                               transform,
                               data_path,
                               custom_dataset_hparams={},
                               custom_topk_hparams={},
                               custom_visibility_hparams={},
                               out_dir="../runs/Evaluation",
                               id_suffix=None,
                               verbose=False,
                               visualize_hparams={}):

    # contains all hparams for the visibility invariance
    visibility_invariance_hparams = default_visibility_invariance_hparams.copy()
    visibility_invariance_hparams.update(custom_visibility_hparams)

    start = visibility_invariance_hparams["start"]
    stop = visibility_invariance_hparams["stop"]
    step = visibility_invariance_hparams["step"]

    # unique id for this topk_visibility_invariance evaluation
    if id_suffix is None:
        id_suffix = create_id_suffix()

    visualizers = {}

    for min in np.arange(start, stop, step):
        max = min + step
        range_str = f"[{min}, {max}]"

        print(f"\n----------------------\nEvaluate in visibility range {range_str}\n----------------------\n")

        custom_dataset_hparams["pos_minimum_visibility"] = [min, min]
        custom_dataset_hparams["pos_maximum_visibility"] = [max, max]

        vis = evaluate(models,
                       transform,
                       data_path,
                       custom_dataset_hparams=custom_dataset_hparams,
                       custom_eval_hparams=custom_topk_hparams,
                       out_dir=out_dir,
                       id_suffix=id_suffix,
                       verbose=verbose,
                       visualize_hparams=visualize_hparams)

        visualizers[range_str] = vis

    return visualizers


def eval_transformation_illuminance_invariance(models,
                                   transform,
                                   data_path,
                                   custom_invariance_hparams={},
                                   out_dir="../runs/Evaluation",
                                   id_suffix=None,
                                   verbose=False,
                                   visualize_hparams={}):

    # contains all hparams for the visibility invariance
    invariance_hparams = default_transformation_illuminance_invariance_hparams.copy()
    invariance_hparams.update(custom_invariance_hparams)

    transformation_start = invariance_hparams["transformation_start"]
    transformation_stop = invariance_hparams["transformation_stop"]
    transformation_step = invariance_hparams["transformation_step"]

    illuminance_start = invariance_hparams["illuminance_start"]
    illuminance_stop = invariance_hparams["illuminance_stop"]
    illuminance_step = invariance_hparams["illuminance_step"]

    # unique id for this evaluation
    if id_suffix is None:
        id_suffix = create_id_suffix()

    db = Invariance_Database(root_path=data_path,
                             sample_treshold_per_anchor=invariance_hparams["sample_treshold_per_anchor"],
                             render_size=invariance_hparams["render_size"],
                             verbose=verbose)

    visualizers = {}

    for t_min in np.arange(transformation_start, transformation_stop, transformation_step):
        for i_min in np.arange(illuminance_start, illuminance_stop, illuminance_step):
            t_max = t_min + transformation_step
            i_max = i_min + illuminance_step

            t_range_str = f"[{t_min}, {t_max}]"
            i_range_str = f"[{i_min}, {i_max}]"

            print(f"\n----------------------\nEvaluate in transformation range {t_range_str} and illuminance range {i_range_str}\n----------------------\n")

            invariance_hparams["transformation_range"] = [t_min, t_max]
            invariance_hparams["illuminance_range"] = [i_min, i_max]

            dataset = Invariance_Dataset(db,
                                         transform=transform,
                                         transformation_range=invariance_hparams["transformation_range"],
                                         illuminance_range=invariance_hparams["illuminance_range"],
                                         bbox_data_aligned_vertically=invariance_hparams["bbox_data_aligned_vertically"])

            dataset_size = len(dataset)

            print(f"Evalute {len(dataset)} / {db.size} anchor2pos pairs")

            if dataset_size < 1:
                print(f"Returning without evaluation because no anchor2pos were found for {invariance_hparams}")
                continue

            # create encodings database + dataset for each model
            # --> might calculate encodings for each model once if they were never calculated before
            encoding_database = Encoding_Database(dataset,
                                                  invariance_hparams,
                                                  data_path,
                                                  str(transform),
                                                  out_dir,
                                                  verbose=verbose)
            encoding_datasets = {name: encoding_database.get_dataset(model, name) for name, model in models.items()}

            # create unique id for this run
            eval_id = str(uuid.uuid4())
            log_dir = join(out_dir, id_suffix, eval_id)

            # create top_k_accuracy metric
            log_images_nth = dataset_size // 50 if dataset_size > 50 else 1
            writer = SummaryWriter(log_dir=log_dir)

            #top_k_accuracy = TopK_Accuracy(writer=writer,
            #                               minK=topk,
            #                               hparams=dataset_hparams,
            #                               log_images_nth=log_images_nth)

            feature_distance = Feature_Distance(writer=writer,
                                                hparams=invariance_hparams,
                                                log_images_nth=log_images_nth)

            # run top_k_accuracy metric
            with torch.no_grad():
                for model_name, model in models.items():
                    print(f"Evaluate model {model_name}")

                    # retrieve the encoding_dataset for this model
                    encoding_dataset = encoding_datasets[model_name]

                    # create dataloader out of the encoding_dataset
                    # we could also just iterate the dataset directly since we typically use batch_size of 1 here
                    dataset_size = len(encoding_dataset)
                    indices = list(range(dataset_size))
                    sampler = SubsetRandomSampler(indices)
                    encoding_loader = torch.utils.data.DataLoader(encoding_dataset,
                                                                  batch_size=1,
                                                                  num_workers=0,
                                                                  sampler=sampler)

                    # iterate dataloader --> evaluate top_k_accuracy
                    for idx, encoding in enumerate(tqdm(encoding_loader)):

                        batch = None
                        if idx % log_images_nth == 0:
                            batch = dataset[idx]

                        # evaluate encoding
                        feature_distance.evaluate(encoding,
                                                  batch,
                                                  model_name)

                        # show plots if hparams are provided. if not: do not show the visualization
                        if visualize_hparams:
                            if model_name not in visualizers:
                                visualizers[model_name] = {}
                            visualizers[model_name][t_range_str + "_" + i_range_str] = create_visualizer(encoding_dataset, visualize_hparams, out_dir)

                feature_distance.finish()

                print(f"Finished evaluation: {log_dir}")

                return visualizers


def create_visualizer(dataset, visualize_hparams, out_dir):
    visualizer = Visualizer(dataset,
                            hparams=visualize_hparams,
                            out_dir=out_dir,
                            dim=visualize_hparams["dim"],
                            n=visualize_hparams["n"],
                            color_mode=visualize_hparams["color_mode"],
                            p=visualize_hparams["p"],
                            verbose=True)

    print(f"Visualize encodings {visualize_hparams}")

    return visualizer

if __name__ == "__main__":
    from data.triplet_dataset import Triplet_Dataset
    from models.bounding_box_encoder import BoundingBoxEncoder
    from torchvision import transforms
    transform = transforms.Compose([
        Triplet_Dataset.rotate_vertical_transform,
        transforms.Resize(256),
        transforms.ToTensor()
    ])

    dataset_hparams = {
        "root_path": "/home/lukas/datasets/3RScan/3RScan-10/all_in_one",
        "bbox_data_aligned_vertically": False,
        "transform": str(transform),
        "sample_treshold": 9,
        "sample_subset_percentage": 0.1
    }

    eval_hparams = {

        "positive_types": {
            #"all": [0.25, 0.25, 0.25, 0.25],
            "same": [1, 0, 0, 0],
            "other": [0, 1, 0, 0],
            #"ambiguity": [0, 0, 1, 0],
            #"movement": [0, 0, 0, 1]
        },

        "negative_types": {
            #"easy": {
            #    "prob": [0.5, 0.5, 0, 0, 0],
            #    "samples": 3,
            #    "topk": [1,2,3,5,10,20,30,50,70,90]
            #},
            "hard": {
                "prob": [0, 0, 0, 0, 1],
                "samples": 4,
                "topk": [1,2,3,5,10,15]
            },
            #"mixed": {
            #    "prob": [0.2, 0.2, 0.2, 0.2, 0.2],
            #    "samples": 3,
            #    "topk": [1,2,3,5]
            #}
        }
    }

    invariance_hparams = {
        "sample_treshold_per_anchor": 1,
        "render_size": 64,
        "bbox_data_aligned_vertically": False,
    }

    models = {
        "bounding_box_encoder_foo": BoundingBoxEncoder(requires_grad=False, use_rmac_layer=True)
    }

    verbose = True


    visualizers = evaluate(models,
                           transform,
                           dataset_hparams["root_path"],
                           custom_dataset_hparams=dataset_hparams,
                           custom_eval_hparams=eval_hparams,
                           custom_invariance_hparams=invariance_hparams,
                           verbose=verbose,
                           #visualize_hparams=default_visualization_hparams
                           )

    print(visualizers)

    eval_visibility_invariance(models,
                               transform,
                               dataset_hparams["root_path"],
                               custom_dataset_hparams=dataset_hparams,
                               custom_topk_hparams=eval_hparams,
                               verbose=verbose,
                               #visualize_hparams=default_visualization_hparams
                               )


    eval_transformation_illuminance_invariance(models,
                       transform,
                       dataset_hparams["root_path"],
                       custom_invariance_hparams=invariance_hparams,
                       verbose=verbose,
                       #visualize_hparams=default_visualization_hparams
                       )