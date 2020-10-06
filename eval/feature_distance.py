from models.losses.metrics import l2_dist_sum_weighted
import torch
import torchvision.transforms as tf

import numpy as np

from os.path import join

class Feature_Distance:
    def __init__(self, writer, hparams=None, log_images_nth=100):
        self.writer = writer
        self.hparams = hparams
        self.log_images_nth = log_images_nth

        self.anchor_pos_distances = {
            "instance": {},
            "class": {}
        }

        self.anchor_mean_neg_distances = {
            "instance": {},
            "class": {}
        }

        self.counters = {}

    def get_counter(self, model):
        if model not in self.counters:
            self.counters[model] = 0

        return self.counters[model]

    def update_counter(self, model):
        self.counters[model] += 1

    def update_distances(self, distances_per_instance, distances_per_class, model, room, instance_id, label, distances):
        # check if model already set
        if model not in distances_per_instance:
            distances_per_instance[model] = {}
        if model not in distances_per_class:
            distances_per_class[model] = {}

        # check if room already set
        if room not in distances_per_instance[model]:
            distances_per_instance[model][room] = {}
        if room not in distances_per_class[model]:
            distances_per_class[model][room] = {}

        # check if instance already set
        if instance_id not in distances_per_instance[model][room]:
            distances_per_instance[model][room][instance_id] = []

        # check if label already set
        if label not in distances_per_class[model][room]:
            distances_per_class[model][room][label] = []

        # add to instance list
        distances_per_instance[model][room][instance_id].append(distances)

        # add to class list
        distances_per_class[model][room][label].append(distances)

    def evaluate(self, encodings, batch, model_prefix):

        # retrieve metadata attributes
        room = encodings["anchor"]["meta"]["reference"][0]
        instance_id = str(encodings["anchor"]["meta"]["instance_id"].detach().cpu().numpy().item())
        label = encodings["anchor"]["meta"]["label"][0]

        # retrieve or initialize counters
        total_counter = self.get_counter(model_prefix)

        # --------------
        # ANCHOR TO POS
        # --------------

        # anchor to pos distances per level of encodings
        pos_distances = {
            str(i): l2_dist_sum_weighted([encodings["anchor"]["encodings"][i]], [encodings["pos"]["encodings"][i]]).detach().cpu().numpy()
        for i in range(len(encodings["anchor"]["encodings"]))}

        # add mean pos distance to list
        pos_distances["total"] = l2_dist_sum_weighted(encodings["anchor"]["encodings"], encodings["pos"]["encodings"]).detach().cpu().numpy()

        # add pos distances to lists
        self.update_distances(self.anchor_pos_distances["instance"],
                              self.anchor_pos_distances["class"],
                              model_prefix,
                              room,
                              instance_id,
                              label,
                              pos_distances)

        # log pos distances in line graph plot
        key = join("FD/Pos/Batch", model_prefix)
        self.writer.add_scalars(key, pos_distances, total_counter)

        # --------------
        # ANCHOR TO NEG
        # --------------

        negs = encodings["neg"]

        neg_distances = {str(i): 0.0 for i in range(len(encodings["anchor"]["encodings"]))}
        neg_distances["total"] = 0

        # calculate anchor->neg dist for all negs
        for neg in negs:
            neg_distances["total"] += l2_dist_sum_weighted(encodings["anchor"]["encodings"], neg["encodings"]).detach().cpu().numpy()
            for i in range(len(neg["encodings"])):
                neg_distances[str(i)] += l2_dist_sum_weighted([encodings["anchor"]["encodings"][i]], [neg["encodings"][i]]).detach().cpu().numpy()

        # create mean of neg_distances for all layers and for the total distance per neg
        for k,v in neg_distances.items():
            neg_distances[k] = 1.0 * v / len(negs)

        # add neg distances to lists
        self.update_distances(self.anchor_mean_neg_distances["instance"],
                              self.anchor_mean_neg_distances["class"],
                              model_prefix,
                              room,
                              instance_id,
                              label,
                              neg_distances)

        # log neg distances in line graph plot
        key = join("FD/Neg/Batch", model_prefix)
        self.writer.add_scalars(key, neg_distances, total_counter)

        # log batch images and metadata
        if (total_counter % self.log_images_nth) == 0:
            images = []
            images.append(batch["anchor"]["image"])
            images.append(batch["pos"]["image"])
            for neg in batch["neg"]:
                images.append(neg["image"])

            bboxes = []
            bboxes.append(batch["anchor"]["bbox"])
            bboxes.append(batch["pos"]["bbox"])
            for neg in batch["neg"]:
                bboxes.append(neg["bbox"])

            resizer = tf.Compose([
                tf.ToPILImage(),
                tf.Resize(size=(64, 64)),
                tf.ToTensor()
            ])

            for i in range(len(images)):
                x = bboxes[i]["x"]
                y = bboxes[i]["y"]
                w = bboxes[i]["w"]
                h = bboxes[i]["h"]
                images[i] = resizer(images[i][:, y:y + h, x:x + w])

            key = join("FD/Batch", model_prefix)
            info = join(room, instance_id, label)

            self.writer.add_images(key,
                                   torch.stack(images),
                                   global_step=total_counter)

            self.writer.add_text(key,
                                 info,
                                 global_step=total_counter)

        # update counters for this model_prefix: one new evaluation done
        self.update_counter(model_prefix)

    def calculate_mean_and_variance_of_distance(self, distances):
        mean_distances_per_key = {}
        mean_distances_per_room = {}
        mean_distances_per_model = {}
        mean_distances = {}

        var_distances_per_key = {}
        var_distances_per_room = {}
        var_distances_per_model = {}
        var_distances = {}

        types = []

        for model, rooms in distances.items():
            # init
            mean_distances[model] = {}
            var_distances[model] = {}

            for room, keys in rooms.items():
                # init
                mean_distances[model][room] = {}
                var_distances[model][room] = {}

                for key, distances in keys.items():
                    # init
                    mean_distances[model][room][key] = {}
                    var_distances[model][room][key] = {}
                    types = distances[0].keys()

                    for type in types:
                        mean = np.mean([d[type] for d in distances])
                        var = np.var([d[type] for d in distances])

                        mean_distances_per_key[join(type, model, room, key)] = mean
                        var_distances_per_key[join(type, model, room, key)] = var

                        # for traversing back up the tree
                        mean_distances[model][room][key][type] = mean
                        var_distances[model][room][key][type] = var

                mean_distances[model][room]["total"] = {}
                var_distances[model][room]["total"] = {}

                for type in types:
                    mean = np.mean([key[type] for dict_key, key in mean_distances[model][room].items() if dict_key != "total"])
                    var = np.mean([key[type] for dict_key, key in var_distances[model][room].items() if dict_key != "total"])

                    mean_distances_per_room[join(type, model, room)] = mean
                    var_distances_per_room[join(type, model, room)] = var

                    # for traversing back up the tree
                    mean_distances[model][room]["total"][type] = mean
                    var_distances[model][room]["total"][type] = var

            for type in types:
                mean = np.mean([room["total"][type] for room in mean_distances[model].values()])
                var = np.mean([room["total"][type] for room in var_distances[model].values()])

                mean_distances_per_model[join(type, model)] = mean
                var_distances_per_model[join(type, model)] = var

        return mean_distances_per_key, var_distances_per_key,\
               mean_distances_per_room, var_distances_per_room,\
               mean_distances_per_model, var_distances_per_model

    def write_mean_and_var(self, distances_per_instance, distances_per_class, prefix):
        mean_instance, var_instance, mean_room, var_room, mean_model, var_model = self.calculate_mean_and_variance_of_distance(distances_per_instance)
        mean_class, var_class, _, _, _, _ = self.calculate_mean_and_variance_of_distance(distances_per_class)

        # log mean/variance per instance, per room
        for i, (k, v) in enumerate(mean_instance.items()):
            self.writer.add_scalar(f"{prefix}/Mean/Instance", v, i)
            self.writer.add_text(f"{prefix}/Mean/Instance", k, i)
        for i, (k, v) in enumerate(var_instance.items()):
            self.writer.add_scalar(f"{prefix}/Var/Instance", v, i)
            self.writer.add_text(f"{prefix}/Var/Instance", k, i)

        # log mean/variance per class, per room
        for i, (k, v) in enumerate(mean_class.items()):
            self.writer.add_scalar(f"{prefix}/Mean/Class", v, i)
            self.writer.add_text(f"{prefix}/Mean/Class", k, i)
        for i, (k, v) in enumerate(var_class.items()):
            self.writer.add_scalar(f"{prefix}/Var/Class", v, i)
            self.writer.add_text(f"{prefix}/Var/Class", k, i)

        # log mean/variance per room
        for i, (k, v) in enumerate(mean_room.items()):
            self.writer.add_scalar(f"{prefix}/Mean/Room", v, i)
            self.writer.add_text(f"{prefix}/Mean/Room", k, i)
        for i, (k, v) in enumerate(var_room.items()):
            self.writer.add_scalar(f"{prefix}/Var/Room", v, i)
            self.writer.add_text(f"{prefix}/Var/Room", k, i)

        # log mean/variance
        self.writer.add_scalars(f"{prefix}/Mean", mean_model, 0)
        self.writer.add_scalars(f"{prefix}/Var", var_model, 0)

        for k, v in mean_model.items():
            print(f"{prefix}/Mean/" + k, v)

        return mean_model, var_model

    def finish(self):

        # calculate mean pos
        mean_model_pos, var_model_pos = self.write_mean_and_var(self.anchor_pos_distances["instance"],
                                                                self.anchor_pos_distances["class"], "FD/Pos")

        # calculate mean neg
        mean_model_neg, var_model_neg = self.write_mean_and_var(self.anchor_mean_neg_distances["instance"],
                                                                self.anchor_mean_neg_distances["class"], "FD/Neg")

        # log hparams
        if self.hparams is not None:
            self.writer.add_hparams({str(k): str(v) for k, v in self.hparams.items()}, {"FD/Pos/Mean/"+k: v for k,v in mean_model_pos.items()})


