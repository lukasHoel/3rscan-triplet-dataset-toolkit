from models.losses.metrics import l2_dist_sum_weighted
import torch
import torchvision.transforms as tf


class TopK_Accuracy:

    def __init__(self, writer=None, minK=[1, 5], hparams=None, log_images_nth=100):
        self.correct = {}
        self.incorrect = {}
        self.total = {}
        self.writer = writer
        self.minK = minK
        self.hparams = hparams
        self.log_images_nth = log_images_nth
        # TODO param: precision/recall should be evaluated as well? (treshold of it)

    def add_sample(self, dict, room, instance_id, label):
        if room not in dict["rooms"]:
            dict["rooms"][room] = 1
        else:
            dict["rooms"][room] += 1

        if label not in dict["labels"]:
            dict["labels"][label] = 1
        else:
            dict["labels"][label] += 1

        instance = room + "_" + label + "_" + instance_id
        if instance not in dict["instances"]:
            dict["instances"][instance] = 1
        else:
            dict["instances"][instance] += 1

        dict["total"] += 1

    def evaluate(self, encodings, batch, model_prefix):

        # retrieve metadata attributes
        room = encodings["anchor"]["meta"]["reference"][0]
        instance_id = str(encodings["anchor"]["meta"]["instance_id"].detach().cpu().numpy().item())
        label = encodings["anchor"]["meta"]["label"][0]

        # retrieve correct / total for this model from previous evaluations
        correct = self.correct.get(model_prefix, [{
            "rooms": {},
            "instances": {},
            "labels": {},
            "total": 0
        } for k in self.minK])
        incorrect = self.incorrect.get(model_prefix, [{
            "rooms": {},
            "instances": {},
            "labels": {},
            "total": 0
        } for k in self.minK])
        total = self.total.get(model_prefix, 0)

        # initialize distances for this evaluation
        distances = []

        # get anchor, pos, negs from encodings
        anchor = encodings["anchor"]["encodings"]
        pos = encodings["pos"]["encodings"]
        negs = encodings["neg"]

        # calculate anchor->pos dist
        pos_dist = l2_dist_sum_weighted(anchor, pos)
        distances.append(pos_dist)

        # calculate anchor->neg dist for all negs
        for neg in negs:
            neg_dist = l2_dist_sum_weighted(anchor, neg["encodings"])
            distances.append(neg_dist)

        # sort distances descendingly
        distances.sort()

        # check if pos_dist in shortest k distances.
        for idx, k in enumerate(self.minK):
            # if k < distances (+1?)
            if pos_dist in distances[:k]:  # (k+1?)
                self.add_sample(correct[idx], room, instance_id, label)
            else:
                self.add_sample(incorrect[idx], room, instance_id, label)

        # log batch images with model_prefix
        if self.writer is not None and (total % self.log_images_nth) == 0:

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

            labels = []
            labels.append(batch["anchor"]["label"])
            labels.append(batch["pos"]["label"])
            for neg in batch["neg"]:
                labels.append(neg["label"])

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

            self.writer.add_images("Top-K/Image/Batch/" + model_prefix,
                                   torch.stack(images),
                                   global_step=total)

            self.writer.add_text("Top-K/Image/Batch/" + model_prefix,
                                 " ".join(labels),
                                 global_step=total)

        # complete evaluation
        total += 1

        # update values for this model_prefix: one new evaluation done
        self.correct[model_prefix] = correct
        self.incorrect[model_prefix] = incorrect
        self.total[model_prefix] = total

    def finish(self):
        if self.writer is not None:
            # log classification accuracies
            accs = {}
            for idx, k in enumerate(self.minK):
                # add correct statistics
                key = "Top-K/Acc/Top-" + str(k)
                val = {}
                for model in self.correct.keys():
                    val[model] = self.correct[model][idx]["total"] * 1.0 / self.total[model]
                accs[key] = max(val.values())  # for hparam logging only one number is supported
                print(key, val)
                self.writer.add_scalars(key, val, 0)

                # add incorrect statistics
                for level, incorrect_dict in self.incorrect[model][idx].items():
                    key = "Top-K/Incorrect/Top-" + str(k) + "/" + level
                    # only log it, if we actually have at least one incorrect value, otherwise the statistic has no value
                    if level != "total" and incorrect_dict:
                        ratio_dict = {}
                        # go through each instance and create the ratio r = correct / ( correct + incorrect )
                        for instance, incorrect_counter in incorrect_dict.items():
                            correct_counter = self.correct[model][idx][level].get(instance, 0) if self.correct[model][idx][level] is not None else 0
                            ratio_dict[instance] = 1.0 * correct_counter / ( correct_counter + incorrect_counter)
                        print(key, ratio_dict)
                        self.writer.add_scalars(key, ratio_dict, 0)

            # log hparams
            if self.hparams is not None:
                self.writer.add_hparams({str(k):str(v) for k,v in self.hparams.items()}, accs)


