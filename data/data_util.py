import numpy as np
import torchvision
from PIL import Image
import torch

# original image sizes from the 3RScan dataset's images in the sequence folder
orig_width = 960
orig_height = 540

def rotate_bbox_minus_90(bbox):
    """
    This is a temporary fix for wrongly calculated bboxes.
    They are assumed to be calculated w.r.t. an already rotated image (image is already vertical).
    Instead, we need them to be calculated w.r.t. the camera images in their original orientation (horizontally).
    Thus, we rotate them back 90 degree counter-clockwise.

    TODO This is only a temporary fix because we should not have to expect the bboxes to be in this wrong format.
    TODO As soon as this format issue is fixed, we can completely remove this method.

    :param bbox:
    :return:
    """

    x1 = bbox[1]
    x2 = bbox[3]

    y1 = orig_height - bbox[2]
    y2 = orig_height - bbox[0]

    return x1, y1, x2, y2


def apply_bbox_sanity_check(bbox, parsed_instance):
    """
    Check if a bbox still is large enough and if not make it larger.
    A bbox must at least have a width and height of 8.

    :param bbox: bbox as loaded from a row of 2DInstances.txt (an entry in self.instances)
    :param parsed_instance: instance for this bbox, used for logging purposes

    :return: bbox but with ensured minimum width and height
    """

    if bbox[0] == -1 and bbox[1] == -1 and bbox[2] == -1 and bbox[3] == -1:
        # See this as error, we do not know how to resolve this
        raise ValueError("Needed to fix -1 bbox case for ", parsed_instance["scan"], parsed_instance["frame_nr"],
                         parsed_instance["label"], bbox, parsed_instance["bbox"])

    if bbox[2] - bbox[0] < 8:
        # This can be no error because we just "enlarge" the bbox a little bit
        # print("Needed to fix x bbox case for ", parsed_instance["scan"], parsed_instance["frame_nr"],
        #      parsed_instance["label"], bbox, parsed_instance["bbox"])
        x1 = bbox[0] - 8
        x1 = x1 if x1 >= 0 else 0
        x2 = bbox[2] + 8
        x2 = x2 if x2 <= 959 else 959
        bbox = (x1, bbox[1], x2, bbox[3])

    if bbox[3] - bbox[1] < 8:
        # This can be no error because we just "enlarge" the bbox a little bit
        # print("Needed to fix y bbox case for ", parsed_instance["scan"], parsed_instance["frame_nr"],
        #      parsed_instance["label"], bbox, parsed_instance["bbox"])
        y1 = bbox[1] - 8
        y1 = y1 if y1 >= 0 else 0
        y2 = bbox[3] + 8
        y2 = y2 if y2 <= 959 else 959
        bbox = (bbox[0], y1, bbox[2], y2)

    return bbox


def transform_bbox(bbox, bbox_data_aligned_vertically, transform, cache):
    """
    Resizes/rescales the bbox by applying the specified transformation (self.transform).

    :param bbox: bbox as loaded from a row of 2DInstances.txt (an entry in self.instances)
    :return: transformed bbox in format (x1, y1, x2, y2) or (-1, -1, -1, -1) if bbox is empty after transformation
    """

    if bbox_data_aligned_vertically:
        x1, y1, x2, y2 = rotate_bbox_minus_90(bbox)
    else:
        x1, y1, x2, y2 = bbox

    if transform is None:
        return x1, y1, x2, y2

    cached_transformed_bbox = cache.get(str(bbox), None)
    if cached_transformed_bbox is not None:
        return cached_transformed_bbox

    # convert to boolean mask image
    bbox_mask = np.zeros((orig_height, orig_width),
                         dtype=np.uint8)
    bbox_mask[y1:y2 + 1, x1:x2 + 1] = 255

    # apply torch transforms, convert back to numpy
    pil = Image.fromarray(bbox_mask).convert("RGB")

    if isinstance(transform.transforms[-1], torchvision.transforms.ToTensor):
        transform = torchvision.transforms.Compose([
            *transform.transforms[:-1]
        ])
        pil = transform(pil).convert("L")
    else:
        pil = transform(pil).convert("L")

    bbox_mask = np.asarray(pil)

    # extract new bbox
    mask_indices = np.argwhere(bbox_mask > 0)  # where is mask now == 1 after transforming it?
    if mask_indices.shape[0] != 0:
        # maybe after downscaling a very small mask area, it vanished completely: in such cases do not add the bbox
        min_mask = np.amin(mask_indices, axis=0)  # minimum (y,x) index: upper-left corner of mask
        max_mask = np.amax(mask_indices, axis=0)  # maximum (y,x) index: bottom-right corner of mask
        x1 = min_mask[1]  # min x value
        y1 = min_mask[0]  # min y value
        x2 = max_mask[1]  # max x value
        y2 = max_mask[0]  # max y value
        transformed_bbox = x1, y1, x2, y2
    else:
        transformed_bbox = -1, -1, -1, -1

    cache[str(bbox)] = transformed_bbox

    return transformed_bbox


def triplets_as_batches(batch, number_negative_samples):
        """
        Take in one batch of this dataset and concatenate all anchor, pos and neg attributes into one large image tensor.
        This is done for faster processing on the GPU instead of processing each anchor, pos and neg image tensor independently.

        We also concatenate the bboxes and labels into one large tensor / list.
        The rest of the batch is discarded. (TODO: we might change this later by creating a list for each of the other attributes if needed)

        :param batch: batched samples of this dataset
        :return: dict with format
        {
            "image": images stacked in a tensor with size <batch_size>*(2+self.number_negative_samples).
                     The first <batch_size> images correspond to the anchor images, the second to the positive images.
                     The remaining <batch_size>*self.number_negative_samples correspond to the negative images.

            "bbox": bboxes stacked in a tensor with size <batch_size>*(2+self.number_negative_samples).
                     Same ordering as for images.

            "label": labels stacked in a list with size <batch_size>*(2+self.number_negative_samples)
                     Same ordering as for images.
        }
        """

        x = batch

        # load anchor, pos, negs images and cat them into one tensor.
        # final batch_size = batch_size of incoming batch * (2+self.number_negative_samples)
        anchor_images = x["anchor"]["image"]
        pos_images = x["pos"]["image"]
        if number_negative_samples > 0:
            neg_images = tuple([x["neg"][i]["image"] for i in range(number_negative_samples)])
        else:
            neg_images = ()

        images = (anchor_images, pos_images)
        images += neg_images

        images = torch.cat(images, dim=0)

        # load anchor, pos, negs bboxes and cat each x,y,w,h into one tensor
        # final batch_size = batch_size of incoming x,y,w,h * (2+self.number_negative_samples)
        bboxes = {}
        for k in x["anchor"]["bbox"].keys():
            anchor_bbox = x["anchor"]["bbox"][k]
            pos_bbox = x["pos"]["bbox"][k]
            if number_negative_samples > 0:
                neg_bbox = tuple([x["neg"][i]["bbox"][k] for i in range(number_negative_samples)])
            else:
                neg_bbox = ()

            bboxes[k] = (anchor_bbox, pos_bbox)
            bboxes[k] += neg_bbox

            bboxes[k] = torch.cat(bboxes[k], dim=0)

        # load anchor, pos, negs labels and extend them into one list
        labels = []

        labels.extend(x["anchor"]["label"])
        labels.extend(x["pos"]["label"])
        if number_negative_samples > 0:
            for neg in x["neg"]:
                labels.extend(neg["label"])

        return {
            "image": images,
            "bbox": bboxes,
            "label": labels,
        }


def outputs_as_triplets(output, number_negative_samples):
        """
        Reverses the process of self.triplets_as_batches by splitting the tensors into anchor, pos and neg again.
        The incoming 'output' vector is assumed to contain the image encodings of each image only.
        Either the encodings are stored in a list with the n-th entry being the n-th encoding for one image or we only
        have one encoding per image.

        :param output: image encodings
        :return: dictionary with format
        {
            "anchor": all anchor encodings in a stacked tensor of size <batch_size>,
            "pos": all positive encodings in a stacked tensor of size <batch_size>,
            "neg": all negative encodings in a stacked tensor of size <batch_size>*self.number_negative_samples
        }
        """

        if (isinstance(output, list)):
            out = []
            for i in range(len(output)):
                out_i = output[i]
                bs = int(out_i.shape[0] / (number_negative_samples + 2))

                if number_negative_samples > 0:
                    x = torch.split(out_i, [bs, bs, number_negative_samples * bs], dim=0)
                else:
                    x = torch.split(out_i, [bs, bs], dim=0)

                out_i = {}
                out_i["anchor"] = x[0]
                out_i["pos"] = x[1]
                if number_negative_samples > 0:
                    out_i["neg"] = [x[2][i * bs:(i + 1) * bs] for i in range(number_negative_samples)]

                out.append(out_i)

            return out
        else:
            bs = int(output.shape[0] / (number_negative_samples + 2))

            if number_negative_samples > 0:
                x = torch.split(output, [bs, bs, number_negative_samples * bs], dim=0)
            else:
                x = torch.split(output, [bs, bs], dim=0)

            out = {}
            out["anchor"] = x[0]
            out["pos"] = x[1]
            if number_negative_samples > 0:
                out["neg"] = [x[2][i * bs:(i + 1) * bs] for i in range(number_negative_samples)]

            return [out]