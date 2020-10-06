from eval.encoding_database import Encoding_Dataset
import torch
import uuid
from tqdm.auto import tqdm
import numpy as np
import matplotlib.cm as cm
from enum import Enum
from sklearn import manifold
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from os.path import join
from torchvision.transforms import ToPILImage, Resize, Compose, Lambda
import os

def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    X_center =  torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components  = v[:, :k]
    explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return { 'X':X, 'k':k, 'components':components,
    'explained_variance':explained_variance }

def visualize_scatter_with_images(X_2d_data, images, figsize=(45,45), image_zoom=1):
    """
    from https://www.kaggle.com/gaborvecsei/plants-t-sne
    """
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.show()

class ImageAnnotations3D():
    """
    from https://stackoverflow.com/questions/48180327/matplotlib-3d-scatter-plot-with-images-as-annotations
    """
    def __init__(self, xyz, imgs, ax3d, ax2d):
        self.xyz = xyz
        self.imgs = imgs
        self.ax3d = ax3d
        self.ax2d = ax2d
        self.annot = []
        for s,im in zip(self.xyz, self.imgs):
            x,y = self.proj(s)
            self.annot.append(self.image(im,[x,y]))
        self.lim = self.ax3d.get_w_lims()
        self.rot = self.ax3d.get_proj()
        self.cid = self.ax3d.figure.canvas.mpl_connect("draw_event",self.update)

        self.funcmap = {"button_press_event" : self.ax3d._button_press,
                        "motion_notify_event" : self.ax3d._on_move,
                        "button_release_event" : self.ax3d._button_release}

        self.cfs = [self.ax3d.figure.canvas.mpl_connect(kind, self.cb) \
                        for kind in self.funcmap.keys()]

    def cb(self, event):
        event.inaxes = self.ax3d
        self.funcmap[event.name](event)

    def proj(self, X):
        """ From a 3D point in axes ax1,
            calculate position in 2D in ax2 """
        x,y,z = X
        x2, y2, _ = proj3d.proj_transform(x,y,z, self.ax3d.get_proj())
        tr = self.ax3d.transData.transform((x2, y2))
        return self.ax2d.transData.inverted().transform(tr)

    def image(self,arr,xy):
        """ Place an image (arr) as annotation at position xy """
        im = offsetbox.OffsetImage(arr, zoom=2)
        im.image.axes = self.ax3d
        ab = offsetbox.AnnotationBbox(im, xy, xybox=(-30., 30.),
                            xycoords='data', boxcoords="offset points",
                            pad=0.3, arrowprops=dict(arrowstyle="->"))
        self.ax2d.add_artist(ab)
        return ab

    def update(self,event):
        if np.any(self.ax3d.get_w_lims() != self.lim) or \
                        np.any(self.ax3d.get_proj() != self.rot):
            self.lim = self.ax3d.get_w_lims()
            self.rot = self.ax3d.get_proj()
            for s,ab in zip(self.xyz, self.annot):
                ab.xy = self.proj(s)

class Visualizer():

    class Color_Modes(Enum):
        instance = 0
        label = 1
        room = 2
        image = 3

    default_filter = {
        "labels": [],
        "rooms": [],
        "instances": [], # instances must be formatted in the same way as in top_k_accuracy's incorrect statistic
    }

    def __init__(self,
                 encoding_dataset: Encoding_Dataset,
                 hparams,
                 out_dir,
                 dim=-1,
                 n=3,
                 color_mode: Color_Modes = Color_Modes.instance,
                 p=[1,5,10,50,100],
                 verbose=False):
        self.encoding_dataset = encoding_dataset
        self.dim = dim
        self.n = n
        self.p = p
        self.color_mode = color_mode
        self.out_dir = out_dir
        self.hparams = hparams
        self.verbose = verbose

        visualization = self.load()
        if visualization:
            # use data from file
            self.X = visualization["X"]
            self.X_pca = visualization["X_pca"]
            self.X_pca_for_tsne = visualization.get("X_pca_for_tsne", None)
            self.X_tsne = [torch.from_numpy(x) for x in visualization["X_tsne"]]
            self.colors = visualization["colors"]
            self.images = visualization["images"]
            self.instances = visualization["instances"]
        else:
            # calculate from scratch with dataset
            self.X, self.colors, self.images, self.instances = self.create_encodings()
            self.X_pca = PCA_svd(self.X.T, self.n)["components"]
            self.X_pca_for_tsne = PCA_svd(self.X.T, 50)["components"]

            print("Calculate t-SNE embeddings...")
            self.X_tsne = [
                manifold.TSNE(n_components=self.n,
                              init='random',
                              random_state=0,
                              perplexity=p)
                        .fit_transform(self.X_pca_for_tsne)
            for p in tqdm(self.p)]

        self.remove_black_borders(self.images)

    def remove_black_borders(self, images):
        # assume similar black_border for all images
        first_img = images[0]
        non_black_pixels_mask = np.any(first_img != [0, 0, 0], axis=-1)

        image_transform = Compose([
            ToPILImage(),
            Resize((45, 45)),
            Lambda(lambda x: np.asarray(x))
        ])

        for idx, img in enumerate(images):
            cropped_img = img[np.ix_(non_black_pixels_mask.any(1),non_black_pixels_mask.any(0))]
            images[idx] = image_transform(cropped_img)

    def save(self):
        # unique id for this visualization database
        id = uuid.uuid4()

        data = self.get_data()

        # save the encodings as pytorch .pt file
        save_path = join(self.out_dir, f"visualizations_{id}.pt")
        torch.save(data, save_path)

    def load(self):
        visualization = {} # all encodings from all valid files
        try:
            files = os.listdir(self.out_dir)
            if self.verbose:
                print("Searching for existing databases...")
                files = tqdm(files)
            for file in files:
                if "visualizations" in file:
                    vis_on_disk = torch.load(join(self.out_dir, file))
                    config = self.get_config()
                    if "config" not in vis_on_disk or \
                        config["p"] != vis_on_disk["config"]["p"] or \
                        config["n"] != vis_on_disk["config"]["n"] or \
                        config["dim"] != vis_on_disk["config"]["dim"] or \
                        config["color_mode"].value != vis_on_disk["config"]["color_mode"].value:
                        if self.verbose:
                            print(f"Cannot use {file} because the config did not match")
                        continue
                    else:
                        visualization = vis_on_disk
                        if self.verbose:
                            print(f"Use existing database: {file}")
                        break
        except:
            pass

        return visualization

    def get_config(self):
        return {
            "p": self.p,
            "n": self.n,
            "dim": self.dim,
            "color_mode": self.color_mode,
            #"hparams": self.hparams
        }

    def get_data(self):
        return {
            "X": self.X,
            "X_pca": self.X_pca,
            "X_pca_for_tsne": self.X_pca_for_tsne,
            "X_tsne": self.X_tsne,
            "colors": self.colors,
            "images": self.images,
            "instances": self.instances,
            "config": self.get_config()
        }

    def get_key(self, enc, idx):
        key = None

        if self.color_mode == Visualizer.Color_Modes.instance:
            key = enc["anchor"]["meta"]["instance_id"]
        elif self.color_mode == Visualizer.Color_Modes.label:
            key = enc["anchor"]["meta"]["label"]
        elif self.color_mode == Visualizer.Color_Modes.image:
            key = idx
        elif self.color_mode == Visualizer.Color_Modes.room:
            key = enc["anchor"]["meta"]["reference"]

        return str(key)

    def create_colors(self):
        print("Create visualization colors...")

        # create set of keys
        keys = {self.get_key(enc, idx) for idx, enc in enumerate(tqdm(self.encoding_dataset))}

        # create color for each label
        color_table = cm.rainbow(np.linspace(0, 1, len(keys)))

        # access color key
        color_table = {key: color_table[idx] for idx, key in enumerate(keys)}

        return color_table

    def filter(self, filter, instance):
        # analyze filter. all keys that are different are ignored, only these ones count.
        filter_empty = {
            "rooms": len(filter.get("rooms", [])) == 0,
            "labels": len(filter.get("labels", [])) == 0,
            "instances": len(filter.get("instances", [])) == 0
        }

        # if any filter that is not empty does not contain the object, then it does not pass that filter and we can return False immediately.
        for key, is_empty in filter_empty.items():
            if not is_empty and instance[key] not in filter[key]:
                return False

        # if we reach this, all filters were passed
        return True

    def create_encodings(self):
        color_table = self.create_colors()
        colors = []
        images = []
        X = []
        instances = []

        image_transform = Compose([
            Lambda(lambda x: x.detach().cpu()),
            ToPILImage(),
            Resize(45),
            Lambda(lambda x: np.asarray(x))
        ])

        print("Retrieve visualization data...")
        for idx, enc in enumerate(tqdm(self.encoding_dataset)):
            # save metadata
            instances.append({
                "rooms": enc["anchor"]["meta"]["reference"],
                "labels": enc["anchor"]["meta"]["label"],
                "instances": enc["anchor"]["meta"]["reference"] + "_" + enc["anchor"]["meta"]["label"] + "_" + str(enc["anchor"]["meta"]["instance_id"])
            })

            # load image to be visualized
            image = self.encoding_dataset.dataset[idx]["anchor"]["image"]
            image = image_transform(image)
            images.append(image)

            # load color to be visualized
            key = self.get_key(enc, idx)
            colors.append(color_table[key])

            # load encodings to be visualized. Always load encoding regardless of filter because otherwise the t-SNE and PCA would be different!!
            anchor_encodings = enc["anchor"]["encodings"]
            if self.dim == -1:
                all_encs = torch.cat(anchor_encodings, dim=1)
                X.append(all_encs.clone())
            else:
                X.append(anchor_encodings[self.dim].clone())

        assert (len(X) == len(colors))
        assert (len(colors) == len(images))
        assert (len(images) == len(instances))

        images = np.array(images)
        X = torch.cat(X)

        return X, colors, images, instances

    def apply_filter_to_data(self, X, filter):
        # select the indices that pass the filter and those that dont
        passed_indices = []
        removed_indices = []
        for idx, instance in enumerate(self.instances):
            if self.filter(filter, instance):
                passed_indices.append(idx)
            else:
                removed_indices.append(idx)

        # create a white image and the color white. These will be the colors of the images that are filtered
        # we do not simply filter the images out, because that changes the scale of the pyplt plot.
        # instead by drawing it "invisible" in white we keep the same scale as when drawing it without any filter.
        filter_image = np.zeros([45,45,3],dtype=np.uint8)
        filter_image.fill(255)
        filter_color = [1, 1, 1]

        # create list of all passed images and a list containing the filter_image as many times as an image was filtered
        passed_images = [img for idx, img in enumerate(self.images) if idx in passed_indices]
        removed_images = [filter_image for i in range(len(removed_indices))]

        # create list of all passed colors and a list containing the filter_color as many times as a color was filtered
        passed_colors = [c for idx, c in enumerate(self.colors) if idx in passed_indices]
        removed_colors = [filter_color for i in range(len(removed_indices))]

        # change z order: pyplt draws over existing images in the order of the list that is passed to it.
        # So put the passed indices last such that they are drawn on top
        X = torch.cat((X[removed_indices, :], X[passed_indices, :]))
        images = [*removed_images, *passed_images]
        colors = [*removed_colors, *passed_colors]

        return X, images, colors

    def visualize(self, X, filter):

        # retrieve the filtered data
        X, images, colors = self.apply_filter_to_data(X, filter)

        # visualize according to settings
        if self.color_mode == Visualizer.Color_Modes.image:
            if self.n == 2:
                # static 2d image plot from: https://www.kaggle.com/gaborvecsei/plants-t-sne
                visualize_scatter_with_images(X,
                                              images=images,
                                              figsize=(45,45),
                                              image_zoom=0.7)
            elif self.n == 3:
                # slow 3d interactive image plot from: https://stackoverflow.com/questions/48180327/matplotlib-3d-scatter-plot-with-images-as-annotations
                fig = plt.figure()
                ax = fig.add_subplot(111, projection=Axes3D.name)
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=colors)

                # Create a dummy axes to place annotations to
                ax2 = fig.add_subplot(111, frame_on=False)
                ax2.axis("off")
                ax2.axis([0, 1, 0, 1])

                ImageAnnotations3D(np.c_[X[:, 0], X[:, 1], X[:, 2]], images, ax, ax2)
                plt.show()

        elif self.n == 2:
            # scatter plot with plt in 2d
            plt.scatter(X[:, 0], X[:, 1], color=colors)
            plt.show()
        elif self.n == 3:
            # scatter plot with plt in 3d
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=colors)
            plt.show()

    def visualize_pca(self, filter=default_filter):
        self.visualize(self.X_pca, filter)

    def visualize_tsne(self, filter=default_filter):
        for p, tsne in zip(self.p, self.X_tsne):
            print("Perplexity", p)
            self.visualize(tsne, filter)

if __name__ == "__main__":
    from data.triplet_dataset import Triplet_Dataset
    from eval.encoding_database import Encoding_Database
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
        number_negative_samples=3,
        verbose=True,
        sample_fallback=True,
        sample_treshold=10)

    models = {
        "bounding_box_encoder_foo": BoundingBoxEncoder(requires_grad=False, use_rmac_layer=True)
    }

    filters = {
        "rooms": [
                  #"09582212-e2c2-2de1-9700-fa44b14fbded",
                  #"c92fb594-f771-2064-879b-cc598a9dabe5",
                  #"c92fb576-f771-2064-845a-a52a44a9539f",
                  #"7747a506-9431-24e8-87d9-37a5654d41f4"
                  #"0cac7549-8d6f-2d13-8d56-b895956f571a"
                 ],
        "labels": [
                   #"sofa",
                   #"bed",
                   #"couch"
                   #"cabinet",
                   #"table",
                   #"desk chair"
                    "office chair"
                  ],
        "instances": [
                    #"09582212-e2c2-2de1-9700-fa44b14fbded_chair_8",
                    #"c92fb576-f771-2064-845a-a52a44a9539f_cabinet_24",
                    #"0cac755a-8d6f-2d13-8fed-b1be02f4ef77_sofa_2",
                    #"c92fb594-f771-2064-879b-cc598a9dabe5_office chair_20"
        ]
    }

    database = Encoding_Database(eval_dataset, hparams, hparams["root_path"], str(transform), "../runs/Evaluation", verbose=False)
    for name, model in models.items():
        dataset = database.get_dataset(model, name)
        visualizer = Visualizer(encoding_dataset=dataset,
                                out_dir="../runs/Evaluation",
                                hparams=hparams,
                                n=2,
                                dim=-1,
                                color_mode=Visualizer.Color_Modes.image,
                                p=[50])

        visualizer.visualize_pca(filters)
        visualizer.visualize_pca()

        visualizer.visualize_tsne(filters)
        visualizer.visualize_tsne()