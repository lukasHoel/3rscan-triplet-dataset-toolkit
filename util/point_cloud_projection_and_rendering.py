import torch
import torch.nn as nn

from torchvision.transforms import ToTensor, Resize, Compose, Lambda, ToPILImage

import torch.nn.functional as F

#import open3d as o3d
import numpy as np

from os.path import join

from PIL import Image

import csv

EPS = 1e-2

import torch
from torch import nn

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer.points import rasterize_points

class RasterizePointsXYsBlending(nn.Module):
    """
    Rasterizes a set of points using a differentiable renderer. Points are
    accumulated in a z-buffer using an accumulation function
    defined in opts.accumulation and are normalised with a value M=opts.M.
    Inputs:
    - pts3D: the 3D points to be projected
    - src: the corresponding features
    - C: size of feature
    - learn_feature: whether to learn the default feature filled in when
                     none project
    - radius: where pixels project to (in pixels)
    - size: size of the image being created
    - points_per_pixel: number of values stored in z-buffer per pixel
    - opts: additional options

    Outputs:
    - transformed_src_alphas: features projected and accumulated
        in the new view
    """

    def __init__(
        self,
        C,
        size,
        learn_feature=True,
        radius=1.5,
        rad_pow=2,
        accumulation_tau=0,
        accumulation='alphacomposite',
        points_per_pixel=1,
    ):
        super().__init__()
        if learn_feature:
            default_feature = nn.Parameter(torch.randn(1, C, 1))
            self.register_parameter("default_feature", default_feature)
        else:
            default_feature = torch.zeros(1, C, 1)
            self.register_buffer("default_feature", default_feature)

        self.radius = radius
        self.size = size
        self.points_per_pixel = points_per_pixel

        self.rad_pow = rad_pow
        self.accumulation_tau = accumulation_tau
        self.accumulation = accumulation

    def forward(self, pts3D, src):
        bs = src.size(0)
        if len(src.size()) > 3:
            bs, c, w, _ = src.size()
            image_size = w

            pts3D = pts3D.permute(0, 2, 1)
            src = src.unsqueeze(2).repeat(1, 1, w, 1, 1).view(bs, c, -1)
        else:
            bs = src.size(0)
            image_size = self.size

        # Make sure these have been arranged in the same way
        assert pts3D.size(2) == 3
        assert pts3D.size(1) == src.size(2)

        # pts3D.shape = (bs, w*h, 3) --> (x,y,z) coordinate for ever element in the image raster
        # Because we have done re-projection, the i-th coordinate in the image raster must no longer be identical to (x,y)!
        # src.shape = (bs, c, w*h) --> c features for every element in the image raster (w*h)

        #print("Features: {}".format(src.shape))
        #print("3D Pointcloud: {}".format(pts3D.shape))

        # flips the x and y coordinate
        pts3D[:,:,1] = - pts3D[:,:,1]
        pts3D[:,:,0] = - pts3D[:,:,0]

        # Add on the default feature to the end of the src
        #src = torch.cat((src, self.default_feature.repeat(bs, 1, 1)), 2)

        radius = float(self.radius) / float(image_size) * 2.0 # convert radius to fit the [-1,1] NDC ?? Or is this just arbitrary scaling s.t. radius as meaningful size?
        params = compositing.CompositeParams(radius=radius)

        #print("Radius - before: {}, converted: {}".format(self.radius, radius))

        pts3D = Pointclouds(points=pts3D, features=src.permute(0,2,1))
        points_idx, _, dist = rasterize_points(
            pts3D, image_size, radius, self.points_per_pixel
        ) # see method signature for meaning of these output values

        #print("points_idx: {}".format(points_idx.shape))
        #print("dist: {}".format(points_idx.shape))

        #print("Max dist: ", dist.max(), pow(radius, self.rad_pow))

        dist = dist / pow(radius, self.rad_pow) # equation 1 from the paper (3.2): this calculates N(p_i, l_mn) from the d2 dist

        #print("Max dist: ", dist.max())

        alphas = (
            (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
            .pow(self.accumulation_tau)
            .permute(0, 3, 1, 2)
        ) # equation 2 from the paper (3.2): prepares alpha values for composition of the feature vectors

        #print("alphas: ", alphas.shape)
        #print("pointclouds object: {}".format(pts3D.features_packed().shape))
        #print("alphas: ", alphas)

        if self.accumulation == 'alphacomposite':
            transformed_src_alphas = compositing.alpha_composite(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0), # pts3D also contains features here, because this is now a Pointclouds object
                params,
            )
        elif self.accumulation == 'wsum':
            transformed_src_alphas = compositing.weighted_sum(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
                params,
            )
        elif self.accumulation == 'wsumnorm':
            transformed_src_alphas = compositing.weighted_sum_norm(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
                params,
            )
        else: raise NotImplementedError('Unsupported accumulation type: ' + self.accumulation)

        return transformed_src_alphas


class Projector(nn.Module):

    def __init__(self,
                 W=512,
                 H=512,
                 ):
        super().__init__()

        self.img_shape = (H, W)

        # create coordinate system for x and y
        xs = torch.linspace(0, W - 1, W)
        ys = torch.linspace(0, H - 1, H)

        xs = xs.view(1, 1, 1, W).repeat(1, 1, H, 1)
        ys = ys.view(1, 1, H, 1).repeat(1, 1, 1, W)

        # build homogeneous coordinate system with [X, Y, 1, 1] to prepare for depth
        xyzs = torch.cat(
            (xs, ys, torch.ones(xs.size()), torch.ones(xs.size())), 1
        ).view(1, 4, -1)

        self.register_buffer("xyzs", xyzs)

        self.norm_factor_flow = np.sqrt(W*W + H*H);

    def save_pointcloud(self, colors, points, file_path):
        pass
 #       pcd = o3d.geometry.PointCloud()
 #       pcd.points = o3d.utility.Vector3dVector(
 #           points[:,:3,:].permute(0, 2, 1).cpu().numpy().astype(np.float64).squeeze())
 #       if colors is not None:
 #           pcd.colors = o3d.utility.Vector3dVector(colors.view(3, -1).permute((1, 0)).cpu().numpy().astype(np.float64))
 #       o3d.io.write_point_cloud(file_path, pcd)

    def project_to_other_view(
            self, depth, K, K_inv, cam1_to_world, world_to_cam2, colors=None, save_path=None
    ):
        save_to_file = save_path is not None and colors is not None

        if save_to_file:
            self.save_pointcloud(colors, self.xyzs, join(save_path, "input_image.ply"))

        # add Zs to the coordinate system
        # input_projected is then [X*Z, -Y*Z, -Z, 1] with Z being the depth of the image
        input_projected = self.xyzs * depth
        input_projected[:, -1, :] = 1
        if save_to_file:
            self.save_pointcloud(colors, input_projected, join(save_path, "input_projected.ply"))

        # Transform into camera coordinate of the first view
        cam1 = K_inv.bmm(input_projected)
        if save_to_file:
            self.save_pointcloud(colors, cam1, join(save_path, "cam1.ply"))

        # Transform to World Coordinates with RT of input view
        world = cam1_to_world.bmm(cam1)
        if save_to_file:
            self.save_pointcloud(colors, world, join(save_path, "world.ply"))

        # Transform from World coordinates to camera of output view
        cam2 = world_to_cam2.bmm(world)
        if save_to_file:
            self.save_pointcloud(colors, cam2, join(save_path, "cam2.ply"))

        # Apply intrinsics / go back to image plane
        output_projected = K.bmm(cam2)
        if save_to_file:
            self.save_pointcloud(colors, output_projected, join(save_path, "output_projected.ply"))

        # remove invalid zs that cause nans
        mask = (output_projected[:, 2:3, :].abs() < EPS).detach()
        zs = output_projected[:, 2:3, :]
        zs[mask] = EPS

        # here we concatenate (x,y) / z and the original z-coordinate into a new (x,y,z) vector
        image = torch.cat((output_projected[:, 0:2, :] / zs, output_projected[:, 2:3, :]), 1)
        if save_to_file:
            self.save_pointcloud(colors, image, join(save_path, "output_image.ply"))
            # TODO need to save only x,y and set z to zero to have the correct output_image.ply file?
            # TODO the file here is not looking right --> compare to ICL implementation, it should be flat in input image size again when opened in meshlab!!!

        image[:, 0, :] = image[:, 0, :] / float(self.img_shape[1] - 1) * 2 - 1
        image[:, 1, :] = image[:, 1, :] / float(self.img_shape[0] - 1) * 2 - 1

        # here we set (x,y,z) to -10 where we have invalid zs that cause nans
        image[mask.repeat(1, 3, 1)] = 10

        # calculate flow
        flow = (image[:, 0:2, :] - self.xyzs[:, 0:2, :])
        dcre = torch.norm(flow, p=2, dim=1) # calculates L2 norm for each (x, y) vector where x is in dim_0 and y is in dim_1
        dcre = torch.min(dcre / self.norm_factor_flow, torch.tensor(1.0, device=self.xyzs.get_device()))
        dcre[depth.squeeze(1) < 0] = 1.0

        # TODO Find a good treshold
        # Percentage of dcre values per pixel that are below treshold.
        n = dcre.shape[1]
        dcre = torch.sum(dcre < 0.1) * 1.0 / n

        # Mean of dcre values
        #dcre = torch.mean(dcre)

        return image, flow, dcre


    def forward(
            self, depth, K, K_inv, cam1_to_world, world_to_cam2, colors=None, save_path="/home/lukas/Desktop/"
    ):

        if len(depth.size()) > 3:
            # reshape into the right positioning
            depth = depth.view(depth.shape[0], 1, -1)

        result = self.project_to_other_view(
            depth, K, K_inv, cam1_to_world, world_to_cam2, colors, save_path
        )

        return result
