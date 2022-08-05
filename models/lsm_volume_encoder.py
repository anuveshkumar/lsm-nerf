import torch
import torch.nn as nn
import torch.nn.functional as F

import models.grid_fusion
import models.grid_reasoning
import models.grid_reasoning_simple
import models.image_encoder
import camera.unprojection


class LSM(nn.Module):
    def __init__(self, grid_scale, grid_size, device, use_fusion=True):
        super(LSM, self).__init__()

        # hyperparameters for Grid Fusion model
        height = width = depth = grid_size
        in_channels = 32  # output of image encoder
        hidden_dim = [32, 32]  # last hidden dim is output dim
        kernel_size = (3, 3, 3)  # kernel size for two stacked hidden layer
        num_layers = 2  # number of stacked hidden layer

        # Cuda/gpu setup
        self.device = device

        # 2d unet
        self.image_enc = models.image_encoder.ImUnet().to(self.device)

        # unprojection
        self.grid_scale = grid_scale  # bounding volume of the scene
        self.grid_size = (grid_size,) * 3

        # Grid Fusion model
        self.use_fusion = use_fusion
        self.grid_fusion = models.grid_fusion.ConvGRU(input_size=(depth, height, width),
                                                      input_dim=in_channels,
                                                      hidden_dim=hidden_dim[0],
                                                      output_dim=hidden_dim[-1],
                                                      kernel_size=kernel_size,
                                                      num_layers=num_layers,
                                                      device=self.device,
                                                      bias=True,
                                                      return_all_layers=False).to(self.device)

        self.grid_reasoning = models.grid_reasoning.Modified3DUNet(in_channels=hidden_dim[-1], n_classes=8,
                                                                   base_n_filter=8).to(self.device)

        # self.grid_reasoning = models.grid_reasoning_simple.Simple3DUNet(in_channels=hidden_dim[-1],
        #                                                                 base_n_filter=8).to(self.device)

    def forward(self, data):
        # assuming batch_size == 1
        imgs = data['src_rgbs'].permute(0, 1, 4, 2, 3)
        src_cameras = data['src_cameras'].squeeze()
        batch_size = imgs.shape[0]
        n_views = imgs.shape[1]
        imgs_feats = self.image_enc(imgs.view(-1, 3, imgs.shape[-2], imgs.shape[-1]))
        proj_feats = []
        for j in range(len(imgs_feats)):
            proj_feats.append(
                camera.unprojection.unproj_grid(self.grid_scale, self.grid_size, imgs_feats[j], src_cameras[j],
                                                self.device))
        proj_feats = torch.stack(proj_feats)
        proj_feats = proj_feats.permute(0, 2, 1)
        # proj_feats = self.get_projected_features(imgs_feats, src_cameras)

        proj_feats = proj_feats.view(batch_size, n_views, proj_feats.shape[1], *self.grid_size)
        if self.use_fusion:
            layer_output_list, last_state_list = self.grid_fusion(proj_feats)
            fused_feature_grid = last_state_list[0]
        else:
            feature_grid_mean = torch.mean(proj_feats, dim=1)
            fused_feature_grid = torch.sum(torch.pow(proj_feats - feature_grid_mean, 2), dim=1) / (proj_feats.shape[1] -1)
        final_gird = self.grid_reasoning(fused_feature_grid)

        return final_gird

    def get_projected_features(self, img_feats, src_cameras):
        def rot_to_quat(R):
            batch_size, _, _ = R.shape
            q = torch.ones((batch_size, 4)).cuda()

            R00 = R[:, 0, 0]
            R01 = R[:, 0, 1]
            R02 = R[:, 0, 2]
            R10 = R[:, 1, 0]
            R11 = R[:, 1, 1]
            R12 = R[:, 1, 2]
            R20 = R[:, 2, 0]
            R21 = R[:, 2, 1]
            R22 = R[:, 2, 2]

            q[:, 0] = torch.sqrt(1.0 + R00 + R11 + R22) / 2
            q[:, 1] = (R21 - R12) / (4 * q[:, 0])
            q[:, 2] = (R02 - R20) / (4 * q[:, 0])
            q[:, 3] = (R10 - R01) / (4 * q[:, 0])
            return q

        def create_3d_grid(aabb, grid_size):
            grid_min, grid_max = aabb
            x = torch.linspace(grid_min, grid_max, grid_size)
            y = torch.linspace(grid_min, grid_max, grid_size)

            yy, xx = torch.meshgrid(x, y)
            xyz = []
            for i in x:
                zz = torch.ones_like(xx) * i
                xyz.append(torch.stack((xx, yy, zz), dim=-1))

            grid_points = torch.stack(xyz)
            # grid_points = grid_points[:-1, :-1, :-1]
            return grid_points

        def compute_projection(xyz, src_cameras):
            original_shape = xyz.shape[:3]
            xyz = xyz.reshape(-1, 3)
            num_views = len(src_cameras)
            src_intrinsics = src_cameras[:, 2:18].reshape(-1, 4, 4) # extract the intrinsic
            src_poses = src_cameras[:, -16:].reshape(-1, 4, 4) # extract the extrinsic (rotation/translation)
            xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1) # homogenize the coordinate (basically append 1)
            # the meat
            # 1. src_poses are currently camera_to_world (means, convert points from camera space to world space
            # 2. we inverse it to convert it to world_to_camera (torch.inverse(src_poses))
            # 3. we then matrix multiply all the points with this world_to_camera matrix, to obtain coordinates of the same point
            # but with respect to camera space # torch.inverse(src_poses).bmm(xyz_h...)
            # 4. we then matrix multiply it by src_intrinsics, an intrinsic consists of the focal length, and image_height/width
            # It is essentially scaling/recentering these coordinates to align with the image plane
            projections = src_intrinsics.bmm(torch.inverse(src_poses)) \
                .bmm(xyz_h.t()[None, ...].repeat(num_views, 1, 1))
            projections = projections.permute(0, 2, 1)

            # now that we have the points, in camera space, aligning with the image plane, we divide it by z (distance of the point from the image plane)
            # we therefore obtain 2d points, telling us the x, y location on the image plane
            # it is possible that a 3d point might project outside the bounds of the image plane
            pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)
            pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
            return pixel_locations.reshape((num_views,) + original_shape + (2,))

        def normalize(pixel_locations, h, w):
            resize_factor = torch.tensor([w - 1., h - 1.]).to(pixel_locations.device)[None, None, :]
            normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.

            return normalized_pixel_locations

        grid = create_3d_grid(self.aabb, self.grid_size)
        pixel_points = compute_projection(grid.to(self.device), src_cameras.squeeze())  # assuming batch_size 1
        h, w = src_cameras[:, :2][0]  # assuming all images are of same shape
        normalized_pixel_points = normalize(pixel_points, h, w)
        normalized_pixel_points = normalized_pixel_points.reshape(*normalized_pixel_points.shape[:2], -1, 2)
        projected_features = F.grid_sample(img_feats, normalized_pixel_points, align_corners=True)
        projected_features = projected_features.reshape(*projected_features.shape[:2], *(self.grid_size,) * 3)
        return projected_features
