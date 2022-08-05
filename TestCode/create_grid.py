import torch
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datasets.nerf_synthetic import NerfSyntheticDataset
from torch.utils.data import DataLoader
from sample_ray import RaySamplerSingleImage
from render_ray import sample_along_camera_rays
import math
import torch.nn.functional as F


# def create_grid_3d(aaabbb, grid_density=32, voxel_centers=True):
#     # grid is axis aligned (right, up, forward) such that:
#     # x coordinates, represent the Height
#     # y coordinates, represent the Width
#     # z coordinates, represent the Depth
#
#     # select voxel_size on the basis of x (to ensure voxels coordinates are of a cube)
#     # grid density -> based on the grid size if the grid were to be a cube
#     x_minmax = (aaabbb[0], aaabbb[3])
#     y_minmax = (aaabbb[1], aaabbb[4])
#     z_minmax = (aaabbb[2], aaabbb[5])
#
#     voxel_size = (x_minmax[1] - x_minmax[0]) / grid_density
#     height = grid_density  # traversal only in x, changes height
#     width = len(torch.arange(y_minmax[0], y_minmax[1], voxel_size))
#     depth = len(torch.arange(z_minmax[0], z_minmax[1], voxel_size))
#     if not height == width == depth:
#         raise NotImplementedError("Doesn't work, modify the code to have cuboidal coordinates instead")
#     # working with voxel centers is easier because they are symmetrical
#     x_cube_coords = torch.arange(y_minmax[0] + voxel_size / 2, y_minmax[1], voxel_size).repeat(height, depth, 1)
#     y_cube_coords = torch.arange(x_minmax[0] + voxel_size / 2, x_minmax[1], voxel_size).flip(0). \
#         repeat(width, depth, 1).permute(2, 0, 1)
#     z_cube_coords = torch.arange(z_minmax[0] + voxel_size / 2, z_minmax[1], voxel_size).flip(0). \
#         repeat(height, width, 1).permute(1, 0, 2)
#     planes = []
#     grid_3d = torch.stack((x_cube_coords, y_cube_coords, z_cube_coords))
#     for i in range(height):
#         plane = torch.stack((x_cube_coords[i, :, :], y_cube_coords[:, i, :], z_cube_coords[:, :, i]))
#         planes.append(plane)
#
#     grid_3d = torch.stack(planes, dim=-1)
#
#     if not voxel_centers:
#         # gives bottom lower coordinate of each voxel
#         # this also means that sum of the coordinates will not be zero, because we're missing the 3 planes
#         grid_3d -= voxel_size / 2
#
#     return grid_3d.numpy(), voxel_size


# aaabbb = (-1, -2, -3, 1, 2, 3)
# aaabbb = (-2.0, -2.0, -2.0, 2.0, 2.0, 2.0)
# coordinate_grid, voxel_size = create_grid_3d(aaabbb, grid_density=4, voxel_centers=False)

# min, max = -2.0, 2.0
# grid_density = 10
# # voxel_size = (max - min) / grid_density
# # grid_range = torch.arange(min + voxel_size / 2, max, voxel_size)
# grid_range = torch.linspace(-1.0, 1.0, 10)
# coordinate_grid_base = torch.stack(torch.meshgrid([grid_range, grid_range, grid_range]))

# points = torch.FloatTensor(100, 3).uniform_(-1.0, 1.0).numpy()
#
# points = (points + 1.0) * (grid_density - 1.0) / 2.0
#
# indexes = np.floor(points)
# # indexes = np.floor(points * grid_density / (max - min)) + grid_density / 2  # needs to be symmetrical about center
# indexes = indexes.astype(np.int32)
# indexes_plus = indexes + 1
# index_points = coordinate_grid_base[:, indexes[:, 0], indexes[:, 1], indexes[:, 2]]
# index_points_plus = coordinate_grid_base[:, indexes_plus[:, 0], indexes_plus[:, 1], indexes_plus[:, 2]]
# ratio1, ratio2 = check(points, index_points, index_points_plus)
# print(grid_range, ratio1, ratio2)
# grid_flat = coordinate_grid[:, :, :, ].reshape(3, -1)


def check(points, bottom, top):
    vec1 = (points[0, :] >= bottom[0, :]) * (points[1, :] >= bottom[1, :]) * (points[2, :] >= bottom[2, :])
    ratio1 = torch.sum(vec1) / len(vec1)

    vec1 = (points[:, 0] < top[:, 0]) * (points[:, 1] < top[:, 1]) * (points[:, 2] < top[:, 2])
    ratio2 = torch.sum(vec1) / len(vec1)
    print(ratio1, ratio2)


def meshgrid(depth, height, width):
    x_t = torch.linspace(-1.0, 1.0, width).expand(height * depth, width).reshape(depth, height, width)
    y_t = torch.linspace(-1.0, 1.0, height).expand(width * depth, height).reshape(depth, width, height)
    y_t = y_t.permute(0, 2, 1)

    z_t = torch.linspace(-1.0, 1.0, depth).expand(width * height, depth).reshape(height, width, depth)
    z_t = z_t.permute(2, 0, 1)

    x_t_flat = x_t.reshape(1, -1)
    y_t_flat = y_t.reshape(1, -1)
    z_t_flat = z_t.reshape(1, -1)

    ones = torch.ones_like(x_t_flat)
    grid = torch.cat([x_t_flat, y_t_flat, z_t_flat], 0)
    return grid


def trilineate(feature_grid, coordinate_grid_flat, points, grid_density=10, grid_scale=2.0):
    # print(output)
    channels = feature_grid.squeeze(0).shape[0]  # C, H, W, D
    point_features = torch.zeros(channels, len(points[1]))
    feature_grid_flat = feature_grid.reshape(channels, -1)

    # what I used to do (The proper method)

    # move points to range [-1, 1] # corresponding to the grid. This is to align the points to the grid_indexing
    points = points / grid_scale  # scale 3d points to range [-1, 1]
    mask = inbound_mask(points)
    points = points[:, mask]
    # move the points to [0, 1] and the scale as per the grid
    points_scaled = (points + 1.0) * (grid_density - 1.0) / 2.0

    points_bottom = torch.floor(points_scaled).long()  # get the lower bottom indices
    points_top = points_bottom + 1  # get the top high indices

    bottom_coordinate = coordinate_grid_flat[:,
                        points_bottom[0, :] + points_bottom[1, :] * grid_density + points_bottom[2, :] * (
                                grid_density ** 2)]
    top_coordinate = coordinate_grid_flat[:,
                     points_top[0, :] + points_top[1, :] * grid_density + points_top[2, :] * (grid_density ** 2)]

    coordinates = []
    features = []

    for z in range(2):
        for y in range(2):
            for x in range(2):
                coordinates.append(coordinate_grid_flat[:,
                                   (points_bottom[0, :] + x) +
                                   (points_bottom[1, :] + y) * grid_density +
                                   (points_bottom[2, :] + z) * (grid_density ** 2)])
                features.append(feature_grid_flat[:,
                                (points_bottom[0, :] + x) +
                                (points_bottom[1, :] + y) * grid_density +
                                (points_bottom[2, :] + z) * (grid_density ** 2)])

    back_plane_point = torch.cat((points[:2, :], bottom_coordinate[2:, :]), dim=0)
    front_plane_point = torch.cat((points[:2, :], top_coordinate[2:, :]), dim=0)

    back_up_line_point = torch.cat(
        (back_plane_point[:1, :], top_coordinate[1:2, :], back_plane_point[2:, :]),
        dim=0)
    back_down_line_point = torch.cat(
        (back_plane_point[:1, :], bottom_coordinate[1:2, :], back_plane_point[2:, :]),
        dim=0)
    front_up_line_point = torch.cat(
        (front_plane_point[:1, :], top_coordinate[1:2, :], front_plane_point[2:, :]),
        dim=0)
    front_down_line_point = torch.cat(
        (front_plane_point[:1, :], bottom_coordinate[1:2, :], front_plane_point[2:, :]),
        dim=0)

    # LINEAR INTERPOLATIONS ([variation only in x])
    # back down
    back_down_distance = (coordinates[1] - coordinates[0])[0, :]
    a_back_down = (back_down_line_point - coordinates[0])[0, :] / back_down_distance
    b_back_down = (coordinates[1] - back_down_line_point)[0, :] / back_down_distance
    back_down_feature = features[0] * b_back_down + features[1] * a_back_down

    # back up
    back_up_distance = (coordinates[3] - coordinates[2])[0, :]
    a_back_up = (back_up_line_point - coordinates[2])[0, :]
    b_back_up = (coordinates[3] - back_up_line_point)[0, :]
    back_up_feature = features[2] * b_back_up + features[3] * a_back_up

    # front_down
    front_down_distance = (coordinates[5] - coordinates[4])[0, :]
    a_front_down = (front_down_line_point - coordinates[4])[0, :] / front_down_distance
    b_front_down = (coordinates[5] - front_down_line_point)[0, :] / front_down_distance
    front_down_features = features[4] * b_front_down + features[5] * a_front_down

    # front_up
    front_up_distance = (coordinates[7] - coordinates[6])[0, :]
    a_front_up = (front_up_line_point - coordinates[6])[0, :] / front_up_distance
    b_front_up = (coordinates[7] - front_up_line_point)[0, :] / front_up_distance
    front_up_features = features[6] * b_front_up + features[7] * a_front_up

    # BILINEAR INTERPOLATIONS([variations only in y])
    # back_plane
    back_plane_distance = (back_up_line_point - back_down_line_point)[1, :]
    a_back = (back_plane_point - back_down_line_point)[1, :] / back_plane_distance
    b_back = (back_up_line_point - back_plane_point)[1, :] / back_plane_distance
    back_feature = back_down_feature * b_back + back_up_feature * a_back

    # front_plane
    front_plane_distance = (front_up_line_point - front_down_line_point)[1, :]
    a_front = (front_plane_point - front_down_line_point)[1, :] / front_plane_distance
    b_front = (front_up_line_point - front_plane_point)[1, :] / front_plane_distance
    front_feature = front_down_features * b_front + back_up_feature * a_front
    # What I thought actually needs to be done

    # TRILINEAR INTERPOLATION([variation only in z])
    point_distance = (front_plane_point - back_plane_point)[2, :]
    a = (front_plane_point - points)[2, :] / point_distance
    b = (points - back_plane_point)[2, :] / point_distance
    final_feature = back_feature * b + front_feature * a

    point_features[:, mask] = final_feature
    return point_features.permute(1, 0)


def inbound_mask(points):
    x_mask = (-1 <= points[0, :]) * (points[0, :] < 1)
    y_mask = (-1 <= points[1, :]) * (points[1, :] < 1)
    z_mask = (-1 <= points[2, :]) * (points[2, :] < 1)
    return x_mask * y_mask * z_mask


# .permute(0, 2, 1, 3)
grid_density = 10
grid_scale = 2.0
coordinate_grid_flat = meshgrid(grid_density, grid_density, grid_density)
feature_grid = torch.randn(1, 8, grid_density, grid_density, grid_density).requires_grad_(True)
points = torch.FloatTensor(3, 4096).uniform_(-2.5, 2.5)
point_features = trilineate(points=points, feature_grid=feature_grid, coordinate_grid_flat=coordinate_grid_flat,
                            grid_density=grid_density,
                            grid_scale=grid_scale)

# points = points.permute(1, 0).reshape(1, 1, 100, 1, 3)
# from_library = F.grid_sample(feature_grid, points).squeeze().permute(1, 0)
print('wadu')
# get the image features
# 1. scale grid
# 2. unproject image features onto the 3d points
# 3. get the feature grid

# get the sampled points on the rays


# point_for_grid_sample = points.reshape(1, 1, 1, 1, -1)  # batch_size, num_rays, num_samples, (x, y, z)
# interpolated_feature = F.grid_sample(feature_grid, point_for_grid_sample)
# print(interpolated_feature)

# Option/Hack 1 (euclidean based)
# distances = []
# for i in range(len(coordinates)):
#     distances.append(torch.sqrt(torch.sum(((points - coordinates[i]) ** 2), dim=0)))
#
# weights = torch.softmax((1 / torch.stack(distances)), dim=0)
#
# point_features = []
# for i in range(len(coordinates)):
#     point_features.append(weights[i] * features[i])
#
# point_features = torch.stack(point_features).sum(dim=0)

# Option/Hack 2 (grid corners)
# weight_0 = (coordinates[7][0, :] - points[0, :]) * (coordinates[7][1, :] - points[1, :]) * \
#            (coordinates[7][2, :] - points[2, :])
# weight_1 = (points[0, :] - coordinates[6][0, :]) * (coordinates[6][1, :] - points[1, :]) * \
#            (coordinates[6][2, :] - points[2, :])
# weight_2 = (coordinates[5][0, :] - points[0, :]) * (points[1, :] - coordinates[5][1, :]) * \
#            (coordinates[5][2, :] - points[2, :])
# weight_3 = (points[0, :] - coordinates[4][0, :]) * (points[1, :] - coordinates[4][1, :]) * \
#            (coordinates[4][2, :] - points[2, :])
# weight_4 = (coordinates[3][0, :] - points[0, :]) * (coordinates[3][1, :] - points[1, :]) * \
#            (points[2, :] - coordinates[3][2, :])
# weight_5 = (points[0, :] - coordinates[2][0, :]) * (coordinates[2][1, :] - points[1, :]) * \
#            (points[2, :] - coordinates[2][2, :])
# weight_6 = (coordinates[1][0, :] - points[0, :]) * (points[1, :] - coordinates[1][1, :]) * \
#            (points[2, :] - coordinates[1][2, :])
# weight_7 = (points[0, :] - coordinates[0][0, :]) * (points[1, :] - coordinates[1][1, :]) * \
#            (points[2, :] - coordinates[1][2, :])
#
# output = weight_0 * features[0] + weight_1 * features[1] + weight_2 * features[2] + weight_3 * features[3] \
#          + weight_4 * features[4] + weight_5 * features[5] + weight_6 * features[6] + weight_7 * features[7]

#
# fig = go.Figure(data=[
#     # go.Scatter3d(x=bottom_coordinate[0, :], y=bottom_coordinate[1, :], z=bottom_coordinate[2, :],
#     #              mode='markers',
#     #              marker=dict(size=8)),
#     # go.Scatter3d(x=top_coordinate[0, :], y=top_coordinate[1, :], z=top_coordinate[2, :],
#     #              mode='markers',
#     #              marker=dict(size=8)),
#     go.Scatter3d(x=coordinates[0][0, :], y=coordinates[0][1, :], z=coordinates[0][2, :],
#                  mode='markers',
#                  marker=dict(size=8)),
#     go.Scatter3d(x=coordinates[1][0, :], y=coordinates[1][1, :], z=coordinates[1][2, :],
#                  mode='markers',
#                  marker=dict(size=8)),
#     go.Scatter3d(x=coordinates[2][0, :], y=coordinates[2][1, :], z=coordinates[2][2, :],
#                  mode='markers',
#                  marker=dict(size=8)),
#     go.Scatter3d(x=coordinates[3][0, :], y=coordinates[3][1, :], z=coordinates[3][2, :],
#                  mode='markers',
#                  marker=dict(size=8)),
#     go.Scatter3d(x=coordinates[4][0, :], y=coordinates[4][1, :], z=coordinates[4][2, :],
#                  mode='markers',
#                  marker=dict(size=8)),
#     go.Scatter3d(x=coordinates[5][0, :], y=coordinates[5][1, :], z=coordinates[5][2, :],
#                  mode='markers',
#                  marker=dict(size=8)),
#     go.Scatter3d(x=coordinates[6][0, :], y=coordinates[6][1, :], z=coordinates[6][2, :],
#                  mode='markers',
#                  marker=dict(size=8)),
#     go.Scatter3d(x=coordinates[7][0, :], y=coordinates[7][1, :], z=coordinates[7][2, :],
#                  mode='markers',
#                  marker=dict(size=8)),
#     # go.Scatter3d(x=coordinate_grid_flat[0, :], y=coordinate_grid_flat[1, :], z=coordinate_grid_flat[2, :],
#     #              mode='markers',
#     #              marker=dict(size=3, color='rgb(128, 128, 128)')),
#     go.Scatter3d(x=coordinate_grid_flat_inversed[0, :], y=coordinate_grid_flat_inversed[1, :],
#                  z=coordinate_grid_flat_inversed[2, :],
#                  mode='markers',
#                  marker=dict(size=3, color='rgb(156, 156, 156)')),
#     go.Scatter3d(x=points[0, :], y=points[1, :], z=points[2, :],
#                  mode='markers',
#                  marker=dict(size=12))
# ])
#
# fig.show()
# # (grid_3d * grid_density) / 5

# train_dataset = NerfSyntheticDataset(mode='train', rootdir="D:\\Implementations\\MVRNet", views='random',
#                                      downscale_factor=32, num_reference_views=5,
#                                      scenes=('lego'), load_specific_pose=0
#                                      # scenes=('mic')
#                                      )
# train_dataloader = DataLoader(train_dataset, shuffle=False)
# device = 'cpu'
# data = next(iter(train_dataloader))
# plt.imshow(data['rgb'].squeeze())
# plt.show()
# ray_sampler = RaySamplerSingleImage(data, device)
# ray_batch = ray_sampler.get_all()  # (N_rand=1, sample_mode='uniform', center_ratio=0.6)

# pts, z_vals = sample_along_camera_rays(ray_o=ray_batch['ray_o'][[0, 24, -24, -1]],
#                                        ray_d=ray_batch['ray_d'][[0, 24, -24, -1]],
#                                        depth_range=torch.tensor([[0.001, 6]]).cuda(),
#                                        N_samples=32, inv_uniform=False, det=True)
#
# pts_plane, z_vals = sample_along_camera_rays(ray_o=ray_batch['ray_o'],
#                                              ray_d=ray_batch['ray_d'],
#                                              depth_range=torch.tensor([[0.001, 4]]).cuda(),
#                                              N_samples=5, inv_uniform=False, det=True)
# pts = pts.reshape(-1, 3).cpu().numpy()
# pts_plane = pts_plane.reshape(-1, 3).cpu().numpy()
# grid_flat = coordinate_grid.reshape(3, -1)
#
# fig = go.Figure(data=[go.Scatter3d(x=grid_flat[0, :], y=grid_flat[1, :], z=grid_flat[2, :], mode='markers',
#                                    marker=dict(size=7, color='rgb(128, 128, 128)')),
#                       go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers',
#                                    marker=dict(size=3)),
#                       go.Scatter3d(x=pts_plane[:, 0], y=pts_plane[:, 1], z=pts_plane[:, 2], mode='markers',
#                                    marker=dict(size=3))
#                       ])
# fig.show()
