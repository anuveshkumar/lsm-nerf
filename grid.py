import torch


def inbound_mask(points):
    x_mask = (-1 <= points[0, :]) * (points[0, :] < 1)
    y_mask = (-1 <= points[1, :]) * (points[1, :] < 1)
    z_mask = (-1 <= points[2, :]) * (points[2, :] < 1)
    return x_mask * y_mask * z_mask


def meshgrid(grid_size):
    if type(grid_size) == int:
        grid_size = (grid_size,) * 3
    elif len(grid_size) != 3:
        grid_size = grid_size
    depth, height, width = grid_size
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


def trilineate(volume_feature, coordinate_grid_flat, points, grid_size=10, grid_scale=2.0):
    # print(output)
    device = volume_feature.device
    coordinate_grid_flat = coordinate_grid_flat.to(device)
    n_rays, n_samples = points.shape[:2]
    points = points.reshape(-1, 3).permute(1, 0)
    channels = volume_feature.squeeze(0).shape[0]  # C, H, W, D
    point_features = torch.zeros(channels, len(points[1])).to(device)
    feature_grid_flat = volume_feature.reshape(channels, -1)

    # what I used to do (The proper method)

    # move points to range [-1, 1] # corresponding to the grid. This is to align the points to the grid_indexing
    points = points / grid_scale  # scale 3d points to range [-1, 1]
    mask = inbound_mask(points)
    points = points[:, mask]
    # move the points to [0, 1] and the scale as per the grid
    points_scaled = (points + 1.0) * (grid_size - 1.0) / 2.0

    points_bottom = torch.floor(points_scaled).long()  # get the lower bottom indices
    points_top = points_bottom + 1  # get the top high indices

    bottom_coordinate = coordinate_grid_flat[:,
                        points_bottom[0, :] + points_bottom[1, :] * grid_size + points_bottom[2, :] * (
                                grid_size ** 2)].to(device)
    top_coordinate = coordinate_grid_flat[:,
                     points_top[0, :] + points_top[1, :] * grid_size + points_top[2, :] * (grid_size ** 2)].to(device)

    coordinates = []
    features = []

    for z in range(2):
        for y in range(2):
            for x in range(2):
                coordinates.append(coordinate_grid_flat[:,
                                   (points_bottom[0, :] + x) +
                                   (points_bottom[1, :] + y) * grid_size +
                                   (points_bottom[2, :] + z) * (grid_size ** 2)])
                features.append(feature_grid_flat[:,
                                (points_bottom[0, :] + x) +
                                (points_bottom[1, :] + y) * grid_size +
                                (points_bottom[2, :] + z) * (grid_size ** 2)])

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
    return point_features.permute(1, 0).reshape(n_rays, n_samples, -1)


if __name__ == "__main__":
    grid_size = 10
    grid_scale = 2.0
    coordinate_grid_flat = meshgrid(grid_size)
    feature_grid = torch.randn(1, 8, grid_size, grid_size, grid_size).requires_grad_(True)
    points = torch.FloatTensor(3, 4096).uniform_(-2.5, 2.5)
    point_features = trilineate(points=points, volume_feature=feature_grid, coordinate_grid_flat=coordinate_grid_flat,
                                grid_size=grid_size,
                                grid_scale=grid_scale)
