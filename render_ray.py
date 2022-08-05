import torch
from collections import OrderedDict
import torch.nn.functional as F
from torch import searchsorted
import matplotlib.pyplot as plt
from sample_ray import parse_camera
from grid import trilineate


def normalize(pixel_locations, h, w):
    resize_factor = torch.tensor([w - 1., h - 1.]).to(pixel_locations.device)[None, None, :]
    normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.

    return normalized_pixel_locations


def sample_along_camera_rays(ray_o, ray_d, depth_range, N_samples, inv_uniform=False, det=False):
    """
    :param ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
    :param ray_d: homogeneous ray direction vectors in scene cooordinate system; tensor of shape [N_rays, 3]
    :param depth_range: [near_depth, far_depth]
    :param inv_uniform: if True, uniformly sampling inverse depth
    :param det: if True, will perform deterministic sampling
    :return: tensor of shape [N_rays, N_samples, 3]
    """
    # will sample inside [near_depth, far_depth]
    # assume the nearest possible depth is at least (min_ratio * depth)

    near_depth_value = depth_range[0, 0]
    far_depth_value = depth_range[0, 1]
    assert near_depth_value > 0 and far_depth_value > 0 and far_depth_value > near_depth_value

    near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])
    far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])

    if inv_uniform:
        start = 1. / near_depth
        step = (1. / far_depth - start) / (N_samples - 1)
        inv_z_vals = torch.stack([start + i * step for i in range(N_samples)], dim=1)  # [N_rays, N_samples]
        z_vals = 1. / inv_z_vals

    else:
        start = near_depth
        step = (far_depth - near_depth) / (N_samples - 1)
        z_vals = torch.stack([start + i * step for i in range(N_samples)], dim=1)

    if not det:
        # get intervals between samples
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
        # uniform samples in those intervals
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    ray_d = ray_d.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, 3]
    ray_o = ray_o.unsqueeze(1).repeat(1, N_samples, 1)
    pts = z_vals.unsqueeze(2) * ray_d + ray_o

    return pts, z_vals


def sample_pdf(bins, weights, N_samples, det=False):
    """
    :param bins: tensor of shape [N_rays, M+1], M is the number of bins
    :param weights: tensor of shape [N_rays, M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [N_rays, N_samples]
    """

    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # adding zero prod to the start of cdf
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=cdf.device)

    # Invert CDF
    # for a val in 0-1 in cdf find where it came from along the ray
    inds = searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    # use min and max coordinates for indices
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)  # (batch, N_importance_samples)
    inds_g = torch.stack([below, above], -1)  # (batch, N_importance_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[1]]  # [N_rand, N_samples, N_samples-1]

    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # find fraction of how much we moved in between the bins, by interpolating and normalizing cdf vals
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom

    # weights are computed on pts samples and bins are defined in between samples
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def raw2outputs(raw, z_vals, mask=None, white_bkgd=False):
    """
    :param raw: raw network output; tensor of shape [N_rays, N_samples, 4]
    :param z_vals: depth of point samples along rays; tensor of shape [N_rays, N_samples]
    :param mask: [N_rays, 3]
    :return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights':[N_rays,], 'depth_std':[N_rays,]}
    """
    rgb = torch.sigmoid(raw[:, :, :3])  # [N_rays, N_samples, 3]
    sigma = raw[:, :, 3]

    # note: we did not use the intervals here, because in practice different scenes from COLMAP can have
    # very different scales, and using interval can affect the model's generalization ability.
    # Therefore we don't use the intervals for both training and evaluation
    sigma2alpha = lambda sigma, dists, act_fn=F.relu: 1. - torch.exp(
        -act_fn(sigma)) #* dists)  # only thing that's different

    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat((dists, dists[:, -1:]), dim=-1)

    alpha = sigma2alpha(sigma, dists)

    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]  # [N_rays, N_samples - 1]
    T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

    # maths show weights, and summation of weight along a ray, are always inside [0, 1]
    weights = alpha * T  # [N_rays, N_samples]
    rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - torch.sum(weights, dim=-1, keepdim=True))

    if mask is not None:
        mask = mask.float().sum(
            dim=1) > 8  # should at least have 8 valid observation on the ray, otherwise don't consider it's loss

    depth_map = torch.sum(weights * z_vals, dim=-1)  # [N_rays,]

    ret = OrderedDict([('rgb', rgb_map),
                       ('depth', depth_map),
                       ('weights', weights),
                       ('mask', torch.tensor([]) if mask is None else mask),
                       ('alpha', alpha),
                       ('z_vals', z_vals)
                       ])

    return ret


def compute_projection(xyz, src_cameras):
    original_shape = xyz.shape[:3]
    xyz = xyz.reshape(-1, 3)
    num_views = len(src_cameras)
    src_intrinsics = src_cameras[:, 2:18].reshape(-1, 4, 4)
    src_poses = src_cameras[:, -16:].reshape(-1, 4, 4)
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)
    projections = src_intrinsics.bmm(torch.inverse(src_poses)) \
        .bmm(xyz_h.t()[None, ...].repeat(num_views, 1, 1))
    projections = projections.permute(0, 2, 1)
    pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)
    pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
    return pixel_locations.reshape((num_views,) + original_shape + (2,))


def get_pts_feats(volume_feature, pts, aabb):
    device = volume_feature.device
    pts /= aabb[1]  # scale points between -aabb to +aabb
    n_rays, n_samples = pts.shape[-3: -1]

    grid = pts.view(-1, 1, n_rays, n_samples, 3).to(device)
    features = F.grid_sample(volume_feature, grid, align_corners=True, mode='bilinear')[:, :, 0].permute(2, 3, 0,
                                                                                                         1).squeeze()
    return features


def render_rays(ray_batch, volume_renderer, volume_feature, grid_size, grid_scale, coordinate_grid_flat, N_samples,
                inv_uniform=False, det=False, N_importance=0):
    ret = {'outputs_coarse': None,
           'outputs_fine': None}
    ray_o = ray_batch['ray_o']
    ray_d = ray_batch['ray_d']
    pts, z_vals = sample_along_camera_rays(ray_o=ray_batch['ray_o'],
                                           ray_d=ray_batch['ray_d'],
                                           depth_range=ray_batch['depth_range'],
                                           N_samples=N_samples, inv_uniform=inv_uniform, det=det)
    # print(pts.max(), pts.min())

    if N_importance > 0:
        with torch.no_grad():
            # input_feat = get_pts_feats(volume_feature, pts, aabb)
            input_feat = trilineate(volume_feature=volume_feature, coordinate_grid_flat=coordinate_grid_flat,
                                    points=pts, grid_size=grid_size, grid_scale=grid_scale)
            raw = volume_renderer(pts, ray_batch['ray_d'], input_feat)
            result = raw2outputs(raw, z_vals)

    else:
        # input_feat = get_pts_feats(volume_feature, pts, aabb)
        input_feat = trilineate(volume_feature=volume_feature, coordinate_grid_flat=coordinate_grid_flat,
                                points=pts, grid_size=grid_size, grid_scale=grid_scale)
        raw = volume_renderer(pts, ray_batch['ray_d'], input_feat)
        result = raw2outputs(raw, z_vals)

    ret['outputs_coarse'] = result

    if N_importance > 0:
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, result['weights'][..., 1: -1], N_importance, det=det).detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        pts = ray_o[..., None, :] + ray_d[..., None, :] * z_vals[..., :, None]
        input_feat = trilineate(volume_feature=volume_feature, coordinate_grid_flat=coordinate_grid_flat,
                                points=pts, grid_size=grid_size, grid_scale=grid_scale)
        raw = volume_renderer(pts, ray_batch['ray_d'], input_feat)
        result = raw2outputs(raw, z_vals)
        ret['outputs_fine'] = result

    return ret
