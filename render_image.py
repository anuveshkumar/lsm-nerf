import torch
from collections import OrderedDict
from render_ray import render_rays
import matplotlib.pyplot as plt


def render_single_image(ray_sampler, ray_batch, volume_renderer, volume_feature, chunk_size, N_samples,
                        grid_size, grid_scale, coordinate_grid_flat, N_importance=0,
                        inv_uniform=False, render_stride=1, det=False, aabb=(-2, 2)):
    all_ret = OrderedDict([('outputs_coarse', OrderedDict()),
                           ('outputs_fine', OrderedDict())])
    N_rays = ray_batch['ray_o'].shape[0]

    for i in range(0, N_rays, chunk_size):
        print(f"{i} / {N_rays}")
        chunk = OrderedDict()
        for k in ray_batch:
            if k in ['camera', 'depth_range', 'src_rgbs', 'src_cameras']:
                chunk[k] = ray_batch[k]
            elif ray_batch[k] is not None:
                chunk[k] = ray_batch[k][i: i + chunk_size]
            else:
                chunk[k] = None

        # ret = render_rays(chunk, volume_renderer=volume_renderer, volume_feature=volume_feature, N_samples=N_samples,
        #                   det=det, inv_uniform=inv_uniform, N_importance=N_importance, aabb=aabb)

        ret = render_rays(chunk, volume_renderer=volume_renderer, volume_feature=volume_feature,
                          grid_size=grid_size, grid_scale=grid_scale,
                          coordinate_grid_flat=coordinate_grid_flat,
                          N_samples=N_samples, N_importance=N_importance)

        # handle both coarse and fine outputs
        # cache chunk results on cpu
        if i == 0:
            for k in ret['outputs_coarse']:
                all_ret['outputs_coarse'][k] = []

            if ret['outputs_fine'] is None:
                all_ret['outputs_fine'] = None

            else:
                for k in ret['outputs_fine']:
                    all_ret['outputs_fine'][k] = []

        for k in ret['outputs_coarse']:
            all_ret['outputs_coarse'][k].append(ret['outputs_coarse'][k].cpu())

        if ret['outputs_fine'] is not None:
            for k in ret['outputs_fine']:
                all_ret['outputs_fine'][k].append(ret['outputs_fine'][k].cpu())

    rgb_strided = torch.ones(ray_sampler.H, ray_sampler.W, 3)[::render_stride, ::render_stride, :]
    # merge chunks result and reshape
    for k in all_ret['outputs_coarse']:
        if k == 'random_sigma':
            continue
        tmp = torch.cat(all_ret['outputs_coarse'][k], dim=0).reshape((rgb_strided.shape[0], rgb_strided.shape[1], -1))

        all_ret['outputs_coarse'][k] = tmp.squeeze()

    # all_ret['outputs_coarse']['rgb'][all_ret['outputs_coarse']['mask'] == 0] = 1.

    if all_ret['outputs_fine'] is not None:
        for k in all_ret['outputs_fine']:
            if k == 'random_sigma':
                continue
            tmp = torch.cat(all_ret['outputs_fine'][k], dim=0).reshape((rgb_strided.shape[0], rgb_strided.shape[1], -1))
            all_ret['outputs_fine'][k] = tmp.squeeze()

    return all_ret
