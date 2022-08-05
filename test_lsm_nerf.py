import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datasets.nerf_synthetic import NerfSyntheticDataset
from datasets.llff import LLFFDataset
from grid import meshgrid
from models.lsm_volume_encoder import LSM
from models.volume_renderer import Nerf
from render_image import render_single_image
from sample_ray import RaySamplerSingleImage

# Rays
n_rays = 1024
n_samples = 64
n_importance = 0

device = "cuda:0"
# dataset = NerfSyntheticDataset(mode='train', rootdir="D:\\Implementations\\MVRNet", views='random',
#                                downscale_factor=6.25, num_reference_views=8, scenes=('hotdog'), load_specific_pose=0)
dataset = LLFFDataset(mode='test', rootdir="D:\\Implementations\\MultiViewStereo", downscale_factor=3,
                            views='random', num_reference_views=10)
dataloader = DataLoader(dataset, shuffle=False)
dicts = torch.load(f"checkpoints/7000.pth")
grid_size = dicts['grid_size']
grid_scale = dicts['grid_scale']

vol_encoder = LSM(grid_scale=grid_scale, grid_size=grid_size, device=device, use_fusion=False)
vol_renderer = Nerf(device=device)
vol_encoder.load_state_dict(dicts['vol_encoder_weights'])
vol_renderer.load_state_dict(dicts['vol_renderer_weights'])

data = next(iter(dataloader))
ray_sampler = RaySamplerSingleImage(data, device)
ray_batch = ray_sampler.get_all()
gt_rgb = ray_batch['rgb'].reshape(ray_sampler.H, ray_sampler.W, 3)

coordinate_grid_flat = meshgrid(grid_size=grid_size)
with torch.no_grad():
    vol_feature = vol_encoder(ray_batch)
    output = \
        render_single_image(ray_sampler, ray_batch, vol_renderer, vol_feature, chunk_size=1000, N_samples=64,
                            grid_size=grid_size, grid_scale=grid_scale, coordinate_grid_flat=coordinate_grid_flat,
                            N_importance=0)

rows = 1
columns = 3
plt.subplot(rows, columns, 1)
plt.imshow(output['outputs_coarse']['rgb'])
# plt.subplot(rows, columns, 2)
# plt.imshow(output['outputs_fi ']['rgb'])
plt.subplot(rows, columns, 2)
plt.imshow(output['outputs_coarse']['depth'])
plt.subplot(rows, columns, 3)
plt.imshow(gt_rgb.cpu())
plt.show()
