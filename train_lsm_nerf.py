import statistics

import torch
from models.lsm_volume_encoder import LSM
from models.volume_renderer import Nerf
from datasets.nerf_synthetic import NerfSyntheticDataset
from datasets.llff import LLFFDataset
from torch.utils.data import DataLoader
from criterion import Criterion
from sample_ray import RaySamplerSingleImage
from render_ray import render_rays
from grid import meshgrid

## Parameters
# train
EPOCHS = 20000
TEST_EVERY = 100
SAVE_EVERY = 100
RESUME_TRAINING = 6700

# Vol_Encoder
grid_scale = 2.5
grid_size = 48  # should be divisible by 16

# Rays
n_rays = 1024
n_samples = 64
n_importance = 0

device = "cuda:0"

# train_dataset = NerfSyntheticDataset(mode='train', rootdir="D:\\Implementations\\MVRNet", views='random',
#                                      downscale_factor=6.25, num_reference_views=5,
#                                      scenes=('lego', 'drum', 'ship')
#                                      # scenes=('mic')
#                                      )
train_dataset = LLFFDataset(mode='test', rootdir="D:\\Implementations\\MultiViewStereo", downscale_factor=3,
                            views='random', num_reference_views=10)
train_dataloader = DataLoader(train_dataset, shuffle=True)

# test_dataset = NerfSyntheticDataset(mode='test', rootdir="D:\\Implementations\\MVRNet", views='random',
#                                     downscale_factor=6.25, num_reference_views=5, scenes=('mic'))
test_dataset = LLFFDataset(mode='test', rootdir="D:\\Implementations\\MultiViewStereo", downscale_factor=3,
                           views='nearby', num_reference_views=5)
test_dataloader = DataLoader(test_dataset, shuffle=True)

vol_encoder = LSM(grid_scale=grid_scale, grid_size=grid_size, device=device, use_fusion=False)
vol_renderer = Nerf(device=device)

optimizer = torch.optim.Adam(list(vol_encoder.parameters()) + list(vol_renderer.parameters()), lr=5e-4)
criterion = Criterion()

mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))
epoch = 0
coordinate_grid_flat = meshgrid(grid_size=grid_size)
if RESUME_TRAINING:
    dicts = torch.load(f"checkpoints/{RESUME_TRAINING}.pth")
    vol_encoder.load_state_dict(dicts['vol_encoder_weights'])
    vol_renderer.load_state_dict(dicts['vol_renderer_weights'])
    grid_size = dicts['grid_size']
    aabb = dicts['grid_scale']
    optimizer.load_state_dict(dicts['optimizer_weights'])
    epoch = dicts['epoch']

for epoch in range(epoch, EPOCHS):
    if epoch % TEST_EVERY == 0:
        test_data = next(iter(test_dataloader))
        ray_sampler_test = RaySamplerSingleImage(test_data, device)
        ray_batch_test = ray_sampler_test.random_sample(N_rand=n_rays, sample_mode='uniform')
        with torch.no_grad():
            vol_feature = vol_encoder(ray_batch_test)
            test_outputs = render_rays(ray_batch_test, volume_renderer=vol_renderer, volume_feature=vol_feature,
                                       grid_size=grid_size, grid_scale=grid_scale,
                                       coordinate_grid_flat=coordinate_grid_flat,
                                       N_samples=n_samples, N_importance=n_importance)
            if test_outputs['outputs_fine'] is not None:
                test_output = test_outputs['outputs_fine']
            else:
                test_output = test_outputs['outputs_coarse']
            test_loss = criterion(test_output, ray_batch_test)
            print("_____________________________________")
            print(f"test_loss: {test_loss.item()}")
            print("_____________________________________")

    if epoch % SAVE_EVERY == 0:
        save_dict = {'vol_encoder_weights': vol_encoder.state_dict(),
                     'vol_renderer_weights': vol_renderer.state_dict(),
                     'optimizer_weights': optimizer.state_dict(),
                     'grid_scale': grid_scale,
                     'grid_size': grid_size,
                     'epoch': epoch}
        torch.save(save_dict, f"checkpoints/{epoch}.pth")
    avg_loss = []
    avg_psnr = []
    for scene_index, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        ray_sampler = RaySamplerSingleImage(data, device)
        ray_batch = ray_sampler.random_sample(N_rand=n_rays, sample_mode='uniform', center_ratio=0.6)
        vol_feature = vol_encoder(ray_batch)

        outputs = render_rays(ray_batch, volume_renderer=vol_renderer, volume_feature=vol_feature,
                              grid_size=grid_size, grid_scale=grid_scale, coordinate_grid_flat=coordinate_grid_flat,
                              N_samples=n_samples, N_importance=n_importance)
        if outputs['outputs_fine'] is not None:
            output = outputs['outputs_fine']
        else:
            output = outputs['outputs_coarse']
        loss = criterion(output, ray_batch)
        psnr = mse2psnr(loss)
        avg_loss.append(loss.item())
        avg_psnr.append(psnr.item())
        loss.backward()
        optimizer.step()  # vol_encoder.feature_network.conv0[0].conv.weight.grad
    print(f"epoch: {epoch} loss: {statistics.mean(avg_loss)}, psnr: {statistics.mean(avg_psnr)}")
