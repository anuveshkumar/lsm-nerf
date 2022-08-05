import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
import random
from PIL import Image
import matplotlib.pyplot as plt
from datasets.data_utils import get_nearby_pose_ids
from torchvision import transforms as T
import math

sys.path.append('../')
from datasets.llff_data_utils import load_llff_data, batch_parse_llff_poses
import torch.nn.functional as F


class LLFFDataset(Dataset):
    def __init__(self, mode, rootdir, downscale_factor, views='random',
                 num_reference_views=10, load_specific_pose=None):

        base_dir = os.path.join(rootdir, f'data\\real_world_360\\{mode}')
        self.num_reference_views = num_reference_views
        self.mode = mode
        scenes = os.listdir(base_dir)
        self.downscale_factor = downscale_factor
        self.views = views

        self.rgb_files = []
        self.poses = []
        self.intrinsics = []
        self.near_far = []
        for i, scene in enumerate(scenes):
            scene_path = os.path.join(base_dir, scene)
            _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(scene_path, load_imgs=False,
                                                                            factor=downscale_factor, spherify=True)
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)

            self.near_far.append([near_depth, far_depth])
            self.rgb_files.append(rgb_files)
            self.intrinsics.append(intrinsics)
            self.poses.append(c2w_mats)

        self.src_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        self.load_specific_pose = load_specific_pose

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_files = self.rgb_files[idx]
        poses = self.poses[idx]
        intrinsics = self.intrinsics[idx]

        if self.views == 'nearby':
            if self.load_specific_pose is not None:
                target_id = self.load_specific_pose
            else:
                target_id = random.sample(range(len(rgb_files)), 1)[0]
            reference_ids = get_nearby_pose_ids(poses[target_id], ref_poses=poses, num_select=self.num_reference_views,
                                                tar_id=target_id)
        if self.views == 'random':
            sampling_idx = random.sample(range(len(rgb_files)), self.num_reference_views + 1)
            if self.load_specific_pose is not None:
                target_id = self.load_specific_pose
            else:
                target_id = random.sample(sampling_idx, 1).pop()
            try:
                sampling_idx.remove(target_id)
            except:
                sampling_idx = sampling_idx[:-1]
            reference_ids = sampling_idx

        target_image = Image.open(rgb_files[target_id])
        target_image = self.pad_image(np.array(target_image, dtype=np.float32) / 255.)
        # target_image = self.src_transform(target_image).permute(1, 2, 0).numpy()
        target_intrinsics = intrinsics[target_id]
        target_pose = poses[target_id]
        img_size = target_image.shape[:2]
        target_camera = np.concatenate((list(img_size), target_intrinsics.flatten(),
                                        target_pose.flatten())).astype(np.float32)

        src_rgbs = []
        src_cameras = []

        for id in reference_ids:
            src_img = Image.open(rgb_files[id])
            src_img = self.pad_image(np.array(src_img, dtype=np.float32) / 255.)
            # src_img = self.src_transform(src_img).permute(1, 2, 0).numpy()
            ref_pose = poses[id]
            ref_intrinsics = intrinsics[id]

            src_rgbs.append(src_img)
            img_size = src_img.shape[:2]  # height, width
            ref_camera = np.concatenate((list(img_size), ref_intrinsics.flatten(),
                                         ref_pose.flatten())).astype(np.float32)
            src_cameras.append(ref_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        depth_range = torch.tensor(self.near_far[idx])

        return {'rgb': torch.from_numpy(target_image[..., :3]),
                'camera': torch.from_numpy(target_camera),
                'rgb_path': rgb_files[target_id],
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range
                }

    def pad_image(self, image):
        pad_h, pad_w = 0, 0
        if image.shape[0] % 16 != 0:
            multiple = math.ceil(image.shape[0] / 16)
            pad_h = (16 * multiple - image.shape[0]) // 2
        if image.shape[1] % 16 != 0:
            multiple = math.ceil(image.shape[1] / 16)
            pad_w = (16 * multiple - image.shape[1]) // 2

        if pad_h > 0 or pad_w > 0:
            image = F.pad(torch.from_numpy(image).permute(2, 0, 1), (pad_w + 1, pad_w, pad_h, pad_h), "constant",
                          0).permute(1, 2, 0).numpy()

        return image


if __name__ == "__main__":
    train_dataset = LLFFDataset(mode='train', rootdir="D:\\Implementations\\MultiViewStereo", downscale_factor=8,
                                views='random', num_reference_views=10)
    test_dataset = LLFFDataset(mode='test', rootdir="D:\\Implementations\\MultiViewStereo", downscale_factor=8,
                               views='nearby')
    data = next(iter(train_dataset))
