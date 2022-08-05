import random
import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append('../')
from datasets.data_utils import rectify_inplane_rotation, get_nearby_pose_ids
from PIL import Image

blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def get_intrinsics_from_hwf(h, w, focal, downscale_factor):
    h = h / downscale_factor
    w = w / downscale_factor
    focal = focal / downscale_factor
    return np.array([[focal, 0, 1.0 * w / 2, 0],
                     [0, focal, 1.0 * h / 2, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def read_cameras(pose_file, downscale_factor):
    basedir = os.path.dirname(pose_file)
    with open(pose_file, 'r') as fp:
        meta = json.load(fp)

    camera_angle_x = float(meta['camera_angle_x'])
    rgb_files = []
    c2w_mats = []

    img = imageio.imread(os.path.join(basedir, meta['frames'][0]['file_path'] + '.png'))
    H, W = img.shape[:2]
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    intrinsics = get_intrinsics_from_hwf(H, W, focal, downscale_factor)

    for i, frame in enumerate(meta['frames']):
        rgb_file = os.path.join(basedir, meta['frames'][i]['file_path'][2:] + '.png')
        rgb_files.append(rgb_file)
        c2w_opencv = np.array(frame['transform_matrix']) @ blender2opencv
        c2w_mats.append(c2w_opencv)  # x_down_y_down_z_cam_dir
    c2w_mats = np.array(c2w_mats)
    return rgb_files, np.array([intrinsics] * len(meta['frames'])), c2w_mats


class NerfSyntheticDataset(Dataset):
    def __init__(self, mode, rootdir,
                 # scenes = ('chair', 'drum', 'lego', 'hotdog', 'materials', 'mic', 'ship'),
                 scenes=(), downscale_factor=8,
                 views='random', num_reference_views=10,
                 load_specific_pose=None
                 ):
        self.views = views
        self.downscale_factor = downscale_factor
        self.folder_path = os.path.join(rootdir, 'data/nerf_synthetic/')
        self.rectify_inplace_rotation = False
        if mode == 'validation':
            mode = 'val'
        assert mode in ['train', 'val', 'test']
        self.mode = mode  # train / test / val
        self.batch_size = 1
        self.num_reference_views = num_reference_views

        all_scenes = ('chair', 'drum', 'lego', 'hotdog', 'mic', 'ship')
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes

        print("loading {} for {}".format(scenes, mode))
        self.rgb_files = []
        self.poses = []
        self.intrinsics = []

        for scene in scenes:
            self.scene_path = os.path.join(self.folder_path, scene)
            pose_file = os.path.join(self.scene_path, 'transforms_{}.json'.format(mode))
            rgb_files, intrinsics, poses = read_cameras(pose_file, self.downscale_factor)
            self.rgb_files.append(rgb_files)
            self.poses.append(poses)
            self.intrinsics.append(intrinsics)

        self.load_specific_pose = load_specific_pose

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_files = self.rgb_files[idx]
        poses = self.poses[idx]
        intrinsics = self.intrinsics[idx]

        # three ways to sample data
        # 1. sample n nearby images for IBRNet
        # 2. sample n random images for MultiViewStereo
        # 3. sample 3 nearby images for MVSNerf
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
        h, w = target_image.size
        target_image = np.array(target_image.resize((int(h // self.downscale_factor), int(w // self.downscale_factor))),
                                dtype=np.float32) / 255.
        target_image = target_image[..., [-1]] * target_image[..., :3] + 1 - target_image[..., [-1]]
        target_intrinsics = intrinsics[target_id]
        target_pose = poses[target_id]
        img_size = target_image.shape[:2]
        target_camera = np.concatenate((list(img_size), target_intrinsics.flatten(),
                                        target_pose.flatten())).astype(np.float32)

        reference_images = []
        reference_cameras = []

        for id in reference_ids:
            ref_img = Image.open(rgb_files[id])
            h, w = ref_img.size
            ref_img = np.array(ref_img.resize((int(h // self.downscale_factor), int(w // self.downscale_factor))),
                               dtype=np.float32) / 255.
            ref_img = ref_img[..., [-1]] * ref_img[..., :3] + 1 - ref_img[..., [-1]]
            ref_pose = poses[id]
            ref_intrinsics = intrinsics[id]
            if self.rectify_inplace_rotation:
                ref_pose, ref_img = rectify_inplane_rotation(ref_pose, target_pose, ref_img)

            reference_images.append(ref_img)
            img_size = ref_img.shape[:2]
            ref_camera = np.concatenate((list(img_size), ref_intrinsics.flatten(),
                                         ref_pose.flatten())).astype(np.float32)
            reference_cameras.append(ref_camera)

        reference_images = np.stack(reference_images, axis=0)
        reference_cameras = np.stack(reference_cameras, axis=0)

        near_depth = 2.
        far_depth = 6.

        depth_range = torch.tensor([near_depth, far_depth])

        return {'rgb': torch.from_numpy(target_image[..., :3]),
                'camera': torch.from_numpy(target_camera),
                'rgb_path': rgb_files[target_id],
                'src_rgbs': torch.from_numpy(reference_images[..., :3]),
                'src_cameras': torch.from_numpy(reference_cameras),
                'depth_range': depth_range
                }


if __name__ == "__main__":
    dataset = NerfSyntheticDataset(mode='train', downscale_factor=6.25, rootdir="D:\\Implementations\\MultiViewStereo",
                                   views='random')
    data = next(iter(dataset))
