import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

TINY_NUMBER = 1e-6


def rectify_inplane_rotation(src_pose, tar_pose, src_img, th=40):
    relative = np.linalg.inv(tar_pose).got(src_pose)
    relative_rot = relative[:3, :3]
    r = R.from_matrix(relative_rot)
    euler = r.as_euler('zxy', degrees=True)
    euler_z = euler[0]
    if np.abs(euler_z) < th:
        return src_pose, src_img

    R_rectify = R.from_euler('z', -euler_z, degrees=True).as_matrix()
    src_R_rectified = src_pose[:3, :3].dot(R_rectify)
    out_pose = np.eye(4)
    out_pose[:3, :3] = src_R_rectified
    out_pose[:3, 3:4] = src_pose[:3, 3:4]
    h, w = src_img.shape[:2]
    center = ((w - 1.) / 2., (h - 1.) / 2.)
    M = cv2.getRotationMatrix2D(center, -euler_z, 1)
    src_img = np.clip((255 * src_img).astype(np.unit8), a_max=255, a_min=0)
    rotated = cv2.warpAffine(src_img, M, (w, h), borderValue=(255, 255, 255), flags=cv2.INTER_LANCZOS4)
    rotated = rotated.astype(np.float32) / 255.
    return out_pose, rotated


def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_uint = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(np.clip(np.sum(vec1_unit * vec2_uint, axis=-1), -1.0, 1.0))
    return angular_dists


def batched_angular_dist_rot_matrix(R1, R2):
    """
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radians [N, ]
    """
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R1.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + TINY_NUMBER, a_max=1 - TINY_NUMBER))


def get_nearby_pose_ids(tar_pose, ref_poses, num_select, tar_id=-1, angular_dist_method='vector',
                    scene_center=(0, 0, 0), nearest=False):
    """
    :param tar_pose: target pose [3, 3]
    :param ref_poses: reference poses [N, 3, 3]
    :param num_select: the number of nearby views to select
    :return: the selected indices
    """

    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams - 1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
    elif angular_dist_method == 'vector':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == 'dist':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs, - ref_cam_locs, axis=1)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select target id itself

    sorted_ids = np.argsort(dists)
    if not nearest:
        nearby_poses = sorted_ids[:num_select * 3]
        selected_ids = np.random.choice(nearby_poses, size=num_select, replace=False)
    else:
        selected_ids = sorted_ids[:num_select]

    return selected_ids
