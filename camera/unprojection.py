import torch
from grid import meshgrid


def rot_to_quat(R):
    q = torch.ones(4).cuda()

    R00 = R[0, 0]
    R01 = R[0, 1]
    R02 = R[0, 2]
    R10 = R[1, 0]
    R11 = R[1, 1]
    R12 = R[1, 2]
    R20 = R[2, 0]
    R21 = R[2, 1]
    R22 = R[2, 2]

    q[0] = torch.sqrt(1.0 + R00 + R11 + R22) / 2
    q[1] = (R21 - R12) / (4 * q[0])
    q[2] = (R02 - R20) / (4 * q[0])
    q[3] = (R10 - R01) / (4 * q[0])
    return q


def unproj_grid(grid_scale, grid_size, feats, camera, device):
    # todo: use mask to fill the voxels projections that lie outside the feature plane with zeros #done

    img_shape = camera[:2]
    K = camera[2:18].reshape(4, 4)[:3, :3]  # (3 x 3)
    R = torch.inverse(camera[-16:].reshape(4, 4))[:3, :]  # (3 x 4) the original matrix is c2w and we want w2c
    KR = torch.mm(K, R)

    feats = feats.permute(1, 2, 0)  # convert to H, W, C
    fh, fw = feats.size()[:2]
    rsz_h = float(fh) / img_shape[0]
    rsz_w = float(fw) / img_shape[1]

    # Create voxel grid
    rs_grid = meshgrid(grid_size) * grid_scale
    rs_grid = torch.cat((rs_grid, torch.ones(1, rs_grid.shape[1])), dim=0).to(device)

    # Project grid
    im_p = torch.mm(KR, rs_grid)
    im_x, im_y, im_z = im_p[0, :], im_p[1, :], im_p[2, :]
    mask = im_z > 0 # if voxels are behind the camera
    # (im_x / im_z) gives the point in the original image scale,
    # which we then multiply the factor of feature res and image res
    im_x = (im_x / im_z) * rsz_h
    im_y = (im_y / im_z) * rsz_w

    # if voxel projection don't lie in the image/feature plane
    mask = mask * ((im_x >= 0) * (im_x < fh - 2))
    mask = mask * ((im_y >= 0) * (im_y < fw - 2))
    # Bilinear interpolation
    # now only those voxels and their projections are considered lie on the image plane and are in front of the camera
    im_x = im_x[mask]
    im_y = im_y[mask]
    # x, lower and upper index
    im_x0 = torch.floor(im_x)
    im_x1 = im_x0 + 1
    # y, lower and upper index
    im_y0 = torch.floor(im_y)
    im_y1 = im_y0 + 1

    wa = (im_x1 - im_x) * (im_y1 - im_y)
    wb = (im_x1 - im_x) * (im_y - im_y0)
    wc = (im_x - im_x0) * (im_y1 - im_y)
    wd = (im_x - im_x0) * (im_y - im_y0)

    img_a = feats[im_x1.long(), im_y1.long()]
    img_b = feats[im_x1.long(), im_y0.long()]
    img_c = feats[im_x0.long(), im_y1.long()]
    img_d = feats[im_x0.long(), im_y0.long()]

    bilinear = wa.unsqueeze(1) * img_a + wb.unsqueeze(1) * img_b + wc.unsqueeze(1) * img_c + wd.unsqueeze(1) * img_d
    bilinear_padded = torch.zeros(mask.shape[0], feats.shape[-1]).to(bilinear.device)
    bilinear_padded[mask, :] = bilinear
    # quaternion_translation = torch.cat((rot_to_quat(R[:3, :3]), R[:3, 3])).repeat(bilinear.shape[0], 1)
    # bilinear = torch.cat((bilinear, quaternion_translation), dim=1)
    return bilinear_padded
