import numpy as np
from scipy.optimize import leastsq
import torch


def fit_func(p, x, y):
    """ 数据拟合函数 """
    a, b, c = p
    return a * x + b * y + c


def residuals(p, x, y, z):
    """ 误差函数 """
    return z - fit_func(p, x, y)


def estimate_plane_with_leastsq(pts):
    """ 根据最小二乘拟合出平面参数 """
    p0 = [1, 0, 1]
    np_pts = np.array(pts)
    plsq = leastsq(residuals, p0, args=(np_pts[:, 0], np_pts[:, 1], np_pts[:, 2]))
    return plsq[0]


def get_proper_plane_params(p, pts):
    """ 根据拟合的平面的参数，得到实际显示的最佳的平面 """
    assert isinstance(pts, list), r'输入的数据类型必须依赖 list'

    if np.linalg.norm(p) < 1e-10:
        print(r'plsq 的 norm 值为 0 {}'.format(p))
    plane_normal = p / np.linalg.norm(p)

    return plane_normal


def normal_3d_encoding(xyz):

    n_batch, dim, n_height, n_width = xyz.shape
    padding_xyz = torch.zeros(n_batch, dim, n_height + 2, n_width + 2)
    padding_xyz[:, :, 1:-1, 1:-1] = xyz

    normal = torch.zeros(n_batch, 3, n_height, n_width)

    for n in range(n_batch):
        im = padding_xyz[n, :, :, :]
        # print(im.shape)
        for h in range(n_height):
            for w in range(n_width):
                pts = im[:, h:h + 3, w:w + 3].reshape(3, -1).transpose(1, 0)
                pts = pts.tolist()
                p = estimate_plane_with_leastsq(pts)
                normal_p = get_proper_plane_params(p, pts)
                normal[n, :, h, w] = torch.tensor(normal_p)
    return normal

