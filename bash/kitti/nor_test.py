import torch
import time


def estimate_normal(xyz, device=torch.device('cuda')):
    n_batch, dim, n_height, n_width = xyz.shape
    padding_xyz = torch.zeros(n_batch, dim, n_height + 2, n_width + 2).to(device)
    padding_xyz[:, :, 1:-1, 1:-1] = xyz.clone()

    neighbor_xyz = [padding_xyz[:, :, :-2, :-2], padding_xyz[:, :, :-2, 1:-1], padding_xyz[:, :, :-2, 2:],
                    padding_xyz[:, :, 1:-1, :-2], xyz, padding_xyz[:, :, 1:-1, 2:],
                    padding_xyz[:, :, 2:, :- 2], padding_xyz[:, :, 2:, 1:-1], padding_xyz[:, :, 2:, 2:]]

    neighbor = torch.cat(neighbor_xyz, dim=1).permute(0, 2, 3, 1)
    neighbor = neighbor.reshape(n_batch, n_height, n_width, 9, 3)

    A = neighbor.clone()
    A[:, :, :, :, -1] = 1
    B = neighbor[:, :, :, :, -1].clone().reshape(n_batch, n_height, n_width, 9, 1)
    A_T = A.transpose(4, 3)
    invers = torch.linalg.pinv(torch.matmul(A_T.clone(), A.clone()))

    X = torch.matmul(torch.matmul(invers.clone(), A_T.clone()), B.clone()).squeeze(dim=4)
    f_norm = torch.linalg.norm(X.clone(), dim=3)
    f_norm_new = torch.stack((f_norm, f_norm, f_norm), dim=-1)
    X[:, :, :, -1] = -1
    nor = torch.div(X.clone(), f_norm_new.clone()).permute(0, 3, 1, 2)
    print(nor.shape)
    return nor


def estimate_normal_leastsq(xyz, device=torch.device('cuda')):
    n_batch, dim, n_height, n_width = xyz.shape
    padding_xyz = torch.zeros(n_batch, dim, n_height + 2, n_width + 2).to(device)
    padding_xyz[:, :, 1:-1, 1:-1] = xyz.clone()

    neighbor_xyz = [padding_xyz[:, :, :-2, :-2], padding_xyz[:, :, :-2, 1:-1], padding_xyz[:, :, :-2, 2:],
                    padding_xyz[:, :, 1:-1, :-2], xyz, padding_xyz[:, :, 1:-1, 2:],
                    padding_xyz[:, :, 2:, :- 2], padding_xyz[:, :, 2:, 1:-1], padding_xyz[:, :, 2:, 2:]]

    neighbor = torch.cat(neighbor_xyz, dim=1).permute(0, 2, 3, 1)
    neighbor = neighbor.reshape(n_batch, n_height, n_width, 9, 3)
    a11 = torch.sum(neighbor[:, :, :, :, 0] * neighbor[:, :, :, :, 0], dim=3)
    a12 = torch.sum(neighbor[:, :, :, :, 0] * neighbor[:, :, :, :, 1], dim=3)
    a13 = torch.sum(neighbor[:, :, :, :, 0], dim=3)
    a21 = torch.sum(neighbor[:, :, :, :, 0] * neighbor[:, :, :, :, 1], dim=3)
    a22 = torch.sum(neighbor[:, :, :, :, 1] * neighbor[:, :, :, :, 1], dim=3)
    a23 = torch.sum(neighbor[:, :, :, :, 1], dim=3)
    a31 = a13
    a32 = a12
    # print(a32.shape, a11.shape)
    n = torch.ones_like(a31) * 9

    b1 = torch.sum(neighbor[:, :, :, :, 2] * neighbor[:, :, :, :, 0], dim=3)
    b2 = torch.sum(neighbor[:, :, :, :, 2] * neighbor[:, :, :, :, 1], dim=3)
    b3 = torch.sum(neighbor[:, :, :, :, 2], dim=3)

    a = torch.stack([a11, a12, a13, a21, a22, a23, a31, a32, n], dim=3)\
        .reshape(n_batch, n_height, n_width, 3, 3)

    b = torch.stack([b1, b2, b3], dim=3).reshape(n_batch, n_height, n_width, 3, 1)

    x = torch.matmul(torch.linalg.pinv(a), a)
    nor = torch.matmul(x, b).reshape(n_batch, n_height, n_width, 3).permute(0, 3, 1, 2)

    print('new', nor.shape)




if __name__ == '__main__':
    device = torch.device('cuda')
    xyz = torch.randn(4, 3, 160, 384).to(device)
    # estimate_normal(xyz)
    estimate_normal_leastsq(xyz)
