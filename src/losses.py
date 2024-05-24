'''
Author: Alex Wong <alexw@cs.ucla.edu>

If you use this code, please cite the following paper:

A. Wong, and S. Soatto. Unsupervised Depth Completion with Calibrated Backprojection Layers.
https://arxiv.org/pdf/2108.10531.pdf

@inproceedings{wong2021unsupervised,
  title={Unsupervised Depth Completion with Calibrated Backprojection Layers},
  author={Wong, Alex and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12747--12756},
  year={2021}
}
'''
from cgi import print_arguments
import torch
import open3d
import numpy as np
import net_utils


EPSILON = 1e-8


def color_consistency_loss_func(src, tgt, w, use_pytorch_impl=False):
    '''
    Computes the color consistency loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
        w : torch.Tensor[float32]
            N x 1 x H x W weights
    Returns:
        torch.Tensor[float32] : mean absolute difference between source and target images
    '''

    loss = torch.sum(w * torch.abs(tgt - src), dim=[1, 2, 3])

    return torch.mean(loss / torch.sum(w, dim=[1, 2, 3]))

def structural_consistency_loss_func(src, tgt, w):
    '''
    Computes the structural consistency loss using SSIM

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
        w : torch.Tensor[float32]
            N x 3 x H x W weights
    Returns:
        torch.Tensor[float32] : mean 1 - SSIM scores between source and target images
    '''

    scores = ssim(src, tgt)
    scores = torch.nn.functional.interpolate(scores, size=w.shape[2:4], mode='nearest')
    loss = torch.sum(w * scores, dim=[1, 2, 3])

    return torch.mean(loss / torch.sum(w, dim=[1, 2, 3]))

def sparse_depth_consistency_loss_func(src, tgt, w):
    '''
    Computes the sparse depth consistency loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 1 x H x W source depth
        tgt : torch.Tensor[float32]
            N x 1 x H x W target depth
        w : torch.Tensor[float32]
            N x 1 x H x W weights
    Returns:
        torch.Tensor[float32] : mean absolute difference between source and target depth
    '''

    delta = torch.abs(tgt - src)
    loss = torch.sum(w * delta, dim=[1, 2, 3])

    return torch.mean(loss / torch.sum(w, dim=[1, 2, 3]))

def smoothness_loss_func(predict, image):
    '''
    Computes the local smoothness loss

    Arg(s):
        predict : torch.Tensor[float32]
            N x 1 x H x W predictions
        image : torch.Tensor[float32]
            N x 3 x H x W RGB image
    Returns:
        torch.Tensor[float32] : mean SSIM distance between source and target images
    '''

    predict_dy, predict_dx = gradient_yx(predict)
    image_dy, image_dx = gradient_yx(image)

    # Create edge awareness weights
    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
    smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))
    return smoothness_x + smoothness_y

def normal_loss_func_kdtree(points_3dim, normal):
    batch, _, hw = points_3dim.shpae
    loss = 0
    for i in range(batch):
        point = points_3dim[ i, :, :].view(3, -1)
        normal_i = torch.transpose(normal[ i, :, :].view(3, -1))
        depth_3d = torch.transpose(point)
        normal_from_dense_depth = surface_normal(depth_3d)
        delta = torch.abs(normal_i - normal_from_dense_depth)
        loss = loss + delta

    return loss

def normal_loss_func(image, depth, normal, intrinsics):
    n_batch, _, n_height, n_width = depth.shape

    # Create homogeneous coordinates [u, v, 1]
    # xy_h.shape = N x 3 x H x W 
    # 1 下临近点， 2 右临近点
    xy_h_origanal = net_utils.meshgrid(n_batch, n_height, n_width, device=depth.device, homogeneous=True)
    xy_h = xy_h_origanal[ : , : , :-1 , :-1].reshape(n_batch, 3, -1)

    x_p = torch.matmul(torch.inverse(intrinsics), xy_h).permute(0,2,1).contiguous() 

    # q1 下临近点， q2 右临近点
    x_q1 = torch.matmul(torch.inverse(intrinsics), (xy_h_origanal[:, :, 1: , :-1].reshape(n_batch, 3, -1))).permute(0,2,1)
    x_q2 = torch.matmul(torch.inverse(intrinsics), (xy_h_origanal[: , :, :-1, 1:].reshape(n_batch, 3, -1))).permute(0,2,1)

    normal_p = normal[:, :, 0: n_height-1, 0: n_width-1].contiguous().view(n_batch, 3, -1)
    normal_p = normal_p.permute(0,2,1)   # N * hw * 3
    normal_p = normal_p.view(n_batch, -1, 1, 3)


    x_p = x_p.view(n_batch, -1, 3, 1)
    x_q1 = x_q1.view(n_batch, -1, 3, 1)
    x_q2 = x_q2.view(n_batch, -1, 3, 1)

    # c_pq1 = < N_p, X_q1 >    shape: N x (H-1)*(W-1) x 1 x 1  to  N x 1 x H-1 x W-1
    c_pq1 = torch.matmul(normal_p, x_q1).view(n_batch, 1, n_height-1, n_width-1)
    c_pq2 = torch.matmul(normal_p, x_q2).view(n_batch, 1, n_height-1, n_width-1)
    c_pp = torch.matmul(normal_p, x_p).view(n_batch, 1, n_height-1, n_width-1)

    # depth的倒数
    D_inv_p = 1.0/ depth[ :, :, 0: n_height-1, 0: n_width-1]
    D_inv_q1 = 1.0/ depth[ :, :, 1: n_height, 0: n_width-1]
    D_inv_q2 = 1.0/ depth[ :, :, 1: n_height, 0: n_width-1]

    # loss = D_inv_p*c_pq1 + D_inv_p*c_pq2 + D_inv_q1*c_pp + D_inv_q2*c_pp
    loss_1 = D_inv_p*c_pq1 - D_inv_q1*c_pp
    loss_2 = D_inv_p*c_pq2 - D_inv_q2*c_pp
    loss = 0.5*torch.abs(loss_1) + 0.5*torch.abs(loss_2)

    image_dx = image[:, :, :-1, :-1] - image[:, :, :-1, 1:]
    image_dy = image[:, :, :-1, :-1] - image[:, :, 1:, :-1]

    t = torch.cat([image_dx, image_dy], dim=1)

    G_p = torch.exp(torch.norm(t, dim=1))

    loss = loss * G_p

    # loss = torch.sum(loss, dim=[1, 2, 3])
    loss = torch.sum(loss)

    return loss

def surface_normal(depth_3d):

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(depth_3d)

    pcd.estimate_normals(
        search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    normals = np.array(pcd.normals)

    return normals
'''
Helper functions for constructing loss functions
'''
def gradient_yx(T):
    '''
    Computes gradients in the y and x directions

    Arg(s):
        T : torch.Tensor[float32]
            N x C x H x W tensor
    Returns:
        torch.Tensor[float32] : gradients in y direction
        torch.Tensor[float32] : gradients in x direction
    '''

    dx = T[:, :, :, :-1] - T[:, :, :, 1:]
    dy = T[:, :, :-1, :] - T[:, :, 1:, :]
    return dy, dx

def ssim(x, y):
    '''
    Computes Structural Similarity Index distance between two images

    Arg(s):
        x : torch.Tensor[float32]
            N x 3 x H x W RGB image
        y : torch.Tensor[float32]
            N x 3 x H x W RGB image
    Returns:
        torch.Tensor[float32] : SSIM distance between two images
    '''

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = torch.nn.AvgPool2d(3, 1)(x)
    mu_y = torch.nn.AvgPool2d(3, 1)(y)
    mu_xy = mu_x * mu_y
    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2

    sigma_x = torch.nn.AvgPool2d(3, 1)(x ** 2) - mu_xx
    sigma_y = torch.nn.AvgPool2d(3, 1)(y ** 2) - mu_yy
    sigma_xy = torch.nn.AvgPool2d(3, 1)(x * y) - mu_xy

    numer = (2 * mu_xy + C1)*(2 * sigma_xy + C2)
    denom = (mu_xx + mu_yy + C1) * (sigma_x + sigma_y + C2)
    score = numer / denom

    return torch.clamp((1.0 - score) / 2.0, 0.0, 1.0)
