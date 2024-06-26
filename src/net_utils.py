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
import time

import torch
import nor_test
import numpy

EPSILON = 1e-10


def activation_func(activation_fn):
    '''
    Select activation function

    Arg(s):
        activation_fn : str
            name of activation function
    '''

    if 'linear' in activation_fn:
        return None
    elif 'leaky_relu' in activation_fn:
        return torch.nn.LeakyReLU(negative_slope=0.20, inplace=True)
    elif 'relu' in activation_fn:
        return torch.nn.ReLU()
    elif 'elu' in activation_fn:
        return torch.nn.ELU()
    elif 'sigmoid' in activation_fn:
        return torch.nn.Sigmoid()
    else:
        raise ValueError('Unsupported activation function: {}'.format(activation_fn))


'''
Network layers
'''


class Conv2d(torch.nn.Module):
    '''
    2D convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(Conv2d, self).__init__()

        self.use_batch_norm = use_batch_norm  # FALSE
        padding = kernel_size // 2  # 向下取整

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # 输出：batch_size（一个batch中样例的个数）,output（输出的深度out_channels）, height_3（卷积结果的高度）, width_3 ]
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)

        self.activation_func = activation_func

        if self.use_batch_norm:  # FALSE
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        conv = self.conv(x)
        conv = self.batch_norm(conv) if self.use_batch_norm else conv  # 不执行 设了use_batch_norm=false

        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class TransposeConv2d(torch.nn.Module):
    '''
    Transpose convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel (k x k)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(TransposeConv2d, self).__init__()

        self.use_batch_norm = use_batch_norm
        padding = kernel_size // 2  # 1
        # torch.nn.ConvTranspose2d
        # 输入: (N,C_in,H_in，W_in)
        # 输出: (N,C_out,H_out,W_out)
        self.deconv = torch.nn.ConvTranspose2d(
            in_channels,  # 256
            out_channels,  # 256
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)

        self.activation_func = activation_func

        if self.use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        deconv = self.deconv(x)
        deconv = self.batch_norm(deconv) if self.use_batch_norm else deconv
        if self.activation_func is not None:
            return self.activation_func(deconv)
        else:
            return deconv


class UpConv2d(torch.nn.Module):
    '''
    Up-convolution (upsample + convolution) block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        shape : list[int]
            two element tuple of ints (height, width)
        kernel_size : int
            size of kernel (k x k)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(UpConv2d, self).__init__()

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

    def forward(self, x, shape):
        upsample = torch.nn.functional.interpolate(x,
                                                   size=shape)  # 用来上采样或下采样，可以给定size或者scale_factor来进行上下采样。同时支持3D、4D、5D的张量输入。
        conv = self.conv(upsample)
        return conv


'''
Network encoder blocks
'''


class ResNetBlock(torch.nn.Module):
    '''
    Basic ResNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(ResNetBlock, self).__init__()

        self.activation_func = activation_func

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.projection = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False)

    def forward(self, x):
        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x

        # f(x) + x
        return self.activation_func(conv2 + X)


class ResNetBottleneckBlock(torch.nn.Module):
    '''
    ResNet bottleneck block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(ResNetBottleneckBlock, self).__init__()

        self.activation_func = activation_func

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv3 = Conv2d(
            out_channels,
            4 * out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.projection = Conv2d(
            in_channels,
            4 * out_channels,
            kernel_size=1,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False)

    def forward(self, x):
        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x

        # f(x) + x
        return self.activation_func(conv3 + X)


class VGGNetBlock(torch.nn.Module):
    '''
    VGGNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        n_conv : int
            number of convolution layers
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_conv=1,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(VGGNetBlock, self).__init__()

        layers = []
        for n in range(n_conv - 1):
            conv = Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
            layers.append(conv)
            in_channels = out_channels

        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)
        layers.append(conv)

        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class CalibratedBackprojectionBlock(torch.nn.Module):
    '''
    Calibrated backprojection (KB) layer class

    Arg(s):
        in_channels_image : int
            number of input channels for image (RGB) branch
        in_channels_depth : int
            number of input channels for depth branch
        in_channels_fused : int
            number of input channels for RGB 3D fusion branch
        n_filter_image : int
            number of filters for image (RGB) branch
        n_filter_depth : int
            number of filters for depth branch
        n_filter_fused : int
            number of filters for RGB 3D fusion branch
        n_convolution_image : int
            number of convolution layers in image branch
        n_convolution_depth : int
            number of convolution layers in depth branch
        n_convolution_fused : int
            number of convolution layers in RGB 3D fusion branch
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
    '''

    # n = 0：
    # in_channels_image = 48， in_channels_depth = 16， 
    # in_channels_fused = 48 + 48+ 16

    def __init__(self,
                 in_channels_image,
                 in_channels_depth,
                 in_channels_normal,
                 in_channels_fused,
                 n_filter_image=48,
                 n_filter_depth=16,
                 n_filter_normal=48,
                 n_filter_fused=48,
                 n_convolution_image=1,
                 n_convolution_depth=1,
                 n_convolution_normal=1,
                 n_convolution_fused=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)):
        super(CalibratedBackprojectionBlock, self).__init__()

        self.conv_image = VGGNetBlock(
            in_channels=in_channels_image,  # 48
            out_channels=n_filter_image,  # 48
            n_conv=n_convolution_image,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        self.conv_normal = VGGNetBlock(
            in_channels=in_channels_image+3,  # 48
            out_channels=n_filter_image,  # 48
            n_conv=n_convolution_image,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        self.conv_depth = VGGNetBlock(
            # in_channels=in_channels_depth + 3, depth+coordinates
            in_channels=in_channels_depth + 3,
            out_channels=n_filter_depth,  # 16
            n_conv=n_convolution_depth,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        self.proj_depth = Conv2d(
            # 将depth投影到一维 （16通道缩为一通道）
            in_channels=in_channels_depth,
            out_channels=1,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        self.conv_fused = Conv2d(
            in_channels=in_channels_fused,
            out_channels=n_filter_fused,
            kernel_size=1,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

    def forward(self, image, depth, normal, coordinates, fused=None):

        layers_fused = []
        # Obtain image (RGB) features
        conv_image = self.conv_image(image)

        # Obtain depth (Z) features
        conv_depth = self.conv_depth(torch.cat([depth, coordinates], dim=1))

        # Include image (RGB) features
        layers_fused.append(image)

        # Project depth features to 1 dimension
        z = self.proj_depth(depth)

        # Include backprojected 3D positional (XYZ) encoding: K^-1 [x y 1] z
        xyz = coordinates * z  # N x 3 x H x W

        layers_fused.append(xyz)
        normal_3d_encoding = nor_test.estimate_normal_leastsq_manu(xyz) # N x 3 x H x W
        conv_normal = self.conv_normal(torch.cat([normal, normal_3d_encoding], dim=1))

        layers_fused.append(normal)
        layers_fused.append(normal_3d_encoding)

        # Include previous RGBXYZ representation
        if fused is not None:
            # print('fused', fused.size())
            layers_fused.append(fused)

        # Obtain fused (RGBXYZ) representation
        # print('layers_fused_size', image.size(), xyz.size(), normal.size(), normal_3d_encoding.size())
        layers_fused = torch.cat(layers_fused, dim=1)

        conv_fused = self.conv_fused(layers_fused)
        return conv_image, conv_depth, conv_normal, conv_fused


'''
Network decoder blocks
'''


class DecoderBlock(torch.nn.Module):
    '''
    Decoder block with skip connections (U-Net)

    Arg(s):
        in_channels : int
            number of input channels
        skip_channels : int
            number of skip connection channels
        out_channels : int
            number of output channels
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        deconv_type : str
            deconvolution types: transpose, up
    '''

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 deconv_type='transpose'):
        # deconv_type=up
        super(DecoderBlock, self).__init__()

        self.skip_channels = skip_channels
        self.deconv_type = deconv_type

        if deconv_type == 'transpose':
            self.deconv = TransposeConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        elif deconv_type == 'up':
            self.deconv = UpConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

        concat_channels = skip_channels + out_channels
        self.conv = Conv2d(
            concat_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

    def forward(self, x, skip=None):

        if self.deconv_type == 'transpose':  # no
            deconv = self.deconv(x)
        elif self.deconv_type == 'up':
            if skip is not None:
                shape = skip.shape[2:4]  # 读第二第三维长度（n_height, n_width）
            else:
                n_height, n_width = x.shape[2:4]
                shape = (int(2 * n_height), int(2 * n_width))

            deconv = self.deconv(x, shape=shape)
        
        if self.skip_channels > 0:
            concat = torch.cat([deconv, skip], dim=1)
        else:
            concat = deconv
            
        return self.conv(concat)


'''
Pose regression layer
'''


def pose_matrix(v, rotation_parameterization='axis'):
    '''
    Convert 6 DoF parameters to transformation matrix
    将6自由度参数转换为变换矩阵

    Arg(s):
        v : torch.Tensor[float32]
            N x 6 vector in the order of tx, ty, tz, rx, ry, rz
        rotation_parameterization : str
            axis
    Returns:
        torch.Tensor[float32] : N x 4 x 4 homogeneous transformation matrix
    '''

    # Select N x 3 element rotation vector
    r = v[..., :3]

    # Select N x 3 element translation vector
    t = v[..., 3:]

    # r, t shape都是 torch.Size([8, 3])

    if rotation_parameterization == 'axis':
        Rt = transformation_from_parameters(torch.unsqueeze(r, dim=1), t)
    else:
        raise ValueError('Unsupported rotation parameterization: {}'.format(rotation_parameterization))

    return Rt


'''
Utility functions for rotation
'''


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    # print('angle', angle.shape)  ([8, 1, 1])

    axis = vec / (angle + 1e-7)
    # print('axis', axis.shape)   ([8, 1, 3])

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1
    # print('rot', rot.shape)  torch.Size([8, 4, 4])
    return rot


'''
Utility functions for rigid warping
'''


def meshgrid(n_batch, n_height, n_width, device, homogeneous=True):
    '''
    Creates N x 2 x H x W meshgrid in x, y directions

    Arg(s):
        n_batch : int
            batch size
        n_height : int
            height of tensor
        n_width : int
            width of tensor
        device : torch.device
            device on which to create meshgrid
        homoegenous : bool
            if set, then add homogeneous coordinates (N x H x W x 3)
    Return:
        torch.Tensor[float32]: N x 2 x H x W meshgrid of x, y and 1 (if homogeneous)
    '''
    # linspace 函数的作用是，返回一个一维的tensor（张量），这个张量包含了从start到end，分成steps个线段得到的向量。常用的几个变量
    x = torch.linspace(start=0.0, end=n_width - 1, steps=n_width, device=device)
    y = torch.linspace(start=0.0, end=n_height - 1, steps=n_height, device=device)

    # Create H x W grids
    # torch.meshgrid（）的功能是生成网格，可以用于生成坐标。函数输入两个数据类型相同的一维张量，两个输出张量的行数为第一个输入张量的元素个数，列数为第二个输入张量的元素个数，当两个输入张量数据类型不同或维度不是一维时会报错。
    # 其中第一个输出张量填充第一个输入张量中的元素，各行元素相同；第二个输出张量填充第二个输入张量中的元素各列元素相同。
    grid_y, grid_x = torch.meshgrid(y, x)

    if homogeneous:  # homogeneous = TRUE
        # Create 3 x H x W
        # grid (x, y, 1) 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
        grid_xy = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)  # stack dim=0 直接拼起来
    else:
        # Create 2 x H x W grid (x, y)
        grid_xy = torch.stack([grid_x, grid_y], dim=0)

    grid_xy = torch.unsqueeze(grid_xy, dim=0) \
        .repeat(n_batch, 1, 1, 1)  # ？repeat 怎么四个参数

    return grid_xy


def meshgrid_new(n_batch, n_height, n_width, device, homogeneous=True):
    # linspace 函数的作用是，返回一个一维的tensor（张量），这个张量包含了从start到end，分成steps个线段得到的向量。常用的几个变量
    x_below = torch.linspace(start=0.0, end=n_width - 2, steps=n_width - 1, device=device)
    y_below = torch.linspace(start=1.0, end=n_height - 1, steps=n_height - 1, device=device)

    x_right = torch.linspace(start=1.0, end=n_width - 1, steps=n_width - 1, device=device)
    y_right = torch.linspace(start=0.0, end=n_height - 2, steps=n_height - 1, device=device)

    grid_y_below, grid_x_below = torch.meshgrid(y_below, x_below)
    grid_y_right, grid_x_right = torch.meshgrid(y_right, x_right)

    if homogeneous:  # homogeneous = TRUE
        grid_xy_below = torch.stack([grid_x_below, grid_y_below, torch.ones_like(grid_x_below)], dim=0)
        grid_xy_right = torch.stack([grid_x_right, grid_y_right, torch.ones_like(grid_x_right)], dim=0)
    else:
        # Create 2 x H x W grid (x, y)
        grid_xy_below = torch.stack([grid_x_below, grid_y_below], dim=0)
        grid_xy_right = torch.stack([grid_xy_right, grid_y_right], dim=0)

    grid_xy_below = torch.unsqueeze(grid_xy_below, dim=0) \
        .repeat(n_batch, 1, 1, 1)

    grid_xy_right = torch.unsqueeze(grid_xy_right, dim=0) \
        .repeat(n_batch, 1, 1, 1)

    return grid_xy_below, grid_xy_right


def backproject_to_camera(depth, intrinsics, shape):
    '''
    Backprojects pixel coordinates to 3D camera coordinates

    Arg(s):
        depth : torch.Tensor[float32]
            N x 1 x H x W depth map
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list[int]
            shape of tensor in (N, C, H, W)
    Return:
        torch.Tensor[float32] : N x 4 x (H x W)
    '''
    n_batch, _, n_height, n_width = shape

    # Create homogeneous coordinates [x, y, 1]
    xy_h = meshgrid(n_batch, n_height, n_width, device=depth.device, homogeneous=True)

    # Reshape pixel coordinates to N x 3 x (H x W)
    xy_h = xy_h.view(n_batch, 3, -1)

    # Reshape depth as N x 1 x (H x W)
    depth = depth.view(n_batch, 1, -1)

    # K^-1 [x, y, 1] z
    points = torch.matmul(torch.inverse(intrinsics), xy_h) * depth
    points_3dim = points

    # Make homogeneous
    return points_3dim, torch.cat([points, torch.ones_like(depth)], dim=1)


def backproject_to_camera_normal(normal, intrinsics, shape):
    '''
    Backprojects pixel coordinates to 3D camera coordinates

    Arg(s):
        normal : torch.Tensor[float32]
            N x 3 x H x W depth map
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list[int]
            shape of tensor in (N, C, H, W)
    Return:
        torch.Tensor[float32] : N x 6 x (H x W)
        torch.Tensor[float32] : N x 3 x (H x W)
    '''
    n_batch, _, n_height, n_width = shape

    # Create homogeneous coordinates [x, y, 1]
    xy_h = meshgrid(n_batch, n_height, n_width, device=normal.device, homogeneous=True)

    # Reshape pixel coordinates to N x 3 x (H x W)
    xy_h = xy_h.view(n_batch, 3, -1)

    # N x 1 x (H x W)
    normal_1 = normal[:, 0, :, :].view(n_batch, 1, -1)
    normal_2 = normal[:, 1, :, :].view(n_batch, 1, -1)
    normal_3 = normal[:, 2, :, :].view(n_batch, 1, -1)

    # Reshape depth as N x 1 x (H x W)
    depth = depth.view(n_batch, 1, -1)

    # K^-1 [x, y, 1] z   N x 3 x (H x W)
    points = torch.matmul(torch.inverse(intrinsics), xy_h) * depth

    # Make homogeneous
    return torch.cat([points, normal_1, normal_2, normal_3], dim=1), \
           torch.cat([normal_1, normal_2, normal_3], dim=1)


def process_normal(normal, intrinsics):
    '''
    Backprojects pixel coordinates to 3D camera coordinates

    Arg(s):
        normal : torch.Tensor[float32]
            N x 3 x H x W depth map
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list[int]
            shape of tensor in (N, C, H, W)
    Return:
        torch.Tensor[float32] : N x 6 x (H x W)
        torch.Tensor[float32] : N x 3 x (H x W)
    '''
    n_batch, _, n_height, n_width = normal.shape

    # Create homogeneous coordinates [x, y, 1]
    xy_h = meshgrid(n_batch, n_height, n_width, device=normal.device, homogeneous=True)

    # Reshape pixel coordinates to N x 3 x (H x W)
    xy_h = xy_h.view(n_batch, 3, -1)

    # N x 1 x (H x W)
    normal_1 = normal[:, 0, :, :].view(n_batch, 1, -1)
    normal_2 = normal[:, 1, :, :].view(n_batch, 1, -1)
    normal_3 = normal[:, 2, :, :].view(n_batch, 1, -1)

    # K^-1 [x, y, 1] z   N x 3 x (H x W)
    points = torch.matmul(torch.inverse(intrinsics), xy_h)

    # Make homogeneous
    return torch.cat([points, normal_1, normal_2, normal_3], dim=1), \
           torch.cat([normal_1, normal_2, normal_3], dim=1)


def project_to_pixel(points, pose, intrinsics, shape):
    '''
    Projects points in camera coordinates to 2D pixel coordinates

    Arg(s):
        points : torch.Tensor[float32]
            N x 4 x (H x W) depth map
        pose : torch.Tensor[float32]
            N x 4 x 4 transformation matrix
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list[int]
            shape of tensor in (N, C, H, W)
    Return:
        torch.Tensor[float32] : N x 2 x H x W
    '''

    n_batch, _, n_height, n_width = shape

    # Convert camera intrinsics to homogeneous coordinates
    column = torch.zeros([n_batch, 3, 1], device=points.device)
    row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=points.device) \
        .view(1, 1, 4) \
        .repeat(n_batch, 1, 1)  # [n_batch, 1, 4]
    intrinsics = torch.cat([intrinsics, column], dim=2)  # n_batch x 3 x 4
    intrinsics = torch.cat([intrinsics, row], dim=1)  # n_batch x 4 x 4

    # Apply the transformation and project: \pi K g p
    T = torch.matmul(intrinsics, pose)
    T = T[:, 0:3, :]
    points = torch.matmul(T, points)
    points = points / (torch.unsqueeze(points[:, 2, :], dim=1) + 1e-7)
    points = points[:, 0:2, :]

    # Reshape to N x 2 x H x W
    return points.view(n_batch, 2, n_height, n_width)


def grid_sample(image, target_xy, shape, padding_mode='border'):
    '''
    Samples the image at x, y locations to target x, y locations

    Arg(s):
        image : torch.Tensor[float32]
            N x 3 x H x W RGB image
        target_xy : torch.Tensor[float32]
            N x 2 x H x W target x, y locations in image space
        shape : list[int]
            shape of tensor in (N, C, H, W)
        padding_mode : str
            padding to use when sampled out of bounds
    Return:
        torch.Tensor[float32] : N x 3 x H x W RGB image
    '''

    n_batch, _, n_height, n_width = shape

    # Swap dimensions to N x H x W x 2 for grid sample
    target_xy = target_xy.permute(0, 2, 3, 1)

    # Normalize coordinates between -1 and 1
    target_xy[..., 0] /= (n_width - 1.0)
    target_xy[..., 1] /= (n_height - 1.0)
    target_xy = 2.0 * (target_xy - 0.5)

    # Sample the image at normalized target x, y locations
    return torch.nn.functional.grid_sample(
        image,
        grid=target_xy,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True)


'''
Utility function to pre-process sparse depth
'''


class OutlierRemoval(object):
    '''
    Class to perform outlier removal based on depth difference in local neighborhood

    Arg(s):
        kernel_size : int
            local neighborhood to consider
        threshold : float
            depth difference threshold
    '''

    def __init__(self, kernel_size=7, threshold=1.5):
        self.kernel_size = kernel_size
        self.threshold = threshold

    def remove_outliers(self, sparse_depth, validity_map):
        '''
        Removes erroneous measurements from sparse depth and validity map
        从稀疏深度和有效性贴图中移除错误测量值

        Arg(s):
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W tensor sparse depth
            validity_map : torch.Tensor[float32]
                N x 1 x H x W tensor validity map
        Returns:
            torch.Tensor[float32] : N x 1 x H x W sparse depth
            torch.Tensor[float32] : N x 1 x H x W validity map
        '''

        # Replace all zeros with large values
        #   
        # torch.where(condition, x, y)
        # condition是条件，x 和 y 是同shape 的矩阵, 
        # 针对矩阵中的某个位置的元素, 满足条件就返回x，不满足就返回y
        max_value = 10 * torch.max(sparse_depth)
        sparse_depth_max_filled = torch.where(
            validity_map <= 0,
            torch.full_like(sparse_depth, fill_value=max_value),
            sparse_depth)

        # For each neighborhood find the smallest value
        padding = self.kernel_size // 2
        sparse_depth_max_filled = torch.nn.functional.pad(
            input=sparse_depth_max_filled,
            pad=(padding, padding, padding, padding),
            mode='constant',
            value=max_value)

        min_values = -torch.nn.functional.max_pool2d(
            input=-sparse_depth_max_filled,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0)

        # If measurement differs a lot from minimum value then remove

        validity_map_clean = torch.where(
            min_values < sparse_depth - self.threshold,
            torch.zeros_like(validity_map),
            torch.ones_like(validity_map))

        # Update sparse depth and validity map
        validity_map_clean = validity_map * validity_map_clean
        sparse_depth_clean = sparse_depth * validity_map_clean

        return sparse_depth_clean, validity_map_clean
