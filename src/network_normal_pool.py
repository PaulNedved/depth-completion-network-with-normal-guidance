class SparseToDensePool_for_normal(torch.nn.Module):

    def __init__(self,
                 input_channels,
                 min_pool_sizes,
                 max_pool_sizes,
                 n_filter=24,  # 24
                 n_convolution=3,
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu'):
        super(SparseToDensePool_for_normal, self).__init__()

        activation_func = net_utils.activation_func(activation_func)

        self.min_pool_sizes = min_pool_sizes
        self.max_pool_sizes = max_pool_sizes

        # Construct min pools
        self.min_pools = []
        for s in self.min_pool_sizes:
            padding = s // 2
            pool = torch.nn.MaxPool2d(kernel_size=s, stride=1, padding=padding, return_indices=True)
            self.min_pools.append(pool)

        # Construct max pools
        self.max_pools = []
        for s in self.max_pool_sizes:
            padding = s // 2
            pool = torch.nn.MaxPool2d(kernel_size=s, stride=1, padding=padding, return_indices=True)
            self.max_pools.append(pool)

        self.len_pool_sizes = len(self.min_pool_sizes) + len(self.max_pool_sizes)

        in_channels = len(self.min_pool_sizes) + len(self.max_pool_sizes)

        pool_convs = []
        for n in range(n_convolution):  # n_convolution=3   三层1*1卷积
            conv = net_utils.Conv2d(
                in_channels,
                n_filter,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=False)
            pool_convs.append(conv)

            # Set new input channels as output channels
            in_channels = n_filter  # 8

        self.pool_convs = torch.nn.Sequential(*pool_convs)

        in_channels = n_filter + input_channels  # 3*3卷积输入通道数

        self.conv = net_utils.Conv2d(
            in_channels,
            n_filter,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=False)

    def forward(self, depth, normal, intrinsics):
        print('network.py 1459\n  sparse to dense for normal time begin\n', datetime.datetime.now())
        time_begin = datetime.datetime.now()

        # Input depth
        n_batch, _, n_height, n_width = normal.shape  # N x 3 x H x W
        hw = n_height * n_width
        normal0 = normal.cuda()
        normal_t = normal.view(4, -1).cuda()

        xy_h = net_utils.meshgrid(n_batch, n_height, n_width, device=depth.device, homogeneous=True)
        xy_h = xy_h.view(n_batch, 3, -1)
        validiation_map = depth[:, 0, :, :].view(n_batch, 1, n_height, n_width)
        depth = depth[:, 0, :, :].view(n_batch, 1, -1)

        # K^-1 [u, v, 1]    points.shape= N x 3 x (H x W)
        points = torch.matmul(torch.inverse(intrinsics), xy_h) * depth

        points = torch.nn.functional.normalize(points, dim=1)  # N x 3 x (H x W)

        normal = normal.contiguous().view(n_batch, 3, -1)  # N x 3 x (H x W)

        cos = points * normal
        cos = (torch.sum(cos, dim=1) - 2) * (-1)  # N x HW

        direction = cos.view(n_batch, 1, n_height, n_width) * validiation_map  # N x 1 x H x W *validiation map

        pool_pyramid = []

        print('network.py 1486\n  sparse to dense for normal data process time\n', datetime.datetime.now() - time_begin)
        time_1 = datetime.datetime.now()

        # Use min and max pooling to densify and increase receptive field
        for pool_nor in (self.min_pools):

            normal_new = torch.zeros(n_batch, hw).cuda()

            _, idx = pool_nor(torch.where(direction == 0, -999 * torch.ones_like(direction), -direction))

            time_min_pool = datetime.datetime.now()
            print('time for min_pool', time_min_pool - time_1)

            indices_dim = idx.view(4, -1)

            for batch in range(4):
                for i in range(hw):
                    normal_new[batch, i] = normal_t[batch, indices_dim[batch, i]]

            normal_new = normal_new.view(n_batch, -1, n_height, n_width).cuda()

            print('time for min_save new normal', datetime.datetime.now() - time_min_pool)
            # Remove any 999 from the results

            pool_pyramid.append(normal_new)

        time_max_pool_begin = datetime.datetime.now()
        for pool in self.max_pools:
            normal_new = torch.zeros(n_batch, hw).cuda()
            _, idx = pool(direction)
            time_max_pool = datetime.datetime.now()
            print('time for max pool', time_max_pool - time_max_pool_begin)
            indices_dim = -idx.view(4, -1)
            for batch in range(4):
                for i in range(hw):
                    normal_new[batch, i] = normal_t[batch, indices_dim[batch, i]]

            normal_new = normal_new.view(n_batch, -1, n_height, n_width)

            print('time for max save new normal', datetime.datetime.now() - time_max_pool_begin)
            time_2 = datetime.datetime.now()

            pool_pyramid.append(normal_new)

            print('time for append', datetime.datetime.now() - time_2)

        # Stack max and minpools into pyramid
        pool_pyramid = torch.cat(pool_pyramid, dim=1)

        # Learn weights for different kernel sizes, and near and far structures
        pool_convs = self.pool_convs(pool_pyramid)

        pool_convs = torch.cat([pool_convs, normal0], dim=1)

        print('network.py 1528\n  sparse to dense for normal convolutional time\n', datetime.datetime.now() - time_1)

        return self.conv(pool_convs)  