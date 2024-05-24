import numpy as np
import torch
import datetime
class MaxPooling2D:
    # 为了实现 torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # N x 1 x H x W
    def __init__(self, kernel_size=3, stride=1, padding=1):
        self.kernel_size = kernel_size
        self.w_height = kernel_size
        self.w_width = kernel_size

        self.stride = stride
        self.padding = padding

        self.x = None
        self.in_height = None
        self.in_width = None

        self.out_height = None
        self.out_width = None

        self.arg_max = None

    def __call__(self, x, normal):
        # x 为夹角
        time_begin = datetime.datetime.now()

        self.batch, _, self.in_height, self.in_width = x.shape
        x_new = torch.zeros(self.batch, 1, self.in_height+self.padding*2, self.in_width+self.padding*2)
        x_new[:, :, self.padding:-self.padding, self.padding:-self.padding] = x
        x = x_new
        print(x.shape)
        self.normal = normal

        self.out_height = int((self.in_height - self.w_height + 2*self.padding) / self.stride) + 1
        self.out_width = int((self.in_width - self.w_width + 2*self.padding) / self.stride) + 1
        out = normal
        error_count = 0

        for n in range(self.batch):
            for i in range(self.out_height):
                for j in range(self.out_width):
                    start_i = i * self.stride
                    start_j = j * self.stride
                    end_i = start_i + self.w_height
                    end_j = start_j + self.w_width
                    mini_x = (x[n, :, start_i: end_i, start_j: end_j]).view(self.w_height, self.w_width)
                    index = torch.argmax(mini_x)

                    try:
                        out[n, :, i, j] = normal[n, :, start_i+index % self.w_height, start_j+index // self.w_width]
                    except:
                        error_count += 1
        print('error count', error_count)
        print('time cost', datetime.datetime.now()-time_begin)

        return out

    def backward(self, d_loss):
        dx = np.zeros_like(self.x)
        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                index = np.unravel_index(self.arg_max[i, j], self.kernel_size)
                dx[start_i:end_i, start_j:end_j][index] = d_loss[i, j] #
        return dx

def pool2d(X, pool_size):
    p_h, p_w = pool_size
    X=X.float()
    m,n=0,0
    Y = torch.zeros((int((X.shape[0] - p_h + 5)/5), int((X.shape[1] - p_h + 5)/5)),dtype=torch.float)
    #同时修改pool_size和除数，就可以修改池化的步长
    for i in range(0,Y.shape[0]):
        n=0
        for j in range(Y.shape[1]):
                Y[i, j] = X[m:m+p_h, n:n+p_w].max()
                if n+p_w < X.shape[1]:
                    n+=p_w
        if m+p_h < X.shape[0]:
            m+=p_h
    return Y


normal = torch.randn(8, 3, 320, 768)
x = torch.randn(8, 1, 320, 768)
max_pool = MaxPooling2D(kernel_size=5, stride=1, padding=2)
output = max_pool(x, normal)
time_1= datetime.datetime.now()
pool = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
out = pool(x)
print('time cost = ', datetime.datetime.now()-time_1)




