import paddle.nn.functional as F
import paddle
import paddle.nn as nn

def convWithBias(in_channels,out_channels,kernel_size,stride,padding,bias_attr=True):
    return nn.Conv2D(in_channels,out_channels,kernel_size,stride,padding,bias_attr=bias_attr)

class PReNet(nn.Layer):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            convWithBias(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            convWithBias(32, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.shape[0], input.shape[2], input.shape[3]

        x = input
        # h  = paddle.create_parameter(shape=(batch_size, 32, row, col),dtype='float32',is_bias=True)
        # c  = paddle.create_parameter(shape=(batch_size, 32, row, col),dtype='float32',is_bias=True)

        h = paddle.zeros(shape=(batch_size, 32, row, col),dtype='float32')
        c = paddle.zeros(shape=(batch_size, 32, row, col),dtype='float32')

        x_list = []
        for i in range(self.iteration):
            x = paddle.concat((input, x), 1)
            x = self.conv0(x)

            x = paddle.concat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * paddle.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list


class PReNet_LSTM(nn.Layer):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_LSTM, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            convWithBias(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            convWithBias(32, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.shape[0], input.shape[2], input.shape[3]

        x = input
        h  = paddle.create_parameter(shape=(batch_size, 32, row, col),dtype='float32',is_bias=True)
        c  = paddle.create_parameter(shape=(batch_size, 32, row, col),dtype='float32',is_bias=True)


        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x1 = x
            x = paddle.concat((input, x), 1)
            x = self.conv0(x)

            x = paddle.concat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * paddle.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x_list.append(x)

        return x, x_list


class PReNet_GRU(nn.Layer):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_GRU, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            convWithBias(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_z = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_b = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        # self.conv_o = nn.Sequential(
        #     convWithBias(32 + 32, 32, 3, 1, 1),
        #     nn.Sigmoid()
        #     )
        self.conv = nn.Sequential(
            convWithBias(32, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.shape[0], input.shape[2], input.shape[3]

        x = input
        h  = paddle.create_parameter(shape=(batch_size, 32, row, col),dtype='float32',is_bias=True)

        if self.use_GPU:
            h = h.cuda()

        x_list = []
        for i in range(self.iteration):
            x = paddle.concat((input, x), 1)
            x = self.conv0(x)

            x1 = paddle.concat((x, h), 1)
            z = self.conv_z(x1)
            b = self.conv_b(x1)
            s = b * h
            s = paddle.concat((s, x), 1)
            g = self.conv_g(s)
            h = (1 - z) * h + z * g

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)

            x = self.conv(x)
            x_list.append(x)

        return x, x_list


class PReNet_x(nn.Layer):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_x, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            convWithBias(3, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            convWithBias(32, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.shape[0], input.shape[2], input.shape[3]

        x = input
        h  = paddle.create_parameter(shape=(batch_size, 32, row, col),dtype='float32',is_bias=True)
        c  = paddle.create_parameter(shape=(batch_size, 32, row, col),dtype='float32',is_bias=True)

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            #x = paddle.concat((input, x), 1)
            x = self.conv0(x)

            x = paddle.concat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * paddle.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)

            x = self.conv(x)
            x_list.append(x)

        return x, x_list


class PReNet_r(nn.Layer):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_r, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            convWithBias(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            convWithBias(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            convWithBias(32, 3, 3, 1, 1),
            )


    def forward(self, input):
        batch_size, row, col = input.shape[0], input.shape[2], input.shape[3]
        #mask  = paddle.create_parameter(shape=(batch_size, 3, row, col),dtype='float32',is_bias=True)
        x = input
        # h  = paddle.create_parameter(shape=(batch_size, 32, row, col),dtype='float32',is_bias=True)
        # c  = paddle.create_parameter(shape=(batch_size, 32, row, col),dtype='float32',is_bias=True)
        h = paddle.zeros(shape=(batch_size, 32, row, col),dtype='float32')
        c = paddle.zeros(shape=(batch_size, 32, row, col),dtype='float32')

        x_list = []
        for i in range(self.iteration):
            x = paddle.concat((input, x), 1)
            x = self.conv0(x)

            x = paddle.concat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * paddle.tanh(c)

            x = h
            for j in range(5):
                resx = x
                x = F.relu(self.res_conv1(x) + resx)

            x = self.conv(x)
            x = input + x
            x_list.append(x)

        return x, x_list


## PRN
class PRN(nn.Layer):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PRN, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            convWithBias(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            convWithBias(32, 3, 3, 1, 1),
        )

    def forward(self, input):

        x = input

        x_list = []
        for i in range(self.iteration):
            x = paddle.concat((input, x), 1)
            x = self.conv0(x)
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list


class PRN_r(nn.Layer):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PRN_r, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            convWithBias(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU(),
            convWithBias(32, 32, 3, 1, 1),
            nn.ReLU()
            )

        self.conv = nn.Sequential(
            convWithBias(32, 3, 3, 1, 1),
            )

    def forward(self, input):

        x = input

        x_list = []
        for i in range(self.iteration):
            x = paddle.concat((input, x), 1)
            x = self.conv0(x)

            for j in range(5):
                resx = x
                x = F.relu(self.res_conv1(x) + resx)

            x = self.conv(x)
            x = input + x
            x_list.append(x)

        return x, x_list
