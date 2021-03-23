import torch
import torch.nn as nn
from torch.nn import functional as F

torch_ver = torch.__version__[:3]


class Channel_Attention(nn.Module):
    def __init__(self, in_dim):
        super(Channel_Attention, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.contiguous().view(m_batchsize, C, -1)
        proj_key = x.contiguous().view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.contiguous().view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class _PyramidSelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(_PyramidSelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        # self.gamma_s = nn.Parameter(torch.zeros(1))
        # self.gamma_c = nn.Parameter(torch.zeros(1))

        if out_channels == None:
            self.out_channels = in_channels
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels)
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels)
        )

        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels//2,
                      kernel_size=1, stride=1, padding=0)

        # self.channel_attention = Channel_Attention(in_dim=self.out_channels//2)

        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):
        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)

        local_x = []
        local_y = []
        step_h, step_w = h // self.scale, w // self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y + step_w, w)
                if i == (self.scale - 1):
                    end_x = h
                if j == (self.scale - 1):
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)

        local_list = []
        local_block_cnt = 2 * self.scale * self.scale
        for i in range(0, local_block_cnt, 2):
            value_local = value[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            query_local = query[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            key_local = key[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]

            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size, self.value_channels, -1)
            value_local = value_local.permute(0, 2, 1)

            query_local = query_local.contiguous().view(batch_size, self.key_channels, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size, self.key_channels, -1)

            sim_map = torch.matmul(query_local, key_local)
            sim_map = (self.key_channels ** -.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

            context_local = torch.matmul(sim_map, value_local)
            context_local = context_local.permute(0, 2, 1).contiguous()
            context_local = context_local.view(batch_size, self.value_channels, h_local, w_local)
            local_list.append(context_local)

        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                row_tmp.append(local_list[j + i * self.scale])
            context_list.append(torch.cat(row_tmp, 3))

        context = torch.cat(context_list, 2)
        # ch_context = self.channel_attention(value)

        context = self.W(context)

        return context


class PyramidSelfAttentionBlock2D(_PyramidSelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(PyramidSelfAttentionBlock2D, self).__init__(in_channels,
                                                          key_channels,
                                                          value_channels,
                                                          out_channels,
                                                          scale)


class Pyramid_OC_Module(nn.Module):
    """
    Output the combination of the context features and the original features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, dropout, sizes=([1])):
        super(Pyramid_OC_Module, self).__init__()
        self.group = len(sizes)
        self.stages = []
        # in_channels, key_channels, value_channels, out_channels=None, scale=1
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, in_channels//8, in_channels//8, size) for size in sizes])

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(1024+512, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout)
        )

        # self.gamma_a = nn.Parameter(torch.ones(4))
        # self.softmax = nn.Softmax()
        # self.gamma_b = nn.Parameter(torch.zeros(1))
        # self.downsample = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_channels)
        # )
        #
        # self.channel_attention = Channel_Attention(in_dim=256)

        # self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(out_channels, out_channels, 1))

        # self.up_dr = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels * self.group, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(in_channels * self.group)
        # )

        # self.gamma0 = nn.Parameter(torch.zeros(1))
        # self.gamma1 = nn.Parameter(torch.zeros(1))
        # self.gamma2 = nn.Parameter(torch.zeros(1))
        # self.gamma3 = nn.Parameter(torch.zeros(1))

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return PyramidSelfAttentionBlock2D(in_channels,
                                           key_channels,
                                           value_channels,
                                           output_channels,
                                           size)

    def weighted_avg(self, lst, weight):
        s = 0
        for i in range(len(weight)):
            s += lst[i] * weight[i]
        return s

    def forward(self, feats):

        # downsample_feat = self.downsample(feats)
        # channel_feat = self.channel_attention(downsample_feat)

        # downsample_feat = downsample_feat + channel_feat
        priors = [stage(feats) for stage in self.stages]
        context = [feats]
        # context = [priors[0]]
        for i in range(0, len(priors)):
            context.append(priors[i])
        # weight = F.softmax(self.gamma_a, 0)

        # context = feats + self.gamma_b * self.weighted_avg(priors, weight)

        # out = self.conv_bn_dropout(torch.cat(context, 1))
        # out = feats + self.gamma0 * priors[0] + self.gamma1*priors[1] + \
        #       self.gamma2*priors[2] + self.gamma3*priors[3]

        # out = self.conv8(out)

        # out = self.conv_bn_dropout(context)
        out = self.conv_bn_dropout(torch.cat(context, 1))

        return out

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x:x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params
