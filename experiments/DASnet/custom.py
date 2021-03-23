from models.DASnet import DASnet
from models.features import MultiStageFeature
from models.rpn import RPN, DepthCorr
from models.mask import Mask
import torch
import torch.nn as nn
from utils.load_helper import load_pretrain
from experiments.DASnet.resnet import resnet50
from experiments.gcb_block.pyramid_oc_block import Pyramid_OC_Module

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
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.contiguous().view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x: x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params


class ResDownS(nn.Module):
    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()

        self.downsample_pyramid = Pyramid_OC_Module(in_channels=1024, out_channels=256, dropout=0.05,
                                                    sizes=([1, 2, 3, 6]))

        self.downsample = nn.Sequential(
            nn.Conv2d(inplane, outplane, kernel_size=1, bias=False),
            nn.BatchNorm2d(outplane))

    def forward(self, x):
        # ori_x = x
        down_x = self.downsample(x)
        ori_x = down_x
        if down_x.size(3) < 20:
            l = 4
            r = -4
            down_x = down_x[:, :, l:r, l:r]
        if x.size(3) < 20:
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]
        pyra_x = self.downsample_pyramid(x)

        return down_x, pyra_x, ori_x


class ResDown(MultiStageFeature):
    def __init__(self, pretrain=False):
        super(ResDown, self).__init__()
        self.features = resnet50(layer3=True, layer4=False)
        if pretrain:
            load_pretrain(self.features, 'resnet.model')

        self.downsample = ResDownS(1024, 256)

        self.layers = [self.downsample, self.features.layer2, self.features.layer3]
        self.train_nums = [1, 3]
        self.change_point = [0, 0.5]

        self.unfix(0.0)

    def param_groups(self, start_lr, feature_mult=1):
        lr = start_lr * feature_mult

        def _params(module, mult=1):
            params = list(filter(lambda x: x.requires_grad, module.parameters()))
            if len(params):
                return [{'params': params, 'lr': lr * mult}]
            else:
                return []

        groups = []
        groups += _params(self.downsample)
        groups += _params(self.features, 0.1)
        return groups

    def forward(self, x):
        output = self.features(x)
        p3, pyra, ori = self.downsample(output[-1])
        return p3, pyra, ori

    def forward_all(self, x):
        output = self.features(x)
        p3, pyra, ori = self.downsample(output[-1])
        return output, p3, pyra, ori


class UP(RPN):
    def __init__(self, anchor_num=5, feature_in=256, feature_out=256):
        super(UP, self).__init__()

        self.anchor_num = anchor_num
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.cls_output = 2 * self.anchor_num
        self.loc_output = 4 * self.anchor_num

        self.cls = DepthCorr(feature_in, feature_out, self.cls_output)
        self.loc = DepthCorr(feature_in, feature_out, self.loc_output)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class Decide_RPN(nn.Module):
    def __init__(self, anchor_num):
        super(Decide_RPN, self).__init__()
        self.channel_rpn = UP(anchor_num=anchor_num, feature_in=256, feature_out=256)
        self.spatial_rpn = UP(anchor_num=anchor_num, feature_in=256, feature_out=256)

        self.cls_weight = nn.Parameter(torch.ones(2))
        self.loc_weight = nn.Parameter(torch.ones(2))

    def forward(self, zf, xf):
        cls = []
        loc = []

        channel_cls, channel_loc = self.channel_rpn(zf[0], xf[0])
        spatial_cls, spatial_loc = self.spatial_rpn(zf[1], xf[1])

        cls.append(channel_cls)
        cls.append(spatial_cls)

        loc.append(channel_loc)
        loc.append(spatial_loc)

        cls_weight = F.softmax(self.cls_weight, 0)
        loc_weight = F.softmax(self.loc_weight, 0)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)

    def param_groups(self, start_lr, feature_mult=1, key=None):
        if key is None:
            params = filter(lambda x: x.requires_grad, self.parameters())
        else:
            params = [v for k, v in self.named_parameters() if (key in k) and v.requires_grad]
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params


class MaskCorr(Mask):
    def __init__(self, oSz=63):
        super(MaskCorr, self).__init__()
        self.oSz = oSz
        self.mask = DepthCorr(256, 256, self.oSz**2)

    def forward(self, z, x):
        return self.mask(z, x)


class Custom(SiamMask):
    def __init__(self, pretrain=False, **kwargs):
        super(Custom, self).__init__(**kwargs)
        self.features = ResDown(pretrain=pretrain)
        self.rpn_model = Decide_RPN(anchor_num=self.anchor_num)
        self.mask_model = MaskCorr()

        self.zf_channel_attention_512 = Channel_Attention(in_dim=256)
        self.xf_channel_attention_512 = Channel_Attention(in_dim=256)

    def template(self, template):
        zf, pyra_zf, ori_zf = self.features(template)
        # ori_zf = [b, 256, 15, 15]
        zf = self.zf_channel_attention_512(zf)

        self.zf = [zf, pyra_zf]

    def track(self, search):
        search, pyra_xf, ori_xf = self.features(search)

        search = self.xf_channel_attention_512(search)

        xf = [search, pyra_xf]

        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, xf)

        return rpn_pred_cls, rpn_pred_loc

    def track_mask(self, search):
        self.feature, search, pyra_xf, ori_xf = self.features.forward_all(search)

        search = self.xf_channel_attention_512(search)
        xf = [search, pyra_xf]

        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, xf)

        self.match_search = ori_xf
        self.corr_feature = self.mask_model.mask.forward_corr(self.zf[0], xf[0])
        self.corr_feature = self.mask_fuse(self.corr_feature, self.zf[1], xf[1])

        pred_mask = self.mask_model.mask.head(self.corr_feature)

        return rpn_pred_cls, rpn_pred_loc, pred_mask

