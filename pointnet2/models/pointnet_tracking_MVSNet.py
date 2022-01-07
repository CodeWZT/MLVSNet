from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import etw_pytorch_utils as pt_utils
from collections import namedtuple
import torch.nn.functional as F

from pointnet2.utils.pointnet2_modules import PointnetSAModule, PointnetFPModule, PointnetProposalModule


class TGA(nn.Module):
    def __init__(self, sf_csize, tf_csize):
        super(TGA, self).__init__()
        self.mlp1 = (pt_utils.Seq(sf_csize+tf_csize)
                     .conv1d(sf_csize, bn=True)
                     .conv1d(int(sf_csize / 2), bn=True)
                     .conv1d(1, activation=torch.nn.Sigmoid())
                     )


    def forward(self, sf, tf):
        tf_pool = F.max_pool1d(tf, kernel_size = tf.size(2))
        tps = torch.cat([sf, tf_pool.expand(-1, -1, sf.size(2))], 1)
        attention = self.mlp1(tps)
        output = attention*sf

        return output


class CBAM(nn.Module):
    def __init__(self, cn, pn):
        super(CBAM, self).__init__()
        self.MLP1 = (
            pt_utils.Seq(cn)
                .conv1d(cn, bn=True)
                .conv1d(cn, activation=None))
        self.MLP2 = (pt_utils.Seq(pn)
                .conv1d(pn, bn=True)
                .conv1d(pn, activation=None))

        self.sig1 = torch.nn.Sigmoid()
        self.sig2 = torch.nn.Sigmoid()
        self.conv_layer = pt_utils.Conv1d(2, 1, bn=True, activation=None)


    def forward(self, input):
        MP1 = F.max_pool1d(input, kernel_size=input.size(2))
        AP1 = F.avg_pool1d(input, kernel_size=input.size(2))

        MP1 = self.MLP1(MP1)
        AP1 = self.MLP1(AP1)

        weight_1 = MP1+AP1

        input = input*self.sig1(weight_1)

        MP2 = F.max_pool1d(input.transpose(1, 2).contiguous(), kernel_size=input.size(1))
        AP2 = F.avg_pool1d(input.transpose(1, 2).contiguous(), kernel_size=input.size(1))
        MP_AP = torch.cat([MP2, AP2], 2)
        MP_AP = MP_AP.transpose(1, 2).contiguous()
        MP_AP = self.conv_layer(MP_AP)
        weight_2 = self.sig2(MP_AP)
        input = input*weight_2

        return input

class Pointnet_Backbone(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, use_xyz=True):
        super(Pointnet_Backbone, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.3,
                nsample=32,
                mlp=[input_channels, 64, 64, 128],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.5,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.7,
                nsample=32,
                mlp=[256, 256, 256, 256],
                use_xyz=use_xyz,
            )
        )
        self.cov_final = nn.Conv1d(256, 256, kernel_size=1)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, numpoints):
        # type: (Pointnet2SSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i], numpoints[i])
            l_xyz.append(li_xyz)
            if i == len(self.SA_modules)-1:
                li_features = self.cov_final(li_features)
            l_features.append(li_features)

        return l_xyz, l_features


class Pointnet_Tracking(nn.Module):
    r"""
        xorr the search and the template
    """

    def __init__(self, input_channels=3, use_xyz=True, objective=False):
        super(Pointnet_Tracking, self).__init__()

        self.backbone_net = Pointnet_Backbone(input_channels, use_xyz)


        self.FC_layer_cla_2 = (
            pt_utils.Seq(256)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(1, activation=None))

        self.FC_layer_cla = (
            pt_utils.Seq(256)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(1, activation=None))


        self.vote_layer_2 = (
            pt_utils.Seq(3 + 256)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(3 + 256, activation=None))
        self.vote_layer = (
            pt_utils.Seq(3 + 256)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(3 + 256, activation=None))


        self.trans_layer_2 = PointnetSAModule(
            radius=0.3,
            nsample=32,
            mlp=[256, 256],
            use_xyz=use_xyz)

        self.vote_aggregation = PointnetSAModule(
            radius=0.3,
            nsample=16*3,
            mlp=[1 + 256, 256, 256, 256],
            use_xyz=use_xyz)

        self.num_proposal = 64
        self.FC_proposal = (
            pt_utils.Seq(256)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(3 + 1 + 1, activation=None))



        self.cbam = CBAM(256, 64)

        self.tga2 = TGA(256, 256)
        self.tga3 = TGA(256, 256)


    def forward(self, template, search):
        r"""
            template: B*512*3 or B*512*6
            search: B*1024*3 or B*1024*6
        """
        template_xyz, template_feature = self.backbone_net(template, [256, 128, 64])

        search_xyz, search_feature = self.backbone_net(search, [512, 256, 128])

        new_search_feature_2 = self.tga2(search_feature[2], template_feature[2])
        fusion_feature = self.tga3(search_feature[3], template_feature[3])

        v_2 = self.trans_layer_2(search_xyz[2], new_search_feature_2, 128)

        # layer_2
        estimation_cla_2 = self.FC_layer_cla_2(v_2[1]).squeeze(1)
        score_2 = estimation_cla_2.sigmoid()
        fusion_xyz_feature_2 = torch.cat((v_2[0].transpose(1, 2).contiguous(), v_2[1]), dim=1)
        offset_2 = self.vote_layer_2(fusion_xyz_feature_2)
        vote_2 = fusion_xyz_feature_2 + offset_2
        vote_xyz_2 = vote_2[:, 0:3, :].transpose(1, 2).contiguous()
        vote_feature_2 = vote_2[:, 3:, :]
        vote_feature_2 = torch.cat((score_2.unsqueeze(1), vote_feature_2), dim=1)

        #layer 3
        estimation_cla = self.FC_layer_cla(fusion_feature).squeeze(1)
        score = estimation_cla.sigmoid()
        fusion_xyz_feature = torch.cat((search_xyz[3].transpose(1, 2).contiguous(), fusion_feature), dim=1)
        offset = self.vote_layer(fusion_xyz_feature)
        vote = fusion_xyz_feature + offset
        vote_xyz = vote[:, 0:3, :].transpose(1, 2).contiguous()
        vote_feature = vote[:, 3:, :]
        vote_feature = torch.cat((score.unsqueeze(1), vote_feature), dim=1)

        # concat
        estimation_cla_s = [estimation_cla_2, estimation_cla]
        vote_xyz_s = [vote_xyz_2, vote_xyz]
        vote_xyz_cat = torch.cat([vote_xyz_2, vote_xyz], 1)
        vote_feature_cat = torch.cat([vote_feature_2, vote_feature], 2)
        #
        # vote_xyz_cat = vote_xyz
        # vote_feature_cat = vote_feature


        center_xyzs, proposal_features = self.vote_aggregation(vote_xyz_cat, vote_feature_cat, self.num_proposal)

        proposal_features = self.cbam(proposal_features)

        proposal_offsets = self.FC_proposal(proposal_features)

        estimation_boxs = torch.cat(
            (proposal_offsets[:, 0:3, :] + center_xyzs.transpose(1, 2).contiguous(), proposal_offsets[:, 3:5, :]),
            dim=1)

        return estimation_cla_s, vote_xyz_s, estimation_boxs.transpose(1, 2).contiguous(), center_xyzs
