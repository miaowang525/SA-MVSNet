
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import ConvBnReLU, depth_regression
from .patchmatch import PatchMatch
from .attention import *

###############51net_49+ASFF3####################

class FeatureNet(nn.Module):
    """Feature Extraction Network: to extract features of original images from each view"""

    def __init__(self):
        """Initialize different layers in the network"""

        super(FeatureNet, self).__init__()

        # [B,8,H,W]
        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        # [B,16,H/2,W/2]
        self.conv4 = SwinTransformer3(in_chans=8, patch_size=2, window_size=8, mlp_ratio=4, embed_dim=16, depths=2, num_heads=4)
        # [B,32,H/4,W/4]
        self.conv7 = SwinTransformer3(in_chans=16, patch_size=2, window_size=8, mlp_ratio=4, embed_dim=32, depths=2, num_heads=4)
        # [B,64,H/8,W/8]
        self.conv10 = SwinTransformer3(in_chans=32, patch_size=2, window_size=8, mlp_ratio=4, embed_dim=64, depths=6, num_heads=8)

        self.psa3 = PSAModule(64, 64)


        self.output1 = nn.Conv2d(64, 64, 1, bias=False)
        self.output2 = nn.Conv2d(64, 64, 1, bias=False)
        self.output3 = nn.Conv2d(64, 64, 1, bias=False)

        self.inner0 = nn.Conv2d(32, 64, 1, bias=True)
        self.inner1 = nn.Conv2d(16, 64, 1, bias=True)

        self.asff0 = ASFF3(level=0)
        self.asff1 = ASFF3(level=1)
        self.asff2 = ASFF3(level=2)


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward method

        Args:
            x: images from a single view, in the shape of [B, C, H, W]. Generally, C=3

        Returns:
            output_feature: a python dictionary contains extracted features from stage_1 to stage_3
                keys are "stage_1", "stage_2", and "stage_3"
        """
        output_feature = {}

        #[1, 8, 512, 640]
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)

        #[1, 16, 256, 320]
        conv4 = self.conv4(conv1)

        #[1, 32, 128, 160]
        conv7 = self.conv7(conv4)

        #[1, 64, 64, 80]
        conv10 = self.conv10(conv7)

        intra_feat0 = conv10  # torch.Size([1, 64, 64, 80])

        intra_feat1 = F.interpolate(conv10, scale_factor=2, mode="bilinear") + (self.psa3(self.inner0(conv7)) + self.inner0(conv7))
        # torch.Size([1, 64, 128, 160])
        del conv10, conv7

        intra_feat2 = F.interpolate(intra_feat1, scale_factor=2, mode="bilinear") + (self.psa3(self.inner1(conv4)) + self.inner1(conv4))
        # torch.Size([1, 64, 256, 320])
        del conv4

        # intra_feat0 = self.output1(intra_feat0)
        # intra_feat1 = self.output2(intra_feat1)
        # intra_feat2 = self.output3(intra_feat2)

        stage3 = self.asff0(x_level_0=intra_feat0, x_level_1=intra_feat1, x_level_2=intra_feat2)
        stage2 = self.asff1(x_level_0=intra_feat0, x_level_1=intra_feat1, x_level_2=intra_feat2)
        stage1 = self.asff2(x_level_0=intra_feat0, x_level_1=intra_feat1, x_level_2=intra_feat2)

        output_feature["stage_3"] = stage3
        output_feature["stage_2"] = stage2
        output_feature["stage_1"] = stage1
        # output_feature["stage_3"] = self.output1(stage3)
        # output_feature["stage_2"] = self.output2(stage2)
        # output_feature["stage_1"] = self.output3(stage1)

        return output_feature




class Refinement(nn.Module):
    """Depth map refinement network"""

    def __init__(self):
        """Initialize"""

        super(Refinement, self).__init__()

        # img: [B,3,H,W]
        self.conv0 = ConvBnReLU(in_channels=3, out_channels=8)
        # depth map:[B,1,H/2,W/2]
        self.conv1 = ConvBnReLU(in_channels=1, out_channels=8)
        self.conv2 = ConvBnReLU(in_channels=8, out_channels=8)
        self.deconv = nn.ConvTranspose2d(
            in_channels=8, out_channels=8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
        )

        self.bn = nn.BatchNorm2d(8)
        self.conv3 = ConvBnReLU(in_channels=16, out_channels=8)
        self.res = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1, bias=False)

    def forward(
        self, img: torch.Tensor, depth_0: torch.Tensor, depth_min: torch.Tensor, depth_max: torch.Tensor
    ) -> torch.Tensor:
        """Forward method

        Args:
            img: input reference images (B, 3, H, W)
            depth_0: current depth map (B, 1, H//2, W//2)
            depth_min: pre-defined minimum depth (B, )
            depth_max: pre-defined maximum depth (B, )

        Returns:
            depth: refined depth map (B, 1, H, W)
        """

        batch_size = depth_min.size()[0] #B
        # pre-scale the depth map into [0,1]
        depth = (depth_0 - depth_min.view(batch_size, 1, 1, 1)) / (
            depth_max.view(batch_size, 1, 1, 1) - depth_min.view(batch_size, 1, 1, 1)
        ) #(B, 1, H//2, W//2)-(B, 1, 1, 1)

        conv0 = self.conv0(img)#(B, 8, H, W)
        deconv = F.relu(self.bn(self.deconv(self.conv2(self.conv1(depth)))), inplace=True)
        cat = torch.cat((deconv, conv0), dim=1)
        del deconv, conv0
        # depth residual
        res = self.res(self.conv3(cat))
        del cat

        depth = F.interpolate(depth, scale_factor=2, mode="nearest") + res
        # convert the normalized depth back
        depth = depth * (depth_max.view(batch_size, 1, 1, 1) - depth_min.view(batch_size, 1, 1, 1)) + depth_min.view(
            batch_size, 1, 1, 1
        )

        return depth


class PatchmatchNet(nn.Module):
    """ Implementation of complete structure of PatchmatchNet"""

    def __init__(
        self,
        patchmatch_interval_scale: List[float] = [0.005, 0.0125, 0.025],
        propagation_range: List[int] = [6, 4, 2],
        patchmatch_iteration: List[int] = [1, 2, 2],
        patchmatch_num_sample: List[int] = [8, 8, 16],
        propagate_neighbors: List[int] = [0, 8, 16],
        evaluate_neighbors: List[int] = [9, 9, 9],
    ) -> None:
        """Initialize modules in PatchmatchNet

        Args:
            patchmatch_interval_scale: depth interval scale in patchmatch module depth间隔scale
            propagation_range: propagation range 传播范围用在2D卷积中的dilation参数
            patchmatch_iteration: patchmatch interation number 在每个stage的迭代次数
            patchmatch_num_sample: patchmatch number of samples 深度采样层数
            propagate_neighbors: number of propagation neigbors 传播的邻域个数
            evaluate_neighbors: number of propagation neigbors for evaluation 代价聚合的邻域个数
        """
        super(PatchmatchNet, self).__init__()

        self.stages = 4
        self.feature = FeatureNet()
        self.patchmatch_num_sample = patchmatch_num_sample

        num_features = [8, 64, 64, 64]

        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        # number of groups for group-wise correlation
        self.G = [4, 8, 8]

        for i in range(self.stages - 1):#（0,1,2）   for i in range（3）

            if i == 2:
                patchmatch = PatchMatch(
                    random_initialization=True,
                    propagation_out_range=propagation_range[i],
                    patchmatch_iteration=patchmatch_iteration[i],
                    patchmatch_num_sample=patchmatch_num_sample[i],
                    patchmatch_interval_scale=patchmatch_interval_scale[i],
                    num_feature=num_features[i + 1],
                    G=self.G[i],
                    propagate_neighbors=self.propagate_neighbors[i],
                    stage=i + 1,
                    evaluate_neighbors=evaluate_neighbors[i],
                )
            else:
                patchmatch = PatchMatch(
                    random_initialization=False,
                    propagation_out_range=propagation_range[i],
                    patchmatch_iteration=patchmatch_iteration[i],
                    patchmatch_num_sample=patchmatch_num_sample[i],
                    patchmatch_interval_scale=patchmatch_interval_scale[i],
                    num_feature=num_features[i + 1],
                    G=self.G[i],
                    propagate_neighbors=self.propagate_neighbors[i],
                    stage=i + 1,
                    evaluate_neighbors=evaluate_neighbors[i],
                )
            setattr(self, f"patchmatch_{i+1}", patchmatch)

        self.upsample_net = Refinement()

    def forward(
        self,
        imgs: Dict[str, torch.Tensor],
        proj_matrices: Dict[str, torch.Tensor],
        depth_min: torch.Tensor,
        depth_max: torch.Tensor,
    ) -> Dict[str, Any]:
        """Forward method for PatchMatchNet

        Args:
            imgs: different stages of images (B, 3, H, W) stored in the dictionary
            proj_matrics: different stages of camera projection matrices (B, 4, 4) stored in the dictionary
            depth_min: minimum virtual depth (B, )
            depth_max: maximum virtual depth (B, )

        Returns:
            output dictionary of PatchMatchNet, containing refined depthmap, depth patchmatch
                and photometric_confidence.
        """
        imgs_0 = torch.unbind(imgs["stage_0"], 1)
        imgs_1 = torch.unbind(imgs["stage_1"], 1)
        imgs_2 = torch.unbind(imgs["stage_2"], 1)
        imgs_3 = torch.unbind(imgs["stage_3"], 1)
        del imgs

        self.imgs_0_ref = imgs_0[0]
        self.imgs_1_ref = imgs_1[0]
        self.imgs_2_ref = imgs_2[0]
        self.imgs_3_ref = imgs_3[0]
        del imgs_1, imgs_2, imgs_3

        self.proj_matrices_0 = torch.unbind(proj_matrices["stage_0"].float(), 1)
        self.proj_matrices_1 = torch.unbind(proj_matrices["stage_1"].float(), 1)
        self.proj_matrices_2 = torch.unbind(proj_matrices["stage_2"].float(), 1)
        self.proj_matrices_3 = torch.unbind(proj_matrices["stage_3"].float(), 1)
        del proj_matrices

        assert len(imgs_0) == len(self.proj_matrices_0), "Different number of images and projection matrices"

        # step 1. Multi-scale feature extraction
        # in: images; out: 32-channel feature maps
        # 第一步：特征提取 输入是Bx3xHxW 输出是[B,32,H,W]因为有上采样到原图大小 H/4 W/4
        features = []
        for img in imgs_0:
            output_feature = self.feature(img)
            features.append(output_feature)
        del imgs_0
        ref_feature, src_features = features[0], features[1:]

        depth_min = depth_min.float()
        depth_max = depth_max.float()

        # step 2. Learning-based patchmatch
        depth = None
        view_weights = None
        depth_patchmatch = {}
        refined_depth = {}

        for l in reversed(range(1, self.stages)):
            src_features_l = [src_fea[f"stage_{l}"] for src_fea in src_features]
            projs_l = getattr(self, f"proj_matrices_{l}")
            ref_proj, src_projs = projs_l[0], projs_l[1:]

            if l > 1:
                depth, _, view_weights = getattr(self, f"patchmatch_{l}")(
                    ref_feature=ref_feature[f"stage_{l}"],
                    src_features=src_features_l,
                    ref_proj=ref_proj,
                    src_projs=src_projs,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    depth=depth,
                    img=getattr(self, f"imgs_{l}_ref"),
                    view_weights=view_weights,
                )
            else:
                depth, score, _ = getattr(self, f"patchmatch_{l}")(
                    ref_feature=ref_feature[f"stage_{l}"],
                    src_features=src_features_l,
                    ref_proj=ref_proj,
                    src_projs=src_projs,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    depth=depth,
                    img=getattr(self, f"imgs_{l}_ref"),
                    view_weights=view_weights,
                )

            del src_features_l, ref_proj, src_projs, projs_l

            depth_patchmatch[f"stage_{l}"] = depth

            depth = depth[-1].detach()
            if l > 1:
                # upsampling the depth map and pixel-wise view weight for next stage
                depth = F.interpolate(depth, scale_factor=2, mode="nearest")
                view_weights = F.interpolate(view_weights, scale_factor=2, mode="nearest")

        # step 3. Refinement
        depth = self.upsample_net(self.imgs_0_ref, depth, depth_min, depth_max)
        refined_depth["stage_0"] = depth

        del depth, ref_feature, src_features

        if self.training:
            return {
                "refined_depth": refined_depth,
                "depth_patchmatch": depth_patchmatch,
            }

        else:
            num_depth = self.patchmatch_num_sample[0]
            score_sum4 = 4 * F.avg_pool3d(
                F.pad(score.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0
            ).squeeze(1)
            # [B, 1, H, W]
            depth_index = depth_regression(
                score, depth_values=torch.arange(num_depth, device=score.device, dtype=torch.float)
            ).long()
            depth_index = torch.clamp(depth_index, 0, num_depth - 1)
            photometric_confidence = torch.gather(score_sum4, 1, depth_index)
            photometric_confidence = F.interpolate(photometric_confidence, scale_factor=2, mode="nearest")
            photometric_confidence = photometric_confidence.squeeze(1)

            return {
                "refined_depth": refined_depth,
                "depth_patchmatch": depth_patchmatch,
                "photometric_confidence": photometric_confidence,
            }


def patchmatchnet_loss(
    depth_patchmatch: Dict[str, torch.Tensor],
    refined_depth: Dict[str, torch.Tensor],
    depth_gt: Dict[str, torch.Tensor],
    mask: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Patchmatch Net loss function

    Args:
        depth_patchmatch: depth map predicted by patchmatch net
        refined_depth: refined depth map predicted by patchmatch net
        depth_gt: ground truth depth map
        mask: mask for filter valid points

    Returns:
        loss: result loss value
    """
    stage = 4

    loss = 0
    for l in range(1, stage):
        depth_gt_l = depth_gt[f"stage_{l}"]
        mask_l = mask[f"stage_{l}"] > 0.5
        depth2 = depth_gt_l[mask_l]

        depth_patchmatch_l = depth_patchmatch[f"stage_{l}"]
        for i in range(len(depth_patchmatch_l)):
            depth1 = depth_patchmatch_l[i][mask_l]
            loss = loss + F.smooth_l1_loss(depth1, depth2, reduction="mean")

    l = 0
    depth_refined_l = refined_depth[f"stage_{l}"]
    depth_gt_l = depth_gt[f"stage_{l}"]
    mask_l = mask[f"stage_{l}"] > 0.5

    depth1 = depth_refined_l[mask_l]
    depth2 = depth_gt_l[mask_l]
    loss = loss + F.smooth_l1_loss(depth1, depth2, reduction="mean")

    return loss

