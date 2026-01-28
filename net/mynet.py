import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg
from .modules import *
from .blocks.select_sobel import SEA


class SEA_CDModule(nn.Module):

    def __init__(self, inch):
        """
        参数
        ----
        inch : list[int]
            与输入金字塔各层通道数对应的列表（例如 [2, 2, 2]），即每层 4D 相关张量的 C。
        """
        super().__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):

            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3, outch4 = 16, 32, 64, 128

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])
        self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [4, 2, 2])
        self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [4, 4, 2])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

        # Decoder layers
        self.decoder1 = nn.Sequential(
            nn.Conv2d(outch4, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
            nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU())

        self.decoder2 = nn.Sequential(
            nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
            nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True))

        self.SelectiveEdgeAttention = SEA(dim=64)
        
        pass
    
    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def forward(self, hypercorr_pyramid, query_mask=None):
        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])

        # Propagate encoded 4D-tensor (Mixing building blocks)
        hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)
        # print("query_mask",query_mask)
        
        if query_mask is not None:

            _hypercorr_encoded_ = self.SelectiveEdgeAttention(hypercorr_encoded)         #! SSblock
            hypercorr_encoded = torch.concat([hypercorr_encoded, _hypercorr_encoded_], dim=1)
        else:
            hypercorr_encoded = torch.concat([hypercorr_encoded, hypercorr_encoded], dim=1)
            pass

        hypercorr_decoded = self.decoder1(hypercorr_encoded)
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size,
                                          mode='bilinear', align_corners=True)
        logit = self.decoder2(hypercorr_decoded)
        return logit

    pass

class SegmentationHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.mgcd = SEA_CDModule([2, 2, 2])
        # if test with our .pth file, maybe need to add the following lines
        # self.alpha = nn.Parameter(torch.tensor(0.0))  
        # self.beta = nn.Parameter(torch.tensor(0.0))
        pass

    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
            
        label = support_label.unsqueeze(1)  # [B,1,H0,W0]
        feat = support_feats[2]  # shape: [16, 2048, 8, 8] 高维信息作为先验输入
            
        proto_fg = masked_avg_pool(feat, label)            # 前景 prototype: [16, 2048]
        proto_bg = masked_avg_pool(feat, 1 - label)        # 背景 prototype: [16, 2048]
        
        support_prototypes_fg = proto_fg
        support_prototypes_bg = proto_bg
            
        query_feat_2 = query_feats[2]  # 取高维特征
        support_feat_2 = support_feats[2]  
            
        alpha = 0.5
        beta = 1.0 - alpha

        prior_fg, prior_bg = compute_query_prior(query_feat_2, support_prototypes_fg, support_prototypes_bg, temperature=1.1)
        prior = torch.sigmoid(prior_fg - prior_bg)
            
        query_prototypes_fg = get_query_foreground_prototype(query_feat_2, prior)  # [B, C]
        prototype_fg = alpha * query_prototypes_fg + beta * support_prototypes_fg

        prototype_fg = prototype_fg.unsqueeze(-1).unsqueeze(-1)   # [16, 4096, 1, 1]
        prototype_fg = prototype_fg.expand(-1, -1, 8, 8)    
            
        support_feats[2]= support_feats[2] + prototype_fg
        query_feats[2] = query_feats[2] + prototype_fg
            
        #!#################FGE ENDDING###################

        support_feats_fg = [self.label_feature(
            support_feat, support_label.clone())for support_feat in support_feats]
        support_feats_bg = [self.label_feature(
            support_feat, (1 - support_label).clone())for support_feat in support_feats]
            
        corr_fg = self.multilayer_correlation(query_feats, support_feats_fg)
        corr_bg = self.multilayer_correlation(query_feats, support_feats_bg)
        corr = [torch.concatenate([fg_one[:, None], bg_one[:, None]],
                                    dim=1) for fg_one, bg_one in zip(corr_fg, corr_bg)]

        logit = self.mgcd(corr[::-1], query_mask)
        return logit

    @staticmethod
    def label_feature(feature, label):
        label = F.interpolate(label.unsqueeze(1).float(), feature.size()[2:],
                             mode='bilinear', align_corners=True)
        return feature * label

    @staticmethod
    def multilayer_correlation(query_feats, support_feats):
        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + 1e-5)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + 1e-5)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)
            corr = corr.clamp(min=0)
            corrs.append(corr)
            pass

        return corrs

    pass   

class APSENetwork(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone_type = args.backbone
        self.finetune_backbone = args.finetune_backbone if hasattr(args, "finetune_backbone") else False
        self.fold = args.fold
        if "vgg" in self.backbone_type:
            self.backbone = vgg.vgg16(pretrained=True)
            self.extract_feats = self.extract_feats_vgg
        elif "50" in self.backbone_type:
            self.backbone = resnet.resnet50(pretrained=True)  # 先加载 ImageNet 预训练
            self.extract_feats = self.extract_feats_res
        else:
            self.backbone = resnet.resnet101(pretrained=True)
            self.extract_feats = self.extract_feats_res
            pass

        self.segmentation_head = SegmentationHead()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        pass

    def forward(self, query_img, support_img, support_label, query_mask, support_masks):
        if self.finetune_backbone:
            query_feats = self.extract_feats(query_img, self.backbone)
            support_feats = self.extract_feats(support_img, self.backbone)
        else:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img, self.backbone)
                support_feats = self.extract_feats(support_img, self.backbone)
                pass
            pass

        logit = self.segmentation_head(query_feats, support_feats, support_label.clone(), query_mask, support_masks)
        logit = F.interpolate(logit, support_img.size()[2:], mode='bilinear', align_corners=True)
        return logit

    def predict_nshot(self, batch):
        nshot = batch["support_imgs"].shape[1]
        logit_label_agg = 0
        for s_idx in range(nshot):
            logit_label = self.forward(
                batch['query_img'], batch['support_imgs'][:, s_idx],  batch['support_labels'][:, s_idx],
                query_mask=batch['query_mask'] if 'query_mask' in batch and self.args.mask else None,
                support_masks=batch['support_masks'][:, s_idx] if 'support_masks' in batch and self.args.mask else None)

            result_i = logit_label.argmax(dim=1).clone()
            logit_label_agg += result_i

            # One-Shot
            if nshot == 1: return result_i.float()
            pass

        # Few-Shot
        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_label_agg.size(0)
        max_vote = logit_label_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_label = logit_label_agg.float() / max_vote
        threshold = 0.4
        pred_label[pred_label < threshold] = 0
        pred_label[pred_label >= threshold] = 1
        return pred_label

    def compute_objective(self, logit_label, gt_label):
        bsz = logit_label.size(0)
        logit_label = logit_label.view(bsz, 2, -1)
        gt_label = gt_label.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_label, gt_label)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
        pass

    @staticmethod
    def extract_feats_vgg(img, backbone):
        feat_ids = [16, 23, 30]
        feats = []
        feat = img
        for lid, module in enumerate(backbone.features):
            feat = module(feat)
            if lid in feat_ids:
                feats.append(feat.clone())
        return feats

    @staticmethod
    def extract_feats_res(img, backbone):
        x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(img))))

        feats = []
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        feats.append(x.clone())
        x = backbone.layer3(x)
        feats.append(x.clone())
        x = backbone.layer4(x)
        feats.append(x.clone())
        return feats

    pass