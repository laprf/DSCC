import torch
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_scatter import scatter_sum

from .UNet import UNet
from .VIT import ViT
from .utils import cal_coords, index_points, _normalize


class SpectralDerivative(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("wavelength", torch.tensor([466, 480, 500, 520, 536, 550, 566, 580,
                                        596, 610, 626, 640, 656, 670, 686, 700,
                                        716, 730, 746, 760, 776, 790, 806, 820,
                                        836, 850, 866, 880, 896, 910, 926, 940]))
        self.register_buffer("delta_n", self.wavelength[1:] - self.wavelength[:-1])

    def sdf(self, x):
        derivative = x.diff(dim=1)  # [B, D-1, H, W]
        if x.shape[1] == 32:
            return derivative / self.delta_n.view(1, -1, 1, 1)
        else:
            return derivative


class Clustering(nn.Module):
    def __init__(self, cfg, seman_channel, in_dim, hidden_dim, out_dim, pixel_coords, img_size):
        super(Clustering, self).__init__()
        self.k = cfg['k']
        self.cfg = cfg
        self.img_size = img_size

        self.f_center = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.f_center_out = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU())

        self.f_semantic = nn.Sequential(
            nn.Conv2d(seman_channel, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
        )

        self.f_hsi = nn.Sequential(
            nn.Conv2d(32, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
        )

        self.register_buffer("pixel_coords", pixel_coords)
        self.register_buffer("img_size_factor", torch.tensor(1.0 / img_size))

    def forward(self, fused_centers, hsi, sdf, semantic, hsi_centers, sdf_centers, semantic_centers, center_coords):
        fused_centers = self.f_center(fused_centers)
        semantic_in = self.f_semantic(semantic)  # [B, D, H, W]

        H, W = hsi.shape[-2:]
        B, M = hsi_centers.shape[:2]

        dist = torch.cdist(center_coords, self.pixel_coords.unsqueeze(0).expand(B, -1, -1), p=2.0) * self.img_size_factor # [B,M,N] N=H*W

        dist_mask = torch.zeros_like(dist)  # [B,M,N]
        top_indices = dist.sort(dim=1)[1][:, :self.k, :]  # [B, 9, N]
        dist_mask.scatter_(1, top_indices, 1.0)  # [B,M,N]

        if self.cfg['hsi']:
            dist = dist + torch.cdist(hsi_centers, hsi.reshape(B, hsi.shape[1], -1).permute(0, 2, 1), p=2.0) / (hsi.shape[1] ** 0.5)  # [B,M,N]
        if self.cfg['sdf']:
            dist = dist + torch.cdist(sdf_centers, sdf.reshape(B, sdf.shape[1], -1).permute(0, 2, 1), p=2.0) / (sdf.shape[1] ** 0.5)
        if self.cfg['semantic']:
            dist = dist + torch.cdist(semantic_centers, semantic.reshape(B, semantic.shape[1], -1).permute(0, 2, 1), p=2) / (semantic.shape[1] ** 0.5)  # [B,M,N]
        sim = torch.exp(-dist) * dist_mask

        # we use mask to solely assign each point to one center
        _, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary # [B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)

        # aggregate step, out shape [B,M,C]
        semantic_in = rearrange(semantic_in, 'b c w h -> (b w h) c')  # [B,N,D]
        sim_max_idx = rearrange(sim_max_idx.squeeze(1), 'b n -> (b n)')
        idx_offset = (torch.arange(B, device=sim_max_idx.device) * M).unsqueeze(-1).expand(-1, H * W).flatten()
        sim_max_idx = sim_max_idx + idx_offset
        out = rearrange(scatter_sum(semantic_in, sim_max_idx, dim=0, dim_size=B * M), '(b m) c -> b m c', b=B, m=M)
        out = (out + fused_centers) / (mask.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]

        return self.f_center_out(out), rearrange(mask, "b m (w h) -> b m w h", w=W, h=H)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Cluster_Block(nn.Module):
    def __init__(self, cfg, seman_channel, dim, hidden_dim, pixel_coords, mlp_ratio, drop_path=0., use_layer_scale=False,
                 layer_scale_init_value=1e-5, img_size=256):
        super(Cluster_Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixer = Clustering(cfg, seman_channel, dim, hidden_dim, dim, pixel_coords, img_size)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, drop=cfg['drop_rate'])

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        centers, hsi, sdf, semantic, _, hsi_center, sdf_center, semantic_center, center_coords = x
        centers = self.norm1(centers)
        centers_feat, mask = self.token_mixer(centers, hsi, sdf, semantic, hsi_center, sdf_center, semantic_center,
                                               center_coords)

        if self.use_layer_scale:
            centers = centers + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * centers_feat)
            centers = centers + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(self.norm2(centers)))
        else:
            centers = centers + self.drop_path(centers_feat)
            centers = centers + self.drop_path(self.mlp(self.norm2(centers)))
        return centers_feat, hsi, sdf, semantic, mask, hsi_center, sdf_center, semantic_center, center_coords


class DICF(nn.Module):
    def __init__(self, cfg, cluster_num, embed_dim, dim_out, img_size):
        super().__init__()
        self.cfg = cfg
        self.cluster_num = cluster_num
        self.dim_out = dim_out
        self.linear = nn.Linear(embed_dim, dim_out)
        self.norm = nn.LayerNorm(self.dim_out)
        self.img_size = img_size
        self.k = cfg['k']

        self.hsi_center_linear = nn.Sequential(nn.Linear(32, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.sdf_center_linear = nn.Sequential(nn.Linear(31, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.feat_center_linear = nn.Sequential(nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())

        self.register_buffer("img_size_factor", torch.tensor(1.0 / img_size))

    def forward(self, cal_center, hsi_center, sdf_center, feat_center, center_coords):
        cal_center = self.norm(self.linear(cal_center))

        cluster_num = self.cluster_num
        idx_cluster, cluster_num, new_center_idx = self.cluster_dpc_knn(
            cal_center, hsi_center, sdf_center, feat_center,
            cluster_num, center_coords, self.k
        )

        new_centers = cal_center.gather(1, new_center_idx.unsqueeze(-1).expand(-1, -1, cal_center.size(-1)))

        return_loss = 0
        if self.training:
            return_loss = self.compute_inter_loss(new_centers)

        return new_centers, new_center_idx, return_loss

    def cluster_dpc_knn(self, cal_center, hsi_center, sdf_center, feat_center, cluster_num, center_coords, k):
        B, N, C = cal_center.shape

        dist_matrix = torch.cdist(center_coords, center_coords) * self.img_size_factor  # [B, N, N]
        if self.cfg['hsi']:
            hsi_center = self.hsi_center_linear(hsi_center)
            hsi_dist_mat = torch.cdist(hsi_center, hsi_center) / (C ** 0.5)  # [B, N, N]
            dist_matrix = dist_matrix + hsi_dist_mat
        if self.cfg['sdf']:
            sdf_center = self.sdf_center_linear(sdf_center)
            sdf_dist_mat = torch.cdist(sdf_center, sdf_center) / (C ** 0.5)  # [B, N, N]
            dist_matrix = dist_matrix + sdf_dist_mat
        if self.cfg['semantic']:
            feat_center = self.feat_center_linear(feat_center)
            feat_dist_matrix = torch.cdist(feat_center, feat_center) / (C ** 0.5)  # [B, N, N]
            dist_matrix = dist_matrix + feat_dist_matrix

        with torch.no_grad():
            dist_nearest = dist_matrix.sort(dim=-1)[0][:, :, :k]  # [B, N, k]

            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()  # [B, N]
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(
                density.shape, device=density.device, dtype=density.dtype) * 1e-6

            # get distance indicator
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(cal_center.dtype)  # [B, N, N]
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)  # [B, N]

            # select clustering center according to score
            score = dist * density
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)

            # assign tokens to the nearest center
            dist_matrix = index_points(dist_matrix, index_down)  # [B, cls_num, N]
            idx_cluster = dist_matrix.argmin(dim=1)  # [B, N]

            # make sure cluster center merge to itself
            idx_batch = torch.arange(B, device=cal_center.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=cal_center.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        return idx_cluster, cluster_num, index_down

    @staticmethod
    def compute_inter_loss(cluster_centers):
        B, cluster_num, C = cluster_centers.shape
        all_centers = cluster_centers.view(B * cluster_num, C)
        dist_matrix = torch.cdist(all_centers, all_centers)
        mask = ~torch.eye(B * cluster_num, dtype=bool, device=dist_matrix.device)
        valid_dist = dist_matrix[mask].view(B * cluster_num, -1)
        avg_inter_distance = valid_dist.mean()
        inter_loss = 1.0 / (avg_inter_distance + 1e-6)
        return inter_loss


def basic_blocks(cfg, seman_channel, dim, index, layers, pixel_coords, img_size):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = cfg['drop_path_rate'] * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(Cluster_Block(cfg, seman_channel, dim, cfg['hidden_dims'][index],
                                    pixel_coords, cfg['mlp_ratios'][index], block_dpr, cfg['layer_scale'], img_size))
    return nn.Sequential(*blocks)


class DSCC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # init
        layers = cfg['Clustering']['layers']
        embed_dims = cfg['Clustering']['embed_dims']
        remain_tokens = cfg['Clustering']['remain_tokens']

        self.classes = cfg['num_classes']
        self.cfg = cfg

        # UNet and sdf
        self.UNet = UNet(n_channels=cfg['in_channels'], n_classes=embed_dims[0])
        self.sdf_op = SpectralDerivative()

        # cluster center init
        self.centers_proposal = nn.AdaptiveAvgPool2d((cfg['Clustering']['proposal'], cfg['Clustering']['proposal']))
        center_coords, pixel_coords = cal_coords(cfg['Clustering']['proposal'], cfg['img_size'])
        self.register_buffer("center_coords", center_coords)

        # cluster backbone
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(cfg['Clustering'], embed_dims[0], embed_dims[i], i, layers,
                                 pixel_coords, cfg['img_size'])
            network.append(stage)
            if i < len(layers) - 1:
                network.append(DICF(cfg=cfg['TokenMerger'], cluster_num=remain_tokens[i], embed_dim=embed_dims[i],
                                   dim_out=embed_dims[i + 1], img_size=cfg['img_size']))
        self.cluster = nn.ModuleList(network)

        # vit classification
        self.transformer = ViT(image_size=cfg['img_size'], patch_size=cfg['ViT']['patch_size'], num_classes=cfg['num_classes'],
                       dim=cfg['ViT']['hidden_size'], depth=cfg['ViT']['depth'], heads=cfg['ViT']['num_heads'],
                       mlp_dim=cfg['ViT']['mlp_dim'], channels=embed_dims[-1], dim_head=cfg['ViT']['dim_head'],
                       dropout=cfg['ViT']['dropout'], emb_dropout=cfg['ViT']['attention_dropout'], use_mamba=cfg['ViT']['Mamba'])


    def forward(self, hsi, gt):
        """
            img: [B, D, H, W]
            gt: [B, H, W]
        """
        # ---- sdf and semantic feature ----
        sdf = self.sdf_op.sdf(hsi)  # [B, D-1, H, W]
        img_feat = self.UNet(hsi)  # [B, D, H, W]

        # ---- init cluster centers ----
        hsi_centers = self.centers_proposal(hsi).flatten(2).permute(0, 2, 1)  # [B, p*p, D]
        sdf_centers = self.centers_proposal(sdf).flatten(2).permute(0, 2, 1)  # [B, p*p, D]
        feat_centers = self.centers_proposal(img_feat).flatten(2).permute(0, 2, 1)  # [B, p*p, C]


        sdf_centers = _normalize(sdf_centers)

        # ---- cluster ----
        centers, masks, merge_loss = self.forward_tokens(hsi, sdf, img_feat, hsi_centers, sdf_centers, feat_centers)

        # ----vit classification----
        vit_out = self.transformer(centers)
        if self.training:
            labels = self.gen_labels(gt, masks[-1])
            return vit_out, labels, merge_loss

        spix_map = self.gen_spix_map(masks[-1])
        return vit_out, spix_map

    def forward_tokens(self, hsi, sdf, semantic, hsi_centers, sdf_centers, feat_centers):
        B, N, C = feat_centers.shape
        cal_centers = feat_centers.detach().clone()
        masks, merge_losses = [], []
        center_coords = self.center_coords[None, :, :].repeat(B, 1, 1).to(feat_centers.device)
        for idx, layer in enumerate(self.cluster):
            if isinstance(layer, nn.Sequential):
                outputs = layer(
                    (cal_centers, hsi, sdf, semantic, None, hsi_centers, sdf_centers, feat_centers, center_coords))
                cal_centers, mask = outputs[0], outputs[4]
                masks.append(mask)
            else:
                cal_centers, center_idx, merge_loss = layer(cal_centers, hsi_centers, sdf_centers, feat_centers, center_coords)
                hsi_centers = index_points(hsi_centers, center_idx)
                sdf_centers = index_points(sdf_centers, center_idx)
                feat_centers = index_points(feat_centers, center_idx)
                center_coords = index_points(center_coords, center_idx)
                merge_losses.append(merge_loss)
        return cal_centers, masks, sum(merge_losses) / len(merge_losses)

    def gen_labels(self, gt, mask):
        B, M = mask.shape[:2]
        gt_filt = (gt + 1).unsqueeze(1) * mask  # Shape: [b, M, w, h]
        gt_filt = gt_filt.flatten(start_dim=2).long()  # Shape: [b, M, w*h]

        count = torch.zeros((B, M, self.classes + 1), dtype=torch.long,
                            device=gt.device)  # Shape: [b, M, classes + 1]
        count.scatter_add_(dim=2, index=gt_filt, src=torch.ones_like(gt_filt))
        if self.cfg['soft_label']:
            total_pixels = torch.clamp(count[:, :, 1:].sum(dim=2, keepdim=True), min=1)
            label = (count[:, :, 1:] / total_pixels).permute(0, 2, 1).float()
        else:
            label = torch.argmax(count[..., 1:], dim=2)  # Shape: [b, M]
        return label

    @staticmethod
    def gen_spix_map(mask):
        M = mask.shape[1]
        coef = torch.arange(0, M, dtype=mask.dtype, device=mask.device).view(1, M, 1, 1)
        small_imgs = (mask * coef).sum(dim=1)  # Shape: [b, f*f, w, h]
        bias = torch.arange(0, M, step=M, dtype=mask.dtype, device=mask.device)
        spix_map = small_imgs + bias  # Shape: [b, w, h]
        return spix_map


def model_init(cfg, args=None):
    model = DSCC(cfg)
    if args is None:
        return model, [cfg['Clustering']['proposal'] ** 2] + cfg['Clustering']['remain_tokens']
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch_num, eta_min=0.1 * args.lr)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        return model, optimizer, criterion, scheduler
