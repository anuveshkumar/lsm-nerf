import torch.nn as nn
import torch
import torch.nn.functional as F
from models.embedder import get_embedder


class RendererV1(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, input_ch_feat=8, skips=[4],
                 use_viewdirs=False):
        super(RendererV1, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.in_ch_pts, self.in_ch_views, self.in_ch_feat = input_ch, input_ch_views, input_ch_feat

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.in_ch_pts, W, bias=True)] + [
                nn.Linear(W, W, bias=True) if i not in self.skips else nn.Linear(W + self.in_ch_pts, W) for i in
                range(D - 1)])
        self.pts_bias = nn.Linear(self.in_ch_feat, W)
        self.view_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.pts_linears.apply(weights_init)
        self.view_linears.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.alpha_linear.apply(weights_init)
        self.rgb_linear.apply(weights_init)

    def forward(self, x):
        dim = x.shape[-1]
        in_ch_feat = dim - self.in_ch_pts - self.in_ch_views
        input_pts, input_feats, input_views = torch.split(x, [self.in_ch_pts, in_ch_feat, self.in_ch_views], dim=-1)

        h = input_pts
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) * bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l, in enumerate(self.view_linears):
                h = self.view_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


class Nerf(nn.Module):
    def __init__(self, D=8, W=256, input_ch_feat=8, skips=[4], net_type='v0',
                 pts_embedder=True, dir_embedder=True, device="cuda:0"):
        super(Nerf, self).__init__()

        if pts_embedder:
            self.embed_fn, input_ch_pts = get_embedder(multires=10, i=0, input_dims=3)
        else:
            self.embed_fn, input_ch_pts = None, 3

        if dir_embedder:
            self.embeddirs_fn, input_ch_views = get_embedder(multires=4, i=0, input_dims=3)
        else:
            self.embeddirs_fn, input_ch_views = None, 3

        self.nerf = RendererV1(D=D, W=W, input_ch_feat=input_ch_feat,
                               input_ch=input_ch_pts, output_ch=4, skips=skips,
                               input_ch_views=input_ch_views, use_viewdirs=True).to(device)

    def forward(self, pts, viewdirs, alpha_feat):
        if self.embed_fn is not None:
            pts = self.embed_fn(pts)

        if alpha_feat is not None:
            pts = torch.cat((pts, alpha_feat), dim=-1)

        if viewdirs is not None:
            if viewdirs.dim() != 3:
                viewdirs = viewdirs[:, None].expand(-1, pts.shape[1], -1)

            if self.embeddirs_fn is not None:
                viewdirs = self.embeddirs_fn(viewdirs)
            pts = torch.cat((pts, viewdirs), dim=-1)

        RGBA = self.nerf(pts)
        return RGBA
