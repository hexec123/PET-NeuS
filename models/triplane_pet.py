import torch
import torch.nn as nn
import numpy as np

from third_party.ops import bias_act
from third_party.ops import grid_sample as grid_sample_mod

from models.swin_transformer import WindowAttention, window_partition, window_reverse
from timm.models.layers import to_2tuple

from models.embedder import get_embedder


# ---------------------------
# Helpers for tri-plane axes
# ---------------------------

def generate_planes(device: torch.device | None = None) -> torch.Tensor:
    """Return the 3 canonical plane axes as a (3, 3, 3) tensor on the given device."""
    planes = torch.tensor(
        [
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]],

            [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]],

            [[0, 0, 1],
             [1, 0, 0],
             [0, 1, 0]],
        ],
        dtype=torch.float32,
    )
    return planes.to(device) if device is not None else planes


def project_onto_planes(planes: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """
    Project 3D coordinates (N, M, 3) onto each plane in `planes` (P, 3, 3).
    Returns (N*P, M, 2) of planar coordinates.
    """
    device = coordinates.device
    planes = planes.to(device)

    N, M, _ = coordinates.shape
    P = planes.shape[0]

    coords_exp = coordinates.unsqueeze(1).expand(-1, P, -1, -1).reshape(N * P, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N * P, 3, 3).to(device)
    projections = torch.bmm(coords_exp, inv_planes)  # (N*P, M, 3)
    return projections[..., :2]


def sample_from_planes(
    plane_axes: torch.Tensor,
    plane_features: torch.Tensor,
    coordinates: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    box_warp: float | None = None,
):
    """
    plane_features: (N, P, C, H, W)
    coordinates:    (N, M, 3) in [-box_warp/2, box_warp/2] coords (or normalized if you pass box_warp)
    returns:        (N, P, M, C)
    """
    assert padding_mode == 'zeros', "Only zeros padding is supported by this wrapper."
    device = plane_features.device
    plane_axes = plane_axes.to(device)
    coordinates = coordinates.to(device)

    N, P, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape

    if box_warp is None or box_warp == 0:
        raise ValueError("`box_warp` must be a positive float to normalize coordinates for grid sampling.")
    # normalize to [-1, 1] space for grid_sample
    norm_coords = (2.0 / box_warp) * coordinates  # (N, M, 3)

    # Project to per-plane 2D coords
    proj = project_onto_planes(plane_axes, norm_coords).unsqueeze(1)  # (N*P, 1, M, 2)

    # Prepare features & sample
    feats = plane_features.view(N * P, C, H, W)
    out = grid_sample_mod.grid_sample_2d(feats, proj.float())  # (N*P, C, 1, M)
    out = out.permute(0, 3, 2, 1).reshape(N, P, M, C)  # (N, P, M, C)
    return out


# ---------------------------
# TriPlane Generator
# ---------------------------

class TriPlaneGenerator(nn.Module):
    def __init__(
        self,
        img_resolution,             # Output resolution.
        img_channels,               # Number of output channels per plane.
        rendering_kwargs={},
        triplane_sdf={},
        triplane_sdf_ini={},
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.rendering_kwargs = rendering_kwargs
        self.progress = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.tritype = 0

        # Embedding MLP to initialize planes
        self.sdf_para = SDFNetwork(**triplane_sdf_ini)
        # Decoder that consumes sampled features + PE(x)
        self.decoder = OSG_PE_SDFNetwork(
            **triplane_sdf,
            multires=self.rendering_kwargs['PE_res'],
            geometric_init=self.rendering_kwargs['is_dec_geoinit'],
        )

        self._last_planes = None
        self.plane_axes = generate_planes()  # moved to correct device in forward

        # Initialize tri-planes with analytic SDF prior
        ini_sdf = torch.randn(3, self.img_channels, self.img_resolution, self.img_resolution)

        # Build regular grids in [-1, 1] for initialization
        N = self.img_resolution
        xs = (torch.arange(N) - (N / 2 - 0.5)) / (N / 2 - 0.5)
        ys = (torch.arange(N) - (N / 2 - 0.5)) / (N / 2 - 0.5)
        ys, xs = torch.meshgrid(-ys, xs, indexing='ij')
        zs = torch.zeros_like(xs)

        # Three orthogonal planes
        inputx = torch.stack([zs, xs, ys], dim=-1).reshape(N * N, 3)
        inputy = torch.stack([xs, zs, ys], dim=-1).reshape(N * N, 3)
        inputz = torch.stack([xs, ys, zs], dim=-1).reshape(N * N, 3)
        ini_sdf[0] = self.sdf_para(inputx).permute(1, 0).reshape(self.img_channels, N, N)
        ini_sdf[1] = self.sdf_para(inputy).permute(1, 0).reshape(self.img_channels, N, N)
        ini_sdf[2] = self.sdf_para(inputz).permute(1, 0).reshape(self.img_channels, N, N)

        self.planes = nn.Parameter(ini_sdf.unsqueeze(0), requires_grad=True)

        # Attention setup
        self.window_size = int(self.rendering_kwargs['attention_window_size'])
        self.numheads = int(self.rendering_kwargs['attention_numheads'])
        self.attn  = WindowAttention(self.img_channels, window_size=to_2tuple(self.window_size),  num_heads=self.numheads)
        self.window_size4 = self.window_size * 2
        self.attn4 = WindowAttention(self.img_channels, window_size=to_2tuple(self.window_size4), num_heads=self.numheads)
        self.window_size2 = max(1, self.window_size // 2)
        self.attn2 = WindowAttention(self.img_channels, window_size=to_2tuple(self.window_size2), num_heads=self.numheads)

    # ---- DP-safe forward ----
    def forward(self, coordinates: torch.Tensor, directions: torch.Tensor | None = None):
        """
        coordinates: (M, 3) in world space (assumed to match renderer’s convention)
        directions:  (M, 3) or None
        """
        device = self.planes.device
        planes = self.planes.view(len(self.planes), 3, self.img_channels, self.img_resolution, self.img_resolution).to(device)
        coords = coordinates.to(device).unsqueeze(0)  # (1, M, 3)
        dirs = directions.to(device) if directions is not None else None
        return self.run_model(planes, self.decoder, coords, dirs, self.rendering_kwargs)

    # ---- Main model path ----
    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options):
        device = planes.device
        img_channels = self.img_channels
        plane_axes = self.plane_axes.to(device)

        # 1) Sample base features from tri-planes
        sampled_features = sample_from_planes(
            plane_axes, planes, sample_coordinates,
            padding_mode='zeros',
            box_warp=options['box_warp'],
        )  # (1, P, M, C)

        # 2) Multi-scale window attention on planes (skip if not divisible)
        def maybe_attend(_planes, wsize, attn_mod):
            _, P, C, H, W = _planes.shape
            if H % wsize != 0 or W % wsize != 0:
                return None
            pa = _planes.squeeze(0).view(P, C, H, W).permute(0, 2, 3, 1)  # (P,H,W,C)
            xw = window_partition(pa, wsize)  # (num_windows*P, wsize, wsize, C)
            xw = xw.view(-1, wsize * wsize, C)
            aw = attn_mod(xw)
            aw = aw.view(-1, wsize, wsize, C)
            shifted = window_reverse(aw, wsize, H, W)  # (P,H,W,C)
            pa2 = shifted.permute(0, 3, 1, 2).unsqueeze(0)  # (1,P,C,H,W)
            return pa2

        planes_att = maybe_attend(planes, self.window_size,  self.attn)
        planes_att4 = maybe_attend(planes, self.window_size4, self.attn4)
        planes_att2 = maybe_attend(planes, self.window_size2, self.attn2)

        def sample_if_not_none(_p):
            if _p is None:
                return None
            return sample_from_planes(plane_axes, _p, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])

        s_att  = sample_if_not_none(planes_att)
        s_att4 = sample_if_not_none(planes_att4)
        s_att2 = sample_if_not_none(planes_att2)

        feats = [x for x in [s_att4, s_att, s_att2, sampled_features] if x is not None]
        sampled_features = torch.cat(feats, dim=-1)  # (1, P, M, C*)

        # 3) Positional encoding (BARF-style mixing)
        periodic_fns = [torch.sin, torch.cos]
        embed_fn, _ = get_embedder(options['multiply_PE_res'], input_dims=3, periodic_fns=periodic_fns)
        sample_PE = embed_fn(sample_coordinates.to(device))  # (1, M, D)

        inputs = sample_PE
        # reshape PE and tile across planes to match concatenated channels
        d = sampled_features.shape[-1] // (inputs.shape[-1] // 3)
        # (1, M, 4, multiply_PE_res//4*2, 3) slicing per axis
        pe_view = inputs.view(1, -1, 4, options['multiply_PE_res'] // 4 * 2, 3)
        x = pe_view[:, :, :, :, 0]
        y = pe_view[:, :, :, :, 1]
        z = pe_view[:, :, :, :, 2]
        inputs = torch.cat([z, x, y], dim=-1).tile(1, 1, d).view(3, inputs.shape[1], -1).to(device)

        sampled_features = sampled_features * inputs.unsqueeze(0)  # (1, P, M, C*)
        _, dim, N, nf = sampled_features.shape

        # 4) Decode
        out = decoder(
            sampled_features,             # (1, P, M, C*)
            sample_coordinates.to(device),  # inputs_PE
            sample_directions.to(device) if sample_directions is not None else None
        )
        return out

    # API expected by renderer
    def sdf(self, coordinates):
        return self.forward(coordinates)[:, :1]

    def gradient(self, x):
        x = x.requires_grad_(True)
        y = self.sdf(x)
        if isinstance(y, tuple):
            y = y[0]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        return gradients.unsqueeze(1)

    def grid_sampling(self, img_resolution):
        xs = (torch.arange(img_resolution) - (img_resolution / 2 - 0.5)) / (img_resolution / 2 - 0.5)
        ys = (torch.arange(img_resolution) - (img_resolution / 2 - 0.5)) / (img_resolution / 2 - 0.5)
        ys, xs = torch.meshgrid(ys, xs, indexing='ij')
        index = torch.stack([ys, xs]).permute(1, 2, 0).unsqueeze(0).tile(3, 1, 1, 1)
        return index


# ---------------------------
# MLPs
# ---------------------------

class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation='linear',
        lr_multiplier=8,
        bias_init=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = nn.Parameter(
            (torch.ones([out_features, in_features]) + torch.randn([out_features, in_features]) * 0.01)
            / lr_multiplier / self.in_features
        )
        self.bias = nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


class FullyConnectedLayer2(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation='linear',
        lr_multiplier=8,
        bias_init=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


class SDFNetwork(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=(4,),
        multires=0,
        bias=0.5,
        scale=1,
        geometric_init=True,
        weight_norm=True,
        inside_outside=False
    ):
        super().__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.embed_fn_fine = None
        self.multires = multires
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1] - (dims[0] if (l + 1) in self.skip_in else 0)
            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if multires > 0 and l == 0:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.constant_(lin.weight[:, 3:], 0.0)
                    nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, f"lin{l}", lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, f"lin{l}")
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 1:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)


class OSG_PE_SDFNetwork(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=(10,),
        multires=0,
        bias=0.5,
        scale=1,
        geometric_init=True,
        weight_norm=True,
        inside_outside=False
    ):
        super().__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.embed_fn_fine = None
        self.multires = multires
        self.progress = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        d_PE = 3
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_PE)
            self.embed_fn_fine = embed_fn
            self.num_eoc = int((input_ch - d_PE) / 2)
            dims[0] = d_in + self.num_eoc + d_PE
        else:
            dims[0] = d_in + d_PE

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1] - (dims[0] if (l + 1) in self.skip_in else 0)
            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        nn.init.constant_(lin.bias, -bias)
                    else:
                        nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.constant_(lin.weight[:, 3:], 0.0)
                    nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, f"lin{l}", lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs, inputs_PE, sample_directions):
        # inputs: (1, P, M, C*)
        _, dim, N, nf = inputs.shape
        x = inputs.squeeze(0).permute(1, 2, 0).reshape(N, nf * dim)  # (N, nf*P)
        inputs_PE = inputs_PE.squeeze(0)
        x = x * self.scale

        if self.embed_fn_fine is not None:
            input_enc = self.embed_fn_fine(inputs_PE)
            nfea_eachband = int(input_enc.shape[1] / self.multires)
            Nbands = int(self.multires / 2)

            inputs_enc, weight = coarse2fine(0.5 * (self.progress.data - 0.1), input_enc, self.multires)
            inputs_enc = inputs_enc.view(-1, self.multires, nfea_eachband)[:, :Nbands, :].view([-1, self.num_eoc])

            input_enc = input_enc.view(-1, self.multires, nfea_eachband)[:, :Nbands, :].view([-1, self.num_eoc]).contiguous()
            input_enc = (input_enc.view(-1, Nbands) * weight[:Nbands]).view([-1, self.num_eoc])

            flag = weight[:Nbands].tile(input_enc.shape[0], nfea_eachband, 1).transpose(1, 2).contiguous().view([-1, self.num_eoc])
            inputs_enc = torch.where(flag > 0.01, inputs_enc, input_enc)

            inputs_PE = torch.cat([inputs_PE, inputs_enc], dim=-1)

        x_in = torch.cat([inputs_PE, x], dim=-1)

        y = x_in
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, f"lin{l}")
            if l in self.skip_in:
                y = torch.cat([y, x_in], 1) / np.sqrt(2)
            y = lin(y)
            if l < self.num_layers - 2:
                y = self.activation(y)

        return torch.cat([y[:, :1] / self.scale, y[:, 1:]], dim=-1)


def coarse2fine(progress_data, inputs, L):
    barf_c2f = [0.1, 0.5]
    if barf_c2f is not None:
        start, end = barf_c2f
        alpha = (progress_data - start) / (end - start) * L
        k = torch.arange(L, dtype=torch.float32, device=inputs.device)
        weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
        shape = inputs.shape
        input_enc = (inputs.view(-1, L, int(shape[1] / L)) * weight.tile(int(shape[1] / L), 1).T).view(*shape)
    return input_enc, weight
