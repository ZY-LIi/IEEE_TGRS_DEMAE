import torch
import torch.nn as nn
import numpy as np
import math

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class sinusoidal_time_embedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class time_mlp(nn.Module):
    def __init__(self, in_features, out_features):
        super(time_mlp, self).__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.SiLU(),
            nn.Linear(out_features, out_features),
        )

    def forward(self, t):
        time_emb = self.time_mlp(t)
        return time_emb


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConditionalTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.AdaptiveNorm = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 4)
        )

    def forward(self, x, t):
        attn_scale, attn_shift, mlp_scale, mlp_shift = self.AdaptiveNorm(t).chunk(4, dim=-1)
        x = x + self.attn(self.norm1(x) * (1 + attn_scale) + attn_shift)
        x = x + self.mlp(self.norm2(x) * (1 + mlp_scale) + mlp_shift)
        return x


class DEMAE_pretrain(nn.Module):
    def __init__(self, dim, patch_size, num_heads=4, mlp_ratio=4.):
        super(DEMAE_pretrain, self).__init__()
        self.patch_embed = nn.Linear(dim, dim * 2)
        num_tokens = patch_size ** 2
        self.time_token = sinusoidal_time_embedding(dim=dim * 4)
        self.time_mlp = time_mlp(in_features=dim * 4, out_features=dim * 2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim * 2))
        self.pos_embed_e = get_sinusoid_encoding_table(num_tokens + 2, dim * 2)

        # encoder
        self.encoder_block1 = ConditionalTransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.encoder_block2 = ConditionalTransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.encoder_block3 = ConditionalTransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.encoder_block4 = ConditionalTransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)

        # encoder to decoder
        self.en2de_norm = nn.LayerNorm(dim * 2)
        self.encoder_to_decoder = nn.Linear(dim * 2, dim * 2)
        self.masked_token = nn.Parameter(torch.zeros(1, 1, dim * 2))
        self.pos_embed_d = get_sinusoid_encoding_table(num_tokens + 2, dim * 2)

        # decoder - diffusion
        self.decoder_dfs_conv1 = nn.Linear(dim * 4, dim * 2)
        self.decoder_dfs_blk1 = TransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.decoder_dfs_conv2 = nn.Linear(dim * 4, dim * 2)
        self.decoder_dfs_blk2 = TransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.decoder_dfs_conv3 = nn.Linear(dim * 4, dim * 2)
        self.decoder_dfs_blk3 = TransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.decoder_dfs_conv4 = nn.Linear(dim * 4, dim * 2)
        self.decoder_dfs_blk4 = TransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)

        # decoder - mae
        self.decoder_mae_blk1 = TransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.decoder_mae_blk2 = TransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)

        # projection head
        self.dfs_norm = nn.LayerNorm(dim * 2)
        self.dfs_head = nn.Linear(dim * 2, dim)
        self.mae_norm = nn.LayerNorm(dim * 2)
        self.mae_head = nn.Linear(dim * 2, dim)

        # initialize
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.masked_token, std=.02)

    def forward(self, x, timesteps, mask):

        B, N, C = x.shape
        x = self.patch_embed(x)
        time_token = self.time_mlp(self.time_token(timesteps))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)

        cls_token = self.cls_token
        x = torch.cat((cls_token.tile(dims=(B, 1, 1)), x), dim=1)
        x = x + self.pos_embed_e.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()

        # encoder process
        cls_time_mask = torch.zeros((B, 2)).to(x.device)
        mask = torch.cat((cls_time_mask, mask), dim=1).bool()
        B, N, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)
        skip = []
        x_vis = self.encoder_block1(x_vis, time_token)
        skip.append(x_vis)
        x_vis = self.encoder_block2(x_vis, time_token)
        skip.append(x_vis)
        x_vis = self.encoder_block3(x_vis, time_token)
        skip.append(x_vis)
        x_vis = self.encoder_block4(x_vis, time_token)
        skip.append(x_vis)

        # encoder to decoder
        x_vis = self.en2de_norm(x_vis)
        x_vis = self.encoder_to_decoder(x_vis)

        # decoder process - diffusion
        x_bar = torch.cat((x_vis, skip.pop()), dim=-1)
        x_bar = self.decoder_dfs_conv1(x_bar)
        x_bar = self.decoder_dfs_blk1(x_bar)
        x_bar = torch.cat((x_bar, skip.pop()), dim=-1)
        x_bar = self.decoder_dfs_conv2(x_bar)
        x_bar = self.decoder_dfs_blk2(x_bar)
        x_bar = torch.cat((x_bar, skip.pop()), dim=-1)
        x_bar = self.decoder_dfs_conv3(x_bar)
        x_bar = self.decoder_dfs_blk3(x_bar)
        x_bar = torch.cat((x_bar, skip.pop()), dim=-1)
        x_bar = self.decoder_dfs_conv4(x_bar)
        x_bar = self.decoder_dfs_blk4(x_bar)

        # decoder process - mae
        pos_embed_d = self.pos_embed_d.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        B, N, C = x.shape
        pos_embed_vis = pos_embed_d[~mask].reshape(B, -1, C)
        pos_embed_mask = pos_embed_d[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_embed_vis, self.masked_token.expand(B, pos_embed_mask.shape[1], -1) + pos_embed_mask], dim=1)
        x_full = self.decoder_mae_blk1(x_full)
        x_full = self.decoder_mae_blk2(x_full)

        # projection
        denoise = self.dfs_head(self.dfs_norm(x_bar))
        reconstruct = self.mae_head(self.mae_norm(x_full[:, -pos_embed_mask.shape[1]:, :]))

        return denoise[:, 2:, :], reconstruct


class DEMAE_finetune(nn.Module):
    def __init__(self, dim, num_class, patch_size, num_heads=4, mlp_ratio=4.):
        super(DEMAE_finetune, self).__init__()
        self.patch_embed = nn.Linear(dim, dim * 2)
        num_tokens = patch_size ** 2
        self.time_token = sinusoidal_time_embedding(dim=dim * 4)
        self.time_mlp = time_mlp(in_features=dim * 4, out_features=dim * 2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim * 2))
        self.pos_embed_e = get_sinusoid_encoding_table(num_tokens + 2, dim * 2)

        # encoder
        self.encoder_block1 = ConditionalTransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.encoder_block2 = ConditionalTransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.encoder_block3 = ConditionalTransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.encoder_block4 = ConditionalTransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)

        # encoder to decoder
        self.en2de_norm = nn.LayerNorm(dim * 2)
        self.cls_token_classifier = nn.Linear(dim * 2, num_class)

        # initialize
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward(self, x, timesteps, mask):

        B, N, C = x.shape
        x = self.patch_embed(x)
        time_token = self.time_mlp(self.time_token(timesteps))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)

        cls_token = self.cls_token
        x = torch.cat((cls_token.tile(dims=(B, 1, 1)), x), dim=1)
        x = x + self.pos_embed_e.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()

        # encoder process
        cls_time_mask = torch.zeros((B, 2)).to(x.device)
        mask = torch.cat((cls_time_mask, mask), dim=1).bool()
        B, N, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)
        x_vis = self.encoder_block1(x_vis, time_token)
        x_vis = self.encoder_block2(x_vis, time_token)
        x_vis = self.encoder_block3(x_vis, time_token)
        x_vis = self.encoder_block4(x_vis, time_token)

        # encoder to decoder
        x_vis = self.en2de_norm(x_vis)
        x_cls_token = x_vis[:, 0, :]
        y_hat_cls = self.cls_token_classifier(x_cls_token)

        return y_hat_cls
        # return x_cls_token  # for feature separability experiment


class DEMAE_test(nn.Module):
    def __init__(self, dim, num_class, patch_size, num_heads=4, mlp_ratio=4.):
        super(DEMAE_test, self).__init__()
        self.patch_embed = nn.Linear(dim, dim * 2)
        num_tokens = patch_size ** 2
        self.time_token = sinusoidal_time_embedding(dim=dim * 4)
        self.time_mlp = time_mlp(in_features=dim * 4, out_features=dim * 2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim * 2))
        self.pos_embed_e = get_sinusoid_encoding_table(num_tokens + 2, dim * 2)

        # encoder
        self.encoder_block1 = ConditionalTransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.encoder_block2 = ConditionalTransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.encoder_block3 = ConditionalTransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.encoder_block4 = ConditionalTransformerBlock(dim=dim * 2, num_heads=num_heads, mlp_ratio=mlp_ratio)

        # encoder to decoder
        self.en2de_norm = nn.LayerNorm(dim * 2)

        # projection head
        self.cls_token_classifier = nn.Linear(dim * 2, num_class)

    def forward(self, x, timesteps):

        B, N, C = x.shape
        x = self.patch_embed(x)
        time_token = self.time_mlp(self.time_token(timesteps))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)

        cls_token = self.cls_token
        x = torch.cat((cls_token.tile(dims=(B, 1, 1)), x), dim=1)
        x = x + self.pos_embed_e.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()

        # encoder process
        x_vis = x
        x_vis = self.encoder_block1(x_vis, time_token)
        x_vis = self.encoder_block2(x_vis, time_token)
        x_vis = self.encoder_block3(x_vis, time_token)
        x_vis = self.encoder_block4(x_vis, time_token)

        # encoder to decoder
        x_vis = self.en2de_norm(x_vis)
        x_cls_token = x_vis[:, 0, :]
        y_hat_cls = self.cls_token_classifier(x_cls_token)

        return y_hat_cls
        # return x_cls_token  # for feature separability experiment


if __name__ == '__main__':
    # mask ratio = 0.5 / pretrain stage
    x = torch.rand((4, 32, 9, 9))
    timesteps = torch.randint(100, (4,))
    num_tokens = 9 * 9
    elements = np.concatenate(([1] * (num_tokens // 2), [0] * (num_tokens // 2)), axis=0)
    mask = []
    for _ in range(4):
        elements = np.random.permutation(elements)
        mask.append(np.insert(elements, num_tokens // 2, 0))
    mask = torch.Tensor(np.array(mask))
    model = DEMAE_pretrain(dim=32, patch_size=9)
    B, C, H, W = x.shape
    x = x.reshape(B, C, -1).transpose(1, 2)
    denoise, reconstruct = model(x, timesteps, mask)

    # # mask ratio = 0.5 / train stage
    # x = torch.rand((4, 32, 9, 9))
    # timesteps = torch.randint(100, (4,))
    # num_tokens = 9 * 9
    # elements = np.concatenate(([1] * (num_tokens // 2), [0] * (num_tokens // 2)), axis=0)
    # mask = []
    # for _ in range(4):
    #     elements = np.random.permutation(elements)
    #     mask.append(np.insert(elements, num_tokens // 2, 0))
    # mask = torch.Tensor(np.array(mask))
    # model = DEMAE_finetune(dim=32, num_class=9, patch_size=9)
    # B, C, H, W = x.shape
    # x = x.reshape(B, C, -1).transpose(1, 2)
    # x_cls_token, denoise, reconstruct = model(x, timesteps, mask)

    # # test stage
    # x = torch.rand((4, 32, 9, 9))
    # timesteps = torch.zeros(4)
    # model = DEMAE_test(dim=32, num_class=9, patch_size=9)
    # B, C, H, W = x.shape
    # x = x.reshape(B, C, -1).transpose(1, 2)
    # x_cls_token = model(x, timesteps)