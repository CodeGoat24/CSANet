import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



BN_MOMENTUM = 0.1


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.fc1 = nn.Conv2d(dim, int(dim * mlp_ratio), 1)
        self.pos = nn.Conv2d(int(dim * mlp_ratio), int(dim * mlp_ratio), 3, padding=1, groups=int(dim * mlp_ratio))
        self.fc2 = nn.Conv2d(int(dim * mlp_ratio), dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x

class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)



    def forward(self, x, y):
        B, Head, HW, C = x.shape

        x = x.permute(0, 3, 2, 1);
        y = y.permute(0, 3, 2, 1);
        x = self.norm(x)
        y = self.norm(y)
        a = self.a(x)
        x = a * self.v(y)
        x = self.proj(x)

        return x


class ConvModBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2., drop_path=0.):
        super().__init__()

        self.attn = ConvMod(dim)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)


    def forward(self, x, y):
        x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x, y)
        x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x)
        return x

class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pool_ratios=[1, 2, 3, 6]):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t * t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()

        self.norm = nn.LayerNorm(dim)

        self.pool = nn.AdaptiveAvgPool2d((1, 84))


    def forward(self, x, H, W, d_convs=None):

        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        pools = []
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        for pool_ratio in self.pool_ratios:
            pool = F.adaptive_avg_pool2d(x_, (round(H / pool_ratio), round(W / pool_ratio)))
            # pool = pool + l(pool)  # fix backward bug in higher torch versions when training
            pools.append(pool.view(B, C, -1))

        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0, 2, 1))

        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)

        x = self.proj(x)
        # x = self.pool(x.permute(0, 2, 1).reshape(B, C, H, W)).transpose(-2, -1)


        return x


class PoolingAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratios=[12, 16, 20, 24]):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, pool_ratios=pool_ratios)

        self.norm2 = norm_layer(dim)


    def forward(self, x, d_convs=None):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.norm1(x)
        x = x + self.attn(x, H, W, d_convs=d_convs)
        x = self.norm2(x)

        return x

def vgg16_bn(pretrained):
    model = models.vgg16_bn(pretrained=False)
    model.features[0] = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True)

    if pretrained:
        pretrained_state_dict = models.vgg16_bn(pretrained=True).state_dict()
        conv = pretrained_state_dict['features.0.weight']
        new = torch.zeros(64, 1, 3, 3)
        for i, output_channel in enumerate(conv):
            new[i] = 0.299 * output_channel[0] + 0.587 * output_channel[1] + 0.114 * output_channel[2]
        pretrained_state_dict['features.0.weight'] = torch.cat((conv, new), dim=1)
        model.load_state_dict(pretrained_state_dict)

    return model


class VAEEncoder(nn.Module): 
    def __init__(self, opt):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(opt.d_model, 1024),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(1024, opt.d_noise)
        self.fc3 = nn.Linear(1024, opt.d_noise)
    
    def encode(self, x):
        h = self.fc1(x)
        mu = self.fc2(h)
        logvar = self.fc3(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar / 2.0)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        latent_code = self.reparameterize(mu, logvar)
        return latent_code, mu, logvar


class FgBgRegression(nn.Module):
    def __init__(self, opt):
        super(FgBgRegression, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(opt.d_model + opt.d_noise, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 3),
        )

    def forward(self, x):
        out = self.regressor(x)
        return out

class FeatLayer(nn.Module):
    def __init__(self, opt, n_mesh):
        super(FeatLayer, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(opt.d_model, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, opt.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(opt.d_model, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((n_mesh, n_mesh))

    def forward(self, x):
        feats = self.features(x)
        pooled_feats = self.pool(feats)
        # nodes = pooled_feats.view(pooled_feats.shape[0], pooled_feats.shape[1], -1).transpose(1, 2).contiguous()
        return pooled_feats


class FeatExtractor(nn.Module):
    def __init__(self, opt, n_mesh_list, multi):
        super(FeatExtractor, self).__init__()
        self.layers = nn.ModuleList([FeatLayer(opt, n_mesh) for n_mesh in n_mesh_list])
        self.multi = multi

    def forward(self, x):
        node_list = []
        for layer in self.layers:
            nodes = layer(x)
            if self.multi:
                nodes = nodes.view(nodes.shape[0], nodes.shape[1], -1).contiguous()
            node_list.append(nodes)
        return torch.cat(node_list, dim=2)

class ScaledDotProductAttention(nn.Module):
   def __init__(self, opt):
       super(ScaledDotProductAttention, self).__init__()
       self.opt = opt
       # self.pos_k = nn.Embedding(opt.n_heads * opt.len_k, opt.d_k // 4)
       # self.pos_k = nn.Embedding(opt.n_heads * opt.len_k, opt.d_k)
       # self.pos_v = nn.Embedding(opt.n_heads * opt.len_k, opt.d_v)
       self.pos_ids = torch.LongTensor(list(range(opt.n_heads * opt.len_k))).view(1, opt.n_heads, opt.len_k)

   def forward(self, Q, K, V):
       # K_pos = self.pos_k(self.pos_ids.cuda())
       # V_pos = self.pos_v(self.pos_ids.cuda())
       scores = torch.matmul(Q, (K).transpose(-1, -2)) / np.sqrt(self.opt.d_k)
       attn = nn.Softmax(dim=-1)(scores)
       context = torch.matmul(attn, V)
       return context, attn

class PoolAttention(nn.Module):
   def __init__(self, opt, pool_ratios=[2, 4, 8]):
       super(PoolAttention, self).__init__()
       self.opt = opt
       self.W_Q = nn.Linear(opt.d_model, opt.d_k * opt.n_heads)
       self.W_K = nn.Linear(opt.d_model, opt.d_k * opt.n_heads)
       self.W_V = nn.Linear(opt.d_model, opt.d_v * opt.n_heads)
       self.att = ScaledDotProductAttention(opt)
       self.W_O = nn.Linear(opt.n_heads * opt.d_v, opt.d_model)
       self.norm = nn.LayerNorm(opt.d_model)
       self.pool_ratios = pool_ratios


   def forward(self, x):
       B, C, H, W = x.shape
       x_ = x.view(B, C, -1).transpose(1, 2)
       residual= x_
       q = self.W_Q(x_).view(B, -1, self.opt.n_heads, self.opt.d_k).transpose(1,2)

       pools = []
       for pool_ratio in self.pool_ratios:
           pool = F.adaptive_avg_pool2d(x, (round(H / pool_ratio), round(W / pool_ratio)))
           # pool = pool + l(pool)  # fix backward bug in higher torch versions when training
           pools.append(pool.view(B, C, -1))

       pools = torch.cat(pools, dim=2)
       pools = self.norm(pools.permute(0, 2, 1))

       k = self.W_K(pools).view(B, -1, self.opt.n_heads, self.opt.d_k).transpose(1, 2)
       v = self.W_V(pools).view(B, -1, self.opt.n_heads, self.opt.d_v).transpose(1,2)
       context, attn = self.att(q, k, v)
       context = context.transpose(1, 2).contiguous().view(B, -1, self.opt.n_heads * self.opt.d_v)
       output = self.W_O(context)
       return self.norm(output + residual)

class CrossConvAtt(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_channels = opt.d_model
        out_channels = opt.d_model
        self.opt = opt
        self.in_channels = in_channels


        self.key1 = nn.Conv2d(in_channels, opt.d_k // 2 * self.opt.n_heads, kernel_size = 1, stride = 1)
        self.value1 = nn.Conv2d(in_channels, opt.d_v * self.opt.n_heads, kernel_size = 1, stride = 1)

        self.key2 = nn.Conv2d(in_channels, opt.d_k // 2 * self.opt.n_heads, kernel_size = 1, stride = 1)
        self.value2 = nn.Conv2d(in_channels, opt.d_v * self.opt.n_heads, kernel_size = 1, stride = 1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)

        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels*2, 512, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                                        nn.ReLU(),
                                        nn.Conv2d(512, out_channels, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                                        nn.ReLU(),
                                      ) # conv_f
        self.W_O_1 = nn.Linear(opt.n_heads * opt.d_v, opt.d_model)
        self.W_O_2 = nn.Linear(opt.n_heads * opt.d_v, opt.d_model)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = nn.LayerNorm(opt.d_model)
        self.convMod1 = ConvMod(dim=64)
        self.convMod2 = ConvMod(dim=64)


    def forward(self, input1, input2):
        batch_size, channels, height, width = input1.shape

        k1 = self.key1(input1).view(batch_size, self.opt.n_heads, -1, height * width).permute(0, 1, 3, 2)
        v1 = self.value1(input1).view(batch_size, self.opt.n_heads,-1, height * width).permute(0, 1, 3, 2)


        k2 = self.key2(input2).view(batch_size, self.opt.n_heads, -1, height * width).permute(0, 1, 3, 2)
        v2 = self.value2(input2).view(batch_size, self.opt.n_heads, -1, height * width).permute(0, 1, 3, 2)


        k = torch.cat([k1, k2], 2).view(batch_size, self.opt.n_heads, -1, height * width).permute(0, 1, 3, 2)


        context1 = self.convMod1(k, v1);
        out1 = context1.transpose(1, 2).contiguous().view(batch_size, -1, self.opt.n_heads * self.opt.d_v)

        out1 = self.W_O_1(out1).view(*input1.shape)
        out1 = self.gamma1 * out1 + input1
        out1 = self.norm(out1.view(batch_size, height, width, channels)).view(*input1.shape)

        context2 = self.convMod1(k, v2);
        out2 = context2.transpose(1, 2).contiguous().view(batch_size, -1, self.opt.n_heads * self.opt.d_v)

        out2 = self.W_O_2(out2).view(*input2.shape)
        out2 = self.gamma2 * out2 + input2
        out2 = self.norm(out2.view(batch_size, height, width, channels)).view(*input2.shape)

        feat_sum = self.conv_cat(torch.cat([out1, out2], 1))
        pooled_feats = self.pool(feat_sum)
        feats = pooled_feats.view(pooled_feats.shape[0], pooled_feats.shape[1], -1).transpose(1, 2).contiguous()
        return feats