import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from einops import rearrange
import copy
import math

from model.base import BaseModule
class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                               dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)            

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads = self.heads, qkv=3)            
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

 

class GradLogPEstimator2d(BaseModule):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8,
                 n_spks=1, spk_emb_dim=64, n_feats=80, pe_scale=1000):
        super(GradLogPEstimator2d, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.n_spks = n_spks if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale
        
        if n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4), Mish(),
                                               torch.nn.Linear(spk_emb_dim * 4, n_feats))
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(),
                                       torch.nn.Linear(dim * 4, dim))

        dims = [2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                     ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                     Residual(Rezero(LinearAttention(dim_in))),
                     Upsample(dim_in)]))
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask, mu, t, spk=None):
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)
        
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if self.n_spks < 2:
            x = torch.stack([mu, x], 1)
        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.stack([mu, x, s], 1)
        mask = mask.unsqueeze(1)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)

 
 

class Como(BaseModule):
    def __init__(self, teacher = True ):
        super().__init__()
        self.denoise_fn = GradLogPEstimator2d(64) 
        self.teacher = teacher
        if not teacher:
            self.denoise_fn_ema = copy.deepcopy(self.denoise_fn)
            self.denoise_fn_pretrained = copy.deepcopy(self.denoise_fn)

 
        self.P_mean =-1.2 # P_mean
        self.P_std =1.2# P_std
        self.sigma_data =0.5# sigma_data
 
        self.sigma_min= 0.002
        self.sigma_max= 80
        self.rho=7
 
 
 
        self.N = 50         #100   
 
        
        # Time step discretization
        step_indices = torch.arange(self.N )   
        t_steps = (self.sigma_min ** (1 / self.rho) + step_indices / (self.N - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho
        self.t_steps = torch.cat([torch.zeros_like(t_steps[:1]), self.round_sigma(t_steps)])  
 

    def EDMPrecond(self, x, sigma ,cond,denoise_fn,mask):
 
        sigma = sigma.reshape(-1, 1, 1 )
 
        c_skip = self.sigma_data ** 2 / ((sigma-self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = (sigma-self.sigma_min) * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
 
        F_x =  denoise_fn((c_in * x), mask, cond,c_noise.flatten()) 
        D_x = c_skip * x + c_out * (F_x  )
        return D_x

 


    def EDMLoss(self, x_start,   cond,nonpadding ):
 
        rnd_normal = torch.randn([x_start.shape[0], 1,  1], device=x_start.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

 
        n = (torch.randn_like(x_start) ) * sigma
        D_yn = self.EDMPrecond(x_start + n, sigma ,cond,self.denoise_fn,nonpadding)
        loss = (weight * ((D_yn - x_start) ** 2))
        loss=loss*nonpadding.unsqueeze(1).unsqueeze(1)
        loss=loss.mean() 
        return loss

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
 

    def edm_sampler(self,
         latents,  cond,nonpadding,
        num_steps=50, sigma_min=0.002, sigma_max=80, rho=7, 
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
        # S_churn=40 ,S_min=0.05,S_max=50,S_noise=1.003,# S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
        # S_churn=30 ,S_min=0.01,S_max=30,S_noise=1.007,
        # S_churn=30 ,S_min=0.01,S_max=1,S_noise=1.007,
        # S_churn=80 ,S_min=0.05,S_max=50,S_noise=1.003,
    ):
 

        # Time step discretization.
        num_steps=num_steps+1
        step_indices = torch.arange(num_steps,   device=latents.device)


        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  

 

        # Main sampling loop.
        x_next = latents * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  
            x_cur = x_next
            # print('step',i+1)
            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur # + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = self.EDMPrecond(x_hat, t_hat , cond,self.denoise_fn,nonpadding) 
            d_cur = (x_hat - denoised) / t_hat #!
            x_next = x_hat + (t_next - t_hat) * d_cur
            if i < num_steps - 1:
                denoised = self.EDMPrecond(x_next, t_next , cond,self.denoise_fn,nonpadding) 
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)


        return x_next


  
  
    def CTLoss_D(self,y, cond,nonpadding): #k 
 
        with torch.no_grad():
            mu = 0.95  
            for p, ema_p in zip(self.denoise_fn.parameters(), self.denoise_fn_ema.parameters()):
                ema_p.mul_(mu).add_(p, alpha=1 - mu)


        n = torch.randint(1, self.N, (y.shape[0],))
 
 

        z = torch.randn_like(y) 

        tn_1 = self.c_t_d(n + 1 ).reshape(-1, 1,   1).to(y.device)
        f_theta = self.EDMPrecond(y + tn_1 * z, tn_1, cond, self.denoise_fn,nonpadding)

        with torch.no_grad():
            tn = self.c_t_d(n ).reshape(-1, 1,   1).to(y.device)

            #euler step
            x_hat = y + tn_1 * z
            denoised = self.EDMPrecond(x_hat, tn_1 , cond,self.denoise_fn_pretrained,nonpadding) 
            d_cur = (x_hat - denoised) / tn_1
            y_tn = x_hat + (tn - tn_1) * d_cur
 
 
  
            denoised2 = self.EDMPrecond(y_tn, tn , cond,self.denoise_fn_pretrained,nonpadding) 
            d_prime = (y_tn - denoised2) / tn
            y_tn = x_hat + (tn - tn_1) * (0.5 * d_cur + 0.5 * d_prime)



            f_theta_ema = self.EDMPrecond( y_tn, tn,cond, self.denoise_fn_ema,nonpadding)

 
        loss =   (f_theta - f_theta_ema.detach()) ** 2 
        loss=loss* nonpadding
        loss=loss.mean() 


 
        
        return loss

    def c_t_d(self, i ):


        return self.t_steps[i]

    def get_t_steps(self,N):
        N=N+1
        step_indices = torch.arange( N ) #, device=latents.device)
        t_steps = (self.sigma_min ** (1 / self.rho) + step_indices / (N- 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho

        return  t_steps.flip(0)


    def CT_sampler(self,  latents, cond,nonpadding,t_steps=1):  

        if t_steps ==1:
            t_steps=[80  ]
        else:
            t_steps=self.get_t_steps(t_steps)

        t_steps = torch.as_tensor(t_steps).to(latents.device)
        latents = latents * t_steps[0]
        x = self.EDMPrecond(latents, t_steps[0],cond,self.denoise_fn,nonpadding)
        
        for t in t_steps[1:-1]:
            z = torch.randn_like(x) 
            x_tn = x +  (t ** 2 - self.sigma_min ** 2).sqrt()*z
            x = self.EDMPrecond(x_tn, t,cond,self.denoise_fn,nonpadding)
        
        return x

    def forward(self, x,nonpadding,cond,t_steps=1, infer=False):
 
        
        if self.teacher: #teacher model  
            if not  infer:
 
                loss = self.EDMLoss(x, cond,nonpadding)
                
                return loss
            else:
 
                shape = (cond.shape[0],   80, cond.shape[2])
                x = torch.randn(shape, device=x.device)   
                x=self.edm_sampler(x, cond,nonpadding,t_steps)
            return x
        else:  #Consistency distillation
            if not  infer:
    
                loss = self.CTLoss_D(x, cond,nonpadding)
                
                return loss
            else:
 
                shape = (cond.shape[0],   80, cond.shape[2])
                x = torch.randn(shape, device=x.device)   
                x=self.CT_sampler(x, cond,nonpadding,t_steps)
 
            return x
 
