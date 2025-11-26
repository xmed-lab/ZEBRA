import torch
import torch.nn as nn
import torch.nn.functional as F



class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.eps = 1e-6

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z, mask=None):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        if z.dtype == torch.float16:
            fp16 = True
            z = z.to(torch.float32)
        else:
            fp16 = False
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        # .........\end

        # with:
        # .........\start
        # min_encoding_indices = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        '''
        # compute loss for embedding
        # TODO:有bug，这里应该交换
        # loss = torch.mean((z_q - z.detach()) ** 2) + self.beta * torch.mean((z_q.detach() - z) ** 2)
        # [B,D,H,W]
        if mask is None:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        else:
            mask = F.interpolate(mask, (z.shape[2], z.shape[3]), mode='nearest')
            mask = mask.repeat(1, z.shape[1], 1, 1)
            mask_sum = torch.sum(mask, dim=[2, 3]) + self.eps
            loss_z = (z_q.detach() - z) ** 2
            loss_zq = (z_q - z.detach()) ** 2
            loss = torch.mean(torch.sum((loss_z + self.beta * loss_zq) * mask, dim=[2, 3]) / mask_sum)
        '''

        # preserve gradients直通估计
        # z_q = z + (z_q - z).detach()


        # perplexity
        # e_mean = torch.mean(min_encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if fp16:
            z_q = z_q.to(torch.float16)

        # return z_q, (perplexity, min_encodings, min_encoding_indices)   # [B C H W]
        return z_q, (min_encodings, min_encoding_indices)   # [B C H W]

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

    def approximate_codebook(self, probs, shape):
        # get quantized latent vectors
        # [B,L,V]X[V,D]=[B,L,D]
        z_q = torch.matmul(probs.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class MaskGIT_VectorQuantizer(nn.Module):
    def __init__(self, num_code, dim_code, norm=True):
        super(MaskGIT_VectorQuantizer, self).__init__()
        self.eps = 1e-6
        self.norm = norm

        self.embedding = nn.Embedding(num_code, dim_code) # [M, C]
        self.embedding.weight.data.uniform_(-1.0 / num_code, 1.0 / num_code)

    def forward(self, z):
        # z: [N, C, H, W]
        if z.dtype == torch.float16:
            fp16 = True
            z = z.to(torch.float32)
        else:
            fp16 = False

        N, C, H, W = z.shape
        # reshape z and flatten: [N, C, H, W] -> [N, H, W, C] -> [NHW, C]
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(N*H*W, C)

         # norm
        if self.norm:
            z_flat_norm = F.normalize(z_flat, p=2, dim=1) # [NHW, C]
            code_norm = F.normalize(self.embedding.weight, p=2, dim=1) # [M, C]
            d = torch.sum(z_flat_norm ** 2, dim=1, keepdim=True) + \
                torch.sum(code_norm ** 2, dim=1) - 2 * \
                torch.matmul(z_flat_norm, code_norm.t()) # [NHW, M]
        else:
            d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
                torch.matmul(z_flat, self.embedding.weight.t()) # [NHW, M]

        min_idx = torch.argmin(d, dim=1, keepdim=True).repeat(1, C) # [NHW, C]
        quant = self.read_codebook(min_idx, shape=(N, H, W, C)) # [N, C, H, W]

        if fp16:
            quant = quant.to(torch.float16)
        index = min_idx[:, 0].view(N, H, W) # [N, H, W]

        return quant, index # [N, C, H, W], [N, H, W]

    def read_codebook(self, index, shape=None):
        # index: [NHW, C]
        quant = torch.gather(self.embedding.weight, 0, index) # [NHW, C]
        if shape is not None:
            quant = quant.view(shape).permute(0, 3, 1, 2).contiguous() # [N, H, W, C] -> [N, C, H, W]

        return quant

class VectorQuantizer2(nn.Module):
    def __init__(self, num_code, dim_code, norm=True):
        super(VectorQuantizer2, self).__init__()
        self.eps = 1e-6
        self.norm = norm

        self.embedding = nn.Embedding(num_code, dim_code) # [M, C]
        self.embedding.weight.data.uniform_(-1.0 / num_code, 1.0 / num_code)

    def forward(self, z):
        # z: [N, C, H, W]
        if z.dtype == torch.float16:
            fp16 = True
            z = z.to(torch.float32)
        else:
            fp16 = False

        N, C, H, W = z.shape
        # reshape z and flatten: [N, C, H, W] -> [N, H, W, C] -> [NHW, C]
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(N*H*W, C)

         # norm
        if self.norm:
            z_flat_norm = F.normalize(z_flat, p=2, dim=1) # [NHW, C]
            code_norm = F.normalize(self.embedding.weight, p=2, dim=1) # [M, C]
            d = torch.sum(z_flat_norm ** 2, dim=1, keepdim=True) + \
                torch.sum(code_norm ** 2, dim=1) - 2 * \
                torch.matmul(z_flat_norm, code_norm.t()) # [NHW, M]
        else:
            d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
                torch.matmul(z_flat, self.embedding.weight.t()) # [NHW, M]

        min_idx = torch.argmin(d, dim=1, keepdim=True).repeat(1, C) # [NHW, C]
        quant = self.read_codebook(min_idx, shape=(N, H, W, C)) # [N, C, H, W]

        if fp16:
            quant = quant.to(torch.float16)
        index = min_idx[:, 0].view(N, H, W) # [N, H, W]

        return quant, index # [N, C, H, W], [N, H, W]

    def read_codebook(self, index, shape=None):
        # index: [NHW, C]
        quant = torch.gather(self.embedding.weight, 0, index) # [NHW, C]
        if shape is not None:
            quant = quant.view(shape).permute(0, 3, 1, 2).contiguous() # [N, H, W, C] -> [N, C, H, W]

        return quant


# including pre-conv and post-conv
class VectorQuantizer3(nn.Module):
    def __init__(self, num_code, dim_code, dim_latent, use_preconv=False, norm=True):
        super(VectorQuantizer3, self).__init__()
        self.eps = 1e-6
        self.norm = norm

        self.embedding = nn.Embedding(num_code, dim_code) # [M, C]
        self.embedding.weight.data.uniform_(-1.0 / num_code, 1.0 / num_code)

        if use_preconv:
            self.pre_conv = nn.Conv2d(dim_latent, dim_code, kernel_size=1)
            self.post_conv = nn.Conv2d(dim_code, dim_latent, kernel_size=1)
        else:
            assert dim_latent == dim_code
            self.pre_conv = nn.Identity()
            self.post_conv = nn.Identity()

    def quantize(self, z):
        # z: [N, C, H, W]
        if z.dtype == torch.float16:
            fp16 = True
            z = z.to(torch.float32)
        else:
            fp16 = False

        N, C, H, W = z.shape
        # reshape z and flatten: [N, C, H, W] -> [N, H, W, C] -> [NHW, C]
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(N*H*W, C)

         # norm
        if self.norm:
            z_flat_norm = F.normalize(z_flat, p=2, dim=1) # [NHW, C]
            code_norm = F.normalize(self.embedding.weight, p=2, dim=1) # [M, C]
            d = torch.sum(z_flat_norm ** 2, dim=1, keepdim=True) + \
                torch.sum(code_norm ** 2, dim=1) - 2 * \
                torch.matmul(z_flat_norm, code_norm.t()) # [NHW, M]
        else:
            d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
                torch.matmul(z_flat, self.embedding.weight.t()) # [NHW, M]

        min_idx = torch.argmin(d, dim=1, keepdim=True).repeat(1, C) # [NHW, C]
        quant = self.read_codebook(min_idx, shape=(N, H, W, C)) # [N, C, H, W]

        if fp16:
            quant = quant.to(torch.float16)
        index = min_idx[:, 0].view(N, H, W) # [N, H, W]

        return quant, index # [N, C, H, W], [N, H, W]

    def forward(self, z):
        # preconv
        z_code = self.pre_conv(z)

        # quantize
        z_quant, z_index = self.quantize(z_code)

        # postconv
        z_quant_ = z_code + (z_quant - z_code).detach()
        z_post = self.post_conv(z_quant_)

        return z_code, z_post, z_quant, z_index

    def read_codebook(self, index, shape=None):
        # index: [NHW, C]
        quant = torch.gather(self.embedding.weight, 0, index) # [NHW, C]
        if shape is not None:
            quant = quant.view(shape).permute(0, 3, 1, 2).contiguous() # [N, H, W, C] -> [N, C, H, W]

        return quant

