import copy
import math
import random

from transformer.embedding import SinePositionalEmbedding
from transformer.transformer import LayerNorm, TransformerEncoder, TransformerEncoderLayer
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from module import commons
from module import modules
from module import attentions

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from module.commons import init_weights, get_padding
from module.quantize import VectorQuantize
from text import symbols


class TextEncoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.text_emb = nn.Embedding(len(symbols), 512)
        self.bert_proj = nn.Linear(1024, 512)

        norm_first = False
        self.position = SinePositionalEmbedding(
            512, dropout=0.1, scale=False, alpha=True)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=512,
                nhead=16,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True,
                norm_first=False, ),
            num_layers=6,
            norm=LayerNorm(512) if norm_first else None, )

        self.proj = nn.Conv1d(512, out_channels, 1)

    def forward_encoder(self, x, mask=None):
        if mask is not None:
            mask = ~mask.squeeze(1).bool()
        x = self.position(x)
        x, _ = self.encoder(
            (x, None), src_key_padding_mask=mask)
        return x

    def forward(self, x, x_lengths, bert):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(-1)), 1).to(x.dtype)
        bert = self.bert_proj(bert.transpose(1, 2))
        x = self.text_emb(x) + bert
        x = self.forward_encoder(x, x_mask)
        x = self.proj(x.transpose(1, 2)) * x_mask
        return x, x_mask


class TransformerCouplingLayer(nn.Module):
    def __init__(
            self,
            channels,
            hidden_channels,
            kernel_size,
            n_layers,
            n_heads,
            p_dropout=0,
            filter_channels=0,
            mean_only=False,
            gin_channels=0,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            isflow=True,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class TransformerCouplingBlock(nn.Module):
    def __init__(
            self,
            channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            n_flows=4,
            gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = None

        for i in range(n_flows):
            self.flows.append(
                TransformerCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    n_layers,
                    n_heads,
                    p_dropout,
                    filter_channels,
                    mean_only=True,
                    gin_channels=self.gin_channels,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, spec_channels, gin_channels=0):

        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [weight_norm(nn.Conv2d(in_channels=filters[i],
                                       out_channels=filters[i + 1],
                                       kernel_size=(3, 3),
                                       stride=(2, 2),
                                       padding=(1, 1))) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        # self.wns = nn.ModuleList([weight_norm(num_features=ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=ref_enc_filters[-1] * out_channels,
                          hidden_size=256 // 2,
                          batch_first=True)
        self.proj = nn.Linear(128, gin_channels)

    def forward(self, inputs, mask=None):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0)).unsqueeze(-1)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class FramePriorNet(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()

        self.enc = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            6,
            kernel_size,
            p_dropout)
        self.out_proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.out_channels = out_channels

    def forward(self, x, x_mask):
        x = self.enc(x, x_mask)
        stats = self.out_proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs


class ProsodyEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels,
                 codebook_size=128,
                 ):
        super().__init__()

        self.in_proj = nn.Conv1d(in_channels, hidden_channels, 1)
        self.duration_emb = nn.Embedding(512, hidden_channels)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        # from dac
        print(codebook_size)
        print(codebook_size)
        print(codebook_size)
        print(codebook_size)
        self.prosody_quantizer = VectorQuantize(
            input_dim=hidden_channels,
            codebook_size=codebook_size,
            codebook_dim=8
        )
        self.duration_dec = modules.WN(hidden_channels, kernel_size, dilation_rate, 3, gin_channels=0)
        self.duration_proj = nn.Linear(hidden_channels, 512, False)

    def encode_dur(self, dur):
        dur = dur.clamp(0, 511)

        dur_emb = self.duration_emb(dur)
        return dur_emb.transpose(1, 2), dur

    def calc_dur_loss(self, x, x_mask, dur):
        x = self.duration_dec(x, x_mask)
        logits = self.duration_proj(x.transpose(1, 2))
        assert logits.shape[:2] == dur.shape[:2], (logits.shape, dur.shape)
        x_mask_bool = x_mask.squeeze(1).bool()
        dur = dur[x_mask_bool].view(-1)
        logits = logits[x_mask_bool].view(-1, 512)
        dur_loss = F.cross_entropy(logits, dur, reduction='mean')
        acc = (logits.argmax(-1) == dur).float().mean() * 100
        print('dur acc', acc.item())
        pred_dur = logits.argmax(-1)

        return dur_loss, pred_dur

    def decode_dur(self, x, x_mask):
        x = self.duration_dec(x, x_mask)
        logits = self.duration_proj(x.transpose(1, 2))
        logits = logits.view(-1, 512)
        pred_dur = logits.argmax(-1).reshape(-1, x.size(2))
        return pred_dur

    def forward(self, y, x_mask, attn, dur, v_mask):
        y = self.in_proj(y)
        s = attn.sum(axis=2).unsqueeze(2)
        s[s == 0] = 1
        attn_pool = attn / s
        x = torch.matmul(attn_pool.squeeze(1).transpose(1, 2), y.transpose(1, 2)).transpose(1, 2)
        x = x * v_mask.unsqueeze(1)

        g, dur = self.encode_dur(dur)
        x = self.enc(x, x_mask, g=g)
        z_q, commitment_loss, codebook_loss, indices, z_e = self.prosody_quantizer(x)
        quantize_loss = codebook_loss + 0.25 * commitment_loss
        dur_loss, pred_dur = self.calc_dur_loss(z_q, x_mask, dur)
        return z_q, quantize_loss, indices, dur_loss, pred_dur

    def decode(self, indices, x_mask):
        z_q = self.prosody_quantizer.decode(indices)
        dur = self.decode_dur(z_q, x_mask)
        return z_q, dur


import random
from tqdm import tqdm


class NARPredictor(nn.Module):
    def __init__(self, codebook_size=128, gin_channels=0):
        super().__init__()

        norm_first = False
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=512,
                nhead=16,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True,
                norm_first=False, ),
            num_layers=6,
            norm=LayerNorm(512) if norm_first else None, )
        self.num_head = 16
        self.position = SinePositionalEmbedding(
            512, dropout=0.1, scale=False, alpha=True)
        self.bert_proj = nn.Linear(1024, 512)
        self.x_proj = nn.Linear(192, 512)
        self.g_emb = nn.Linear(256, 512)

        self.proj = nn.Linear(512, codebook_size, bias=False)

        self.quantize_embedding = nn.Embedding(codebook_size + 1, 512)
        self.n_codes = codebook_size

    def forward_encoder(self, x, mask=None):
        if mask is not None:
            mask = ~mask.squeeze(1).bool()
        x = self.position(x)
        x, _ = self.encoder(
            (x, None), src_key_padding_mask=mask)
        audio_output = self.proj(x)
        return audio_output

    def forward(self, x, x_mask, bert, spk_emb, tgt_code):
        g = self.g_emb(spk_emb).unsqueeze(1)
        bert = self.bert_proj(bert.transpose(1, 2))
        x = self.x_proj((x * x_mask).transpose(1, 2))

        x = x + bert + g

        # make mask
        quantization = tgt_code.clone()

        mask_ratio = random.uniform(0, 1)
        mask = torch.bernoulli(mask_ratio * torch.ones_like(quantization)).bool()
        quantization[mask] = self.n_codes
        quantization_emb = self.quantize_embedding(quantization)

        inp = x + quantization_emb
        audio_output = self.forward_encoder(inp, mask=x_mask)

        x_mask_bool = x_mask.squeeze(1).bool()
        tgt_mask = x_mask_bool & mask

        audio_output = audio_output[tgt_mask]
        tgt_code = tgt_code[tgt_mask].view(-1)
        predict_loss = F.cross_entropy(audio_output, tgt_code, reduction='mean')
        acc = (audio_output.argmax(-1) == tgt_code).float().mean() * 100
        print('prosody acc', acc.item())
        return predict_loss

    @torch.inference_mode()
    @torch.no_grad()
    def infer(self, x, x_mask, bert, spk_emb, step=5, sched_mode="square", temp=30, randomize="None", r_temp=30):

        g = self.g_emb(spk_emb).unsqueeze(1)
        bert = self.bert_proj(bert.transpose(1, 2))
        x = self.x_proj((x * x_mask).transpose(1, 2))
        x = x + bert + g

        nb_sample = x.size(0)
        tgt_length = x.size(1)

        code = torch.full((nb_sample, tgt_length), self.n_codes).long().to(x.device)

        mask = torch.ones((nb_sample, tgt_length)).to(x.device)

        scheduler = self.adap_sche(step, mode=sched_mode, tgt_length=tgt_length)
        for indice, t in enumerate(scheduler):
            # print('step:', indice, '\nmask:', mask, '\nt:', t, '\ncode', code, '\n\n')
            if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                t = int(mask.sum().item())
                print(f"Warning: cannot predict {t} tokens, only {mask.sum().item()} tokens left")
            if mask.sum() == 0:  # Break if code is fully predicted
                break

            quantization_emb = self.quantize_embedding(code.clone())
            inp = x + quantization_emb
            logit = self.forward_encoder(inp)

            prob = torch.softmax(logit * temp, -1)
            distri = torch.distributions.Categorical(probs=prob)
            pred_code = distri.sample()

            conf = torch.gather(prob, 2, pred_code.view(nb_sample, tgt_length, 1))

            if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                ratio = (indice / (len(scheduler) - 1))
                rand = r_temp * np.random.gumbel(size=(nb_sample, tgt_length)) * (1 - ratio)
                conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(x.device)
            elif randomize == "warm_up":  # chose random sample for the 2 first steps
                conf = torch.rand_like(conf) if indice < 2 else conf
            elif randomize == "random":  # chose random prediction at each step
                conf = torch.rand_like(conf)

            # do not predict on already predicted tokens
            conf[~mask.bool()] = -math.inf

            # chose the predicted token with the highest confidence
            tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
            tresh_conf = tresh_conf[:, -1]

            # replace the chosen tokens
            conf = (conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, tgt_length)
            f_mask = (mask.float() * conf.float()).bool()
            code[f_mask] = pred_code[f_mask]

            # update the mask
            for i_mask, ind_mask in enumerate(indice_mask):
                mask[i_mask, ind_mask] = 0
        return code

    def adap_sche(self, step, mode="arccos", leave=False, tgt_length=None):
        """ Create a sampling scheduler
           :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
           :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        r = torch.linspace(1, 0, step)
        if mode == "root":  # root scheduler
            val_to_mask = 1 - (r ** .5)
        elif mode == "linear":  # linear scheduler
            val_to_mask = 1 - r
        elif mode == "square":  # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":  # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":  # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            return

        # fill the scheduler by the ratio of tokens to predict at each step
        sche = (val_to_mask / val_to_mask.sum()) * tgt_length
        sche = sche.round()
        sche[sche == 0] = 1  # add 1 to predict a least 1 token / step
        sche[-1] += tgt_length - sche.sum()  # need to sum up nb of code
        return tqdm(sche.int(), leave=leave)


class SynthesizerTrn(nn.Module):
    """
      Synthesizer for Training
    """

    def __init__(self,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=True,
                 freeze_quantizer=False,
                 **kwargs):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers

        gin_channels = hidden_channels
        self.gin_channels = gin_channels

        self.use_sdp = use_sdp

        self.enc_p = TextEncoder(inter_channels,
                                 hidden_channels,
                                 filter_channels,
                                 n_heads,
                                 n_layers,
                                 kernel_size,
                                 p_dropout)
        print('n_param of enc_p', sum(p.numel() for p in self.enc_p.parameters() if p.requires_grad) / 1e6, 'M')
        self.frame_prior_net = FramePriorNet(inter_channels,
                                             hidden_channels,
                                             filter_channels,
                                             n_heads,
                                             n_layers,
                                             kernel_size,
                                             p_dropout)

        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                             upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16,
                                      gin_channels=gin_channels)
        self.flow = TransformerCouplingBlock(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            4,
            5,
            p_dropout,
            4,
            gin_channels=gin_channels,
        )

        self.ref_enc = modules.MelStyleEncoder(spec_channels, style_vector_dim=gin_channels)
        # self.ref_enc_prosody = modules.MelStyleEncoder(768, style_vector_dim=gin_channels)
        # self.ref_enc = ReferenceEncoder(spec_channels, gin_channels)
        codebook_size = 512
        self.prosody_encoder = ProsodyEncoder(768, hidden_channels, 5, 1, 6, gin_channels, codebook_size)
        self.prosody_predictor = NARPredictor(codebook_size, gin_channels=gin_channels)
        print('n_param of prosody_predictor',
              sum(p.numel() for p in self.prosody_predictor.parameters() if p.requires_grad) / 1e6, 'M')

        # if freeze_quantizer:
        #     print('Freezing quantizer!!!')
        #     print('Freezing quantizer!!!')
        #     print('Freezing quantizer!!!')
        #     print('Freezing quantizer!!!')

        for param in self.prosody_encoder.parameters():
            param.requires_grad = False

    def forward(self, x, x_lengths, y, y_lengths, ssl, duration, bert, spk_emb):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(-1)), 1).to(y.dtype)

        g = self.ref_enc(y * y_mask, y_mask)

        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(-1)), 1).to(y.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)

        attn = commons.generate_path(duration.unsqueeze(1), attn_mask)
        v_mask = (x > 0).float()
        z_q, quantize_loss, indices, dur_loss, pred_dur = self.prosody_encoder(ssl, x_mask, attn, duration, v_mask)

        x, x_mask = self.enc_p(x, x_lengths, bert)
        prosody_predict_loss = self.prosody_predictor(x, x_mask, bert, spk_emb, indices)
        x = x + z_q
        frame_x = torch.matmul(attn.squeeze(1), x.transpose(1, 2)).transpose(1, 2)
        m_p, logs_p = self.frame_prior_net(frame_x, y_mask)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        return o, attn, ids_slice, x_mask, y_mask, (
        z, z_p, m_p, logs_p, m_q, logs_q), z_q, quantize_loss.mean(), dur_loss, pred_dur, prosody_predict_loss

    def reconstruct(self, x, x_lengths, y, y_lengths, ssl, duration, bert, spk_emb, noise_scale=.4):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(-1)), 1).to(y.dtype)
        g = self.ref_enc(y * y_mask, y_mask)

        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(-1)), 1).to(y.dtype)

        # attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        # attn = commons.generate_path(duration.unsqueeze(1), attn_mask)
        # v_mask = (x > 0).float()
        # z_q, quantize_loss, pred_prosody_codes, dur_loss, pred_dur = self.prosody_encoder(ssl, x_mask, attn, duration,v_mask)

        x, x_mask = self.enc_p(x, x_lengths, bert)
        pred_prosody_codes = self.prosody_predictor.infer(x, x_mask, bert, spk_emb)
        z_q, duration = self.prosody_encoder.decode(pred_prosody_codes, x_mask)

        total_duration = torch.sum(duration, dim=1)
        y_lengths = total_duration.long()

        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_lengths.max()), 1).float()
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)

        attn = commons.generate_path(duration.unsqueeze(1), attn_mask)

        x = x + z_q

        frame_x = torch.matmul(attn.squeeze(1), x.transpose(1, 2)).transpose(1, 2)
        m_p, logs_p = self.frame_prior_net(frame_x, y_mask)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask), g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def infer(self, x, x_lengths, ref, bert, spk_emb, noise_scale=.4):
        g = self.ref_enc(ref)

        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(-1)), 1).to(bert.dtype)

        x, x_mask = self.enc_p(x, x_lengths, bert)
        pred_prosody_codes = self.prosody_predictor.infer(x, x_mask, bert, spk_emb)
        z_q, duration = self.prosody_encoder.decode(pred_prosody_codes, x_mask)

        total_duration = torch.sum(duration, dim=1)
        y_lengths = total_duration.long()

        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_lengths.max()), 1).float()
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)

        attn = commons.generate_path(duration.unsqueeze(1), attn_mask)

        x = x + z_q

        frame_x = torch.matmul(attn.squeeze(1), x.transpose(1, 2)).transpose(1, 2)
        m_p, logs_p = self.frame_prior_net(frame_x, y_mask)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask), g=g)
        return o
