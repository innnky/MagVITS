import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning.core.module as pl
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import torch.nn.functional as F
import jiwer
from PIL import Image

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)


def drop_duplicated(chars):
    ret_chars = [chars[0]]
    for prev, curr in zip(chars[:-1], chars[1:]):
        if prev != curr:
            ret_chars.append(curr)
    return ret_chars
def calc_wer(target, pred, ignore_indexes=[0]):
    target_chars = drop_duplicated(list(filter(lambda x: x not in ignore_indexes, map(str, list(target)))))
    pred_chars = drop_duplicated(list(filter(lambda x: x not in ignore_indexes, map(str, list(pred)))))
    target_str = ' '.join(target_chars)
    pred_str = ' '.join(pred_chars)
    error = jiwer.wer(target_str, pred_str)
    return error

class ASRTrainer(pl.LightningModule):
    def __init__(self, model, criterion, mono_start_epoch, lr):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.mono_start_epoch = mono_start_epoch
        self.lr=lr


    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state_dict["state_dict"])

    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm

    @staticmethod
    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

    def calc_mono_loss(self, s2s_attn,input_lengths, mel_input_length, text_mask, mel_mask, n_down):
        s2s_attn = s2s_attn.transpose(-1, -2)
        s2s_attn = s2s_attn[..., 1:]
        s2s_attn = s2s_attn.transpose(-1, -2)

        with torch.no_grad():
            attn_mask = (~mel_mask).unsqueeze(-1).expand(mel_mask.shape[0], mel_mask.shape[1],
                                                     text_mask.shape[-1]).float().transpose(-1, -2)
            attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0],
                                                                              text_mask.shape[1],
                                                                              mel_mask.shape[-1]).float()
            attn_mask = (attn_mask < 1)

        s2s_attn.masked_fill_(attn_mask, 0.0)

        with torch.no_grad():
            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length)
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
        loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

        return loss_mono, s2s_attn_mono

    def get_attention_mono(self, text_input, text_input_length, mel_input, mel_input_length):
        mel_input_length = mel_input_length // (2 ** self.model.n_down)
        future_mask = self.model.get_future_mask(
            mel_input.size(2) // (2 ** self.model.n_down), unmask_future_steps=0).to(self.device)
        mel_mask = self.model.length_to_mask(mel_input_length)
        text_mask = self.model.length_to_mask(text_input_length)
        ppgs, s2s_pred, s2s_attn = self.model(
            mel_input, src_key_padding_mask=mel_mask, text_input=text_input)
        loss_mono, s2s_attn_mono = self.calc_mono_loss(s2s_attn, text_input_length, mel_input_length, text_mask, mel_mask, self.model.n_down)
        return s2s_attn_mono


    @staticmethod
    def get_image(arrs):
        pil_images = []
        height = 0
        width = 0
        for arr in arrs:
            uint_arr = (((arr - arr.min()) / (arr.max() - arr.min())) * 255).astype(np.uint8)
            pil_image = Image.fromarray(uint_arr)
            pil_images.append(pil_image)
            height += uint_arr.shape[0]
            width = max(width, uint_arr.shape[1])

        palette = Image.new('L', (width, height))
        curr_heigth = 0
        for pil_image in pil_images:
            palette.paste(pil_image, (0, curr_heigth))
            curr_heigth += pil_image.size[1]

        return palette

    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    print("%s does not have same shape" % key)
                    print(val.shape, model_states[key].shape)
                    if not force_load:
                        continue

                    min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                print("not exist :%s" % key)

    def configure_optimizers(self):
        optimizer = optim.AdamW([*self.parameters()], lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1)
        return optimizer

    def training_step(self, batch, batch_idx):
        text_input, text_input_length, mel_input, mel_input_length = batch

        mel_input_length = mel_input_length // (2 ** self.model.n_down)
        future_mask = self.model.get_future_mask(
            mel_input.size(2) // (2 ** self.model.n_down), unmask_future_steps=0).to(self.device)
        mel_mask = self.model.length_to_mask(mel_input_length)
        text_mask = self.model.length_to_mask(text_input_length)
        ppgs, s2s_pred, s2s_attn = self.model(
            mel_input, src_key_padding_mask=mel_mask, text_input=text_input)
        loss_mono, s2s_attn_mono = self.calc_mono_loss(s2s_attn, text_input_length, mel_input_length, text_mask,
                                                       mel_mask, self.model.n_down)
        loss_ctc = self.criterion['ctc'](ppgs.log_softmax(dim=2).transpose(0, 1),
                                         text_input, mel_input_length, text_input_length)

        loss_s2s = 0
        for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
            loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
        loss_s2s /= text_input.size(0)
        if  self.current_epoch > self.mono_start_epoch:
            loss_ctc = loss_ctc * 0
        else:
            loss_mono = loss_mono * 0
        loss = loss_ctc + loss_s2s + loss_mono


        self.log("train/loss_ctc", loss_ctc, on_step=True, prog_bar=True)
        self.log("train/loss_s2s", loss_s2s, on_step=True, prog_bar=True)
        self.log("train/loss_mono", loss_mono, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        text_input, text_input_length, mel_input, mel_input_length = batch
        mel_input_length = mel_input_length // (2 ** self.model.n_down)
        future_mask = self.model.get_future_mask(
            mel_input.size(2) // (2 ** self.model.n_down), unmask_future_steps=0).to(self.device)
        mel_mask = self.model.length_to_mask(mel_input_length)
        text_mask = self.model.length_to_mask(text_input_length)
        ppgs, s2s_pred, s2s_attn = self.model(
            mel_input, src_key_padding_mask=mel_mask, text_input=text_input)
        loss_mono, s2s_attn_mono = self.calc_mono_loss(s2s_attn, text_input_length, mel_input_length, text_mask,
                                                       mel_mask, self.model.n_down)
        loss_ctc = self.criterion['ctc'](ppgs.log_softmax(dim=2).transpose(0, 1),
                                         text_input, mel_input_length, text_input_length)
        loss_s2s = 0
        for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
            loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
        loss_s2s /= text_input.size(0)
        loss = loss_ctc + loss_s2s + loss_mono

        self.log("val/ctc", loss_ctc.item(), on_step=False, prog_bar=True)
        self.log("val/s2s", loss_s2s.item(), on_step=False, prog_bar=True)
        self.log("val/loss", loss.item(), on_step=False, prog_bar=True)
        self.log("val/mono", loss_mono.item(), on_step=False, prog_bar=True)

        _, amax_ppgs = torch.max(ppgs, dim=2)
        wers = [calc_wer(target[:text_length],
                         pred[:mel_length],
                         ignore_indexes=list(range(5))) \
                for target, pred, text_length, mel_length in zip(
                text_input.cpu(), amax_ppgs.cpu(), text_input_length.cpu(), mel_input_length.cpu())]
        self.log("val/wers", np.mean(wers), on_step=False, prog_bar=True)

        _, amax_s2s = torch.max(s2s_pred, dim=2)
        acc = [torch.eq(target[:length], pred[:length]).float().mean().item() \
               for target, pred, length in zip(text_input.cpu(), amax_s2s.cpu(), text_input_length.cpu())]

        self.log("val/acc", np.mean(acc), on_step=False, prog_bar=True)
        attn_img = self.get_image([s2s_attn[0].cpu().numpy()])
        attn_mono_img = self.get_image([s2s_attn_mono[0].cpu().numpy()])
        # self.logger.experiment.add_image("val/attn", attn_img, self.current_epoch)
        # self.logger.experiment.add_image("val/attn_mono", attn_mono_img, self.current_epoch)
        self.logger.log_image(key="attn", images=[attn_img, attn_mono_img], caption=["soft", "mono"])
