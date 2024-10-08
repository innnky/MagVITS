import time
import os
import random
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from module import commons
from module.mel_processing import spectrogram_torch
from text import cleaned_text_to_sequence
from text.symbols import v
from utils import load_wav_to_torch, load_filepaths_and_text
import torch.nn.functional as F

f0_bin = 64
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0):
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    # torch.clip_(f0_mel, min=1., max=float(f0_bin - 1))
    f0_coarse = torch.round(f0_mel).long()
    f0_coarse = f0_coarse * (f0_coarse > 0)
    f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
    f0_coarse = f0_coarse * (f0_coarse < f0_bin)
    f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))
    return f0_coarse


"""Multi speaker version"""


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text, hparams, get_path=False, meta=None, val=False,
                 phoneme_path='dump/phoneme.npy'):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.val = val

        self.get_path = get_path
        self.meta = meta
        self.phoneme_data = np.load(phoneme_path, allow_pickle=True).item()

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        print('only chinese and skip other languages!!!!!!!!')
        print('only chinese and skip other languages!!!!!!!!')
        print('only chinese and skip other languages!!!!!!!!')
        print('only chinese and skip other languages!!!!!!!!')

        print("phoneme_data_len:", len(self.phoneme_data.keys()))
        print("wav_data_len:", len(self.audiopaths_sid_text))

        audiopaths_sid_text_new = []
        lengths = []
        skipped = 0
        for item in tqdm(self.audiopaths_sid_text):
            audiopath = item[0]
            if 'zh' not in audiopath:
                skipped += 1
                continue
            try:
                phoneme = self.phoneme_data[audiopath]
                phoneme = phoneme.split(' ')
                phoneme_ids = cleaned_text_to_sequence(phoneme)
            except Exception:
                skipped += 1
                continue

            bert_path = audiopath.replace('.wav', '.bert.pt').replace('.mp3', '.bert.pt')
            if not os.path.exists(bert_path):
                skipped += 1
                continue
            duration_path = audiopath.replace('.wav', '.dur.pt').replace('.mp3', '.dur.pt')
            if not os.path.exists(duration_path):
                skipped += 1
                continue
            sslpath = audiopath.replace('.wav', '.ssl.pt').replace('.mp3', '.ssl.pt')
            if os.path.exists(audiopath) and os.path.exists(sslpath) and (
                    os.path.getsize(audiopath) / self.sampling_rate / 2 > 0.6 or self.val):
                audiopaths_sid_text_new.append([audiopath, sslpath, bert_path, phoneme_ids, duration_path])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            else:
                skipped += 1
        print("skipped: ", skipped, ", total: ", len(self.audiopaths_sid_text))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sslpath, bert_path, phoneme_ids, duration_path = audiopath_sid_text
        phoneme_ids = commons.intersperse(phoneme_ids, 0)

        text = torch.LongTensor(phoneme_ids)
        try:
            spec, wav = self.get_audio(audiopath)
        except:
            spec = torch.zeros(1025, 100)
            wav = torch.zeros(1, 100 * self.hop_length)
            print("load audio error!!!!!!", audiopath)

        duration = torch.load(duration_path)
        assert duration.shape[0] == len(phoneme_ids), (duration.shape, phoneme_ids.shape)
        total_len = duration.sum().item()

        assert abs(total_len - spec.shape[-1]) < 3, (total_len, spec.shape[-1], audiopath)
        if spec.shape[-1] < total_len:
            spec = F.pad(spec, (0, total_len - spec.shape[-1]))
            wav = F.pad(wav, (0, total_len * self.hop_length - wav.shape[-1]))
        elif spec.shape[-1] > total_len:
            spec = spec[:, :total_len]
            wav = wav[:, :total_len * self.hop_length]

        ssl = torch.load(sslpath)
        ssl = F.interpolate(ssl, size=spec.shape[-1], mode="nearest")

        bert_feature = torch.load(bert_path)
        bert_feature = F.interpolate(bert_feature.unsqueeze(0), scale_factor=2, mode='nearest')
        bert_feature = F.pad(bert_feature, (0, 1), value=0).squeeze(0)

        assert bert_feature.shape[-1] == len(phoneme_ids)

        spk_emb = np.load(audiopath.replace('.wav', '.spk.npy').replace('.mp3', '.spk.npy'))
        spk_emb = torch.FloatTensor(spk_emb)

        return (ssl, spec, wav, text, bert_feature, spk_emb, duration)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(audio_norm, self.filter_length,
                                 self.sampling_rate, self.hop_length, self.win_length,
                                 center=False)
        spec = torch.squeeze(spec, 0)
        return spec, audio_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])
        max_text_len = max([x[3].size(0) for x in batch])

        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        text_lengths = torch.LongTensor(len(batch))

        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        ssl_padded = torch.FloatTensor(len(batch), 768, max_spec_len)
        text_padded = torch.LongTensor(len(batch), max_text_len)
        bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        spk_emb_padded = torch.FloatTensor(len(batch), 256)
        duration_padded = torch.LongTensor(len(batch), max_text_len)

        spec_padded.zero_()
        wav_padded.zero_()
        ssl_padded.zero_()
        text_padded.zero_()
        bert_padded.zero_()
        spk_emb_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            ssl = row[0]
            ssl_padded[i, :, :ssl.size(2)] = ssl[0, :, :]

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            text = row[3]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            bert = row[4]
            bert_padded[i, :, :bert.size(-1)] = bert

            spk_emb = row[5]
            spk_emb_padded[i] = spk_emb

            duration = row[6]
            duration_padded[i, :duration.size(0)] = duration

        return ssl_padded, spec_padded, spec_lengths, wav_padded, wav_lengths, text_padded, text_lengths, bert_padded, spk_emb_padded, duration_padded


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
