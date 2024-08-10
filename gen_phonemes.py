import multiprocessing
import os.path
from glob import glob

import torch
from tqdm import tqdm

from data_conf import data_root
from text.bert import get_bert_feature
from text.cleaner import clean_text
import numpy as np
from multiprocessing import Pool

out_dir = "dump"
os.makedirs(out_dir, exist_ok=True)
phoneme_path = os.path.join(out_dir, "phoneme.npy")
phone_dict = {}


def process_file(data):
    wav_path, language = data
    lab_path = wav_path.replace(".wav", ".lab")
    if os.path.exists(lab_path):
        text = open(lab_path).readline().strip()
        if '{' in text or '^' in text:
            print(f"Error, {text}, {wav_path}")
            return None
        try:
            phones, word2ph, norm_text = clean_text(text, language)

            rank = multiprocessing.current_process()._identity
            rank = rank[0] if len(rank) > 0 else 0
            gpu_id = rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{gpu_id}")
            bert_feature = get_bert_feature(norm_text, word2ph, device, language)
            torch.save(bert_feature.cpu(), wav_path.replace(".wav", ".bert.pt"))

            phones = " ".join(phones)
            return (wav_path, phones)
        except Exception as e:
            print(f"Error in {wav_path}, {text}", e)
            return None
    else:
        return None
if __name__ == '__main__':

    for language in ['zh']:
        filenames = glob(f"{data_root}/{language}/**/*.wav", recursive=True)

        # Define the number of processes to use
        num_processes = 8  # You can adjust this as needed
        # multiprocessing.set_start_method("spawn", force=True)

        with Pool(num_processes) as pool:
            results = list(tqdm(pool.imap(process_file, [(f, language) for f in filenames]), total=len(filenames)))

        for result in results:
            if result is not None:
                phone_dict[result[0]] = result[1]
    # 输出前10个
    for k, v in list(phone_dict.items())[:10]:
        print(k, v)
    np.save(phoneme_path, phone_dict)
