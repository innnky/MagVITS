from glob import glob
from random import shuffle
from data_conf import data_root
from tqdm import tqdm
import os
filenames = glob(f"{data_root}/**/*.wav", recursive=True)  # [:10]
filenames += glob(f"{data_root}/**/*.mp3", recursive=True)  # [:10]
filenames = [f for f in tqdm(filenames)]
shuffle(filenames)

val_num = 4
print(len(filenames))
train = filenames[:-val_num]
val = filenames[-val_num:]
train.sort()
val.sort()

with open('dump/train_files.list', 'w') as f:
    f.write('\n'.join(train))
with open('dump/val_files.list', 'w') as f:
    f.write('\n'.join(val))


