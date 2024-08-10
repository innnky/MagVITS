from glob import glob
from random import shuffle
from data_conf import data_root
from tqdm import tqdm
import os
filenames = glob(f"{data_root}/**/*.wav", recursive=True)  # [:10]
filenames = [f for f in tqdm(filenames) if '/zh/' in f and os.path.exists(f.replace('.wav', '.lab'))]
shuffle(filenames)
    
val_num = 12
print(len(filenames))
train = filenames[:-val_num]
val = filenames[-val_num:]
train.sort()
val.sort()

with open('dump/s2_train_files.list', 'w') as f:
    f.write('\n'.join(train))
with open('dump/s2_val_files.list', 'w') as f:
    f.write('\n'.join(val))


