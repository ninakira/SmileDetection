import os
from shutil import copyfile

src = "/data/unzipped_celeba/celeba/train/1"
dest = "/data/augmented_celeba/train/1"

for item in os.listdir("/data/unzipped_celeba/celeba/train/1"):
    src_file = src + item
    dest_file = dest + item + 'original'
    copyfile(src_file, dest_file)

# >>> print(len(os.listdir("/data/unzipped_celeba/celeba/train/1")))
# 88069
# >>> print(len(os.listdir("/data/augmented_celeba/train/1")))
# 373603

# >>> print(len(os.listdir("/data/unzipped_celeba/celeba/train/0")))
# 94258
# p>>> print(len(os.listdir("/data/augmented_celeba/train/0")))
# 391004
