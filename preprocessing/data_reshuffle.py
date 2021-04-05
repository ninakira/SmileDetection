import os
from shutil import copyfile

copied_image_count = 0
n_images_to_copy = 1000
src = "/data/unzipped_celeba/celeba/train/1"
dest = "/data/test_images"

for item in os.listdir(src):
    copied_image_count += 1
    src_file = src + item
    dest_file = dest + item + 'original'
    copyfile(src_file, dest_file)
    if copied_image_count > n_images_to_copy:
        break
