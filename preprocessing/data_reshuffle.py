import os
from shutil import copyfile, move
import re

# copied_image_count = 0
# n_images_to_copy = 1000
# src = "/data/unzipped_celeba/celeba/train/1"
# dest = "/data/test_images/"
#
# for item in os.listdir(src):
#     copied_image_count += 1
#     src_file = src + item
#     dest_file = dest + item
#     copyfile(src_file, dest_file)
#     if copied_image_count > n_images_to_copy:
#         break
#
# for item in os.listdir("/data"):
#     if re.search("original", item) is not None:
#         move("/data/" + item, "/data/test_images/" + item)
