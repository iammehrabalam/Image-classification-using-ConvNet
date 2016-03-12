'''
Copies training images from the downloaded aircraft and nabirds datasets
into the rest of the challenge data.
'''
import os.path
import pickle
import shutil
import sys

nabirds_path = 'CCUB_NABIRDS_EBYB/train_images/'
aircraft_path = 'fgvc-aircraft-2013a/data/images/'
fgcomp_train_im_path = 'train_images'

path_dict = {'aircraft':aircraft_path,'birds':nabirds_path}
total_symbols = 50

with open('licensed_ims_train.pkl', 'rb') as im_file:
  ims = pickle.load(im_file)
  num_ims = len(ims)
  for i, (domain, old_path, new_path) in enumerate(ims):
    part_done = float(i+1) / num_ims
    percent_done = int(round(100 * part_done))
    num_print = int(round(part_done * total_symbols))
    sys.stdout.write("\r[%-50s] %d%% (%d/%d)\r" % ('='*num_print, percent_done, i+1, num_ims))
    new_path = os.path.join(fgcomp_train_im_path, new_path)
    old_path = os.path.join(path_dict[domain], old_path)
    parent_dir = os.path.split(new_path)[0]
    if not os.path.exists(parent_dir):
      os.makedirs(parent_dir)
    shutil.copyfile(old_path, new_path)
print
