import csv
import os
import shutil

split_dir = "./data/dataset/old_split"
new_dir = "./data/dataset/split"
aux_dir = '.data/dataset/old_split2'

with open('new_subdata_2.csv', newline='') as f:
    reader = list(csv.reader(f))
    sample = [x[0] for x in reader]

# os.mkdir(new_dir)

for basename in sample:
    os.rename(os.path.join(split_dir, basename), os.path.join(new_dir, basename))

# shutil.move(split_dir, aux_dir)
# shutil.move(new_dir, split_dir)
