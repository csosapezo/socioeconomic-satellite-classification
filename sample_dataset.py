import csv
import os

split_dir = "./data/dataset/split"
new_dir = "./data/dataset/split2"
aux_dir = '.data/dataset/old_split'

with open('subdata.csv', newline='') as f:
    reader = list(csv.reader(f))
    sample = [x[0] for x in reader]

os.mkdir(new_dir)

for basename in sample:
    os.rename(os.path.join(split_dir, basename), os.path.join(new_dir, basename))

os.rename(split_dir, aux_dir)
os.rename(new_dir, split_dir)
