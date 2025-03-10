import json
import os

label_dir = os.path.join("/root/MonoUNI/dataset/Rope3d")
namelist = os.listdir(os.path.join(label_dir, "label_2"))
with open(os.path.join(label_dir, "ImageSets/all.txt"), "w") as f:
    for file in namelist:
        f.write(file.split(".")[0] + "\n")
