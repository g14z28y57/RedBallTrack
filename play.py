import os
import shutil


for dirname in ["image_train", "image_test"]:
    for filename in os.listdir(dirname):
        if "debug" in filename:
            filepath = os.path.join(dirname, filename)
            os.remove(filepath)
