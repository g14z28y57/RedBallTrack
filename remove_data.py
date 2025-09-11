import os


def main():
    image_dir = "image_train"
    for name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, name)
        