import cv2
import os


def cut_images():
    img_dir = "texture_images"
    dst_dir = "texture_train"
    num = len(os.listdir(img_dir))
    patch_idx = 0
    for img_idx in range(num):
        img_path = os.path.join(img_dir, f"{img_idx}.png")
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        ph = h // 4
        pw = w // 4
        for i in range(4):
            for j in range(4):
                patch = img[i * ph: (i + 1) * ph, j * pw: (j + 1) * ph, :]
                save_path = os.path.join(dst_dir, f"{patch_idx}.png")
                cv2.imwrite(save_path, patch)
                patch_idx += 1


if __name__ == "__main__":
    cut_images()
