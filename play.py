import os


dst_dir = "texture_test"
for i in range(12002, 21267):
    src_path = os.path.join(dst_dir, f"{i}.png")
    dst_path = os.path.join(dst_dir, f"{(i-12002)}.png")
    os.rename(src_path, dst_path)
