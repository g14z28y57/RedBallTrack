import cv2
import os
from tqdm import trange


def create_video_from_images(img_paths, video_path, fps):
    frame = cv2.imread(img_paths[0])
    height, width, layers = frame.shape

    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, size)

    for img_path in img_paths:
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()


def main():
    os.makedirs("videos", exist_ok=True)
    for idx in trange(50):
        img_dir = f"inference/inference_{idx}"
        num_frames = 300
        img_paths = [os.path.join(img_dir, f"{i}.png") for i in range(num_frames)]
        assert len(img_paths) > 0
        video_path = f'videos/video_{idx}.mp4'
        create_video_from_images(img_paths, video_path, fps=30)


if __name__ == "__main__":
    main()
