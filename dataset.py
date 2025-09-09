import torch
from torch.utils.data import Dataset
import cv2
import os
from util import read_json, save_pickle, read_pickle
import numpy as np
from tqdm import trange


class DirectionDataset(Dataset):
    def __init__(self, state_dir, image_dir, cache_path, image_encoder=None, device=None):
        length1 = len(os.listdir(state_dir))
        length2 = len(os.listdir(image_dir))
        length = min(length1, length2)
        self.cache_path = cache_path
        # length = 100
        if os.path.exists(cache_path):
            data = read_pickle(cache_path)
            self.state_list = data["state"]
            self.image_list = data["image"]
        else:
            self.state_list = []
            self.image_list = []
            for idx in trange(length):
                state_path = os.path.join(state_dir, f"{idx}.json")
                state = read_json(state_path)
                image_path = os.path.join(image_dir, f"{idx}.png")
                image = cv2.imread(image_path) / 255.0 * 2.0 - 1.0
                image = np.transpose(image, [2, 0, 1])
                if image_encoder is not None:
                    image = torch.tensor(image, dtype=torch.float, device=device).unsqueeze(0)
                    image = image_encoder(image).squeeze(0).cpu().numpy()
                self.state_list.append(state)
                self.image_list.append(image)
            self.cache()
        print(f"{len(self.state_list)} data loaded")

    def __getitem__(self, idx):
        state = self.state_list[idx]
        # sphere_pos = state["sphere_pos"]
        # focal_pt = state["focal_pos"]
        camera_pos = np.array(state["camera_pos"])
        camera_front = np.array(state["camera_front"])
        sphere_dir = np.array(state["sphere_dir"])
        distance = np.array(state["distance"])
        image = self.image_list[idx]
        image = torch.tensor(image, dtype=torch.float)
        camera_pos = torch.tensor(camera_pos, dtype=torch.float)
        camera_front = torch.tensor(camera_front, dtype=torch.float)
        sphere_dir = torch.tensor(sphere_dir, dtype=torch.float)
        distance = torch.tensor(distance, dtype=torch.float).reshape(1)
        return image, camera_pos, camera_front, sphere_dir, distance

    def cache(self):
        data = {
            "state": self.state_list,
            "image": self.image_list
        }

        save_pickle(data, self.cache_path)

    def __len__(self):
        return len(self.state_list)


# if __name__ == "__main__":
#     dataset = DirectionDataset(state_dir="state", image_dir="image")
#     x = dataset[0]
#     from IPython import embed
#     embed()
