import os.path
from util import read_json
from torch.utils.data import DataLoader
from model import VisionBackbone, DirectionModel
from dataset import DirectionDataset
import torch
from tqdm import tqdm
from util import plot_hist


def compute_stat(data):
    data_min = min(data)
    data_max = max(data)
    data_m1 = sum(data)
    data_m2 = sum([item ** 2 for item in data])
    data_len = len(data)
    data_mean = data_m1 / data_len
    data_variance = data_m2 / data_len - data_mean ** 2
    data_std = data_variance ** 0.5
    print(f"min: {data_min}, max: {data_max}, mean: {data_mean}, std: {data_std}")


@torch.inference_mode()
def test(config):
    device = torch.device("cuda")
    batch_size = 1

    d_input = config["model"]["d_input"]
    d_model = config["model"]["d_model"]
    d_feedforward = config["model"]["d_feedforward"]
    out_layer = config["model"]["output_layer"]
    out_channels = config["model"]["out_channels"]
    num_layers = config["model"]["num_layers"]

    backbone = VisionBackbone(out_layer=out_layer).to(device)
    dataset = DirectionDataset(state_dir="state_test",
                               image_dir="image_test",
                               cache_path="data_test.pkl",
                               image_encoder=backbone,
                               device=device)
    dataset.plot_distance_hist()
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    loss_fn_dir = torch.nn.MSELoss()  # For direction, which is a regression task
    loss_fn_dist = torch.nn.L1Loss()  # For distance, which is also regression

    model = DirectionModel(d_input=d_input,
                           d_model=d_model,
                           d_feedforward=d_feedforward,
                           out_channels=out_channels,
                           num_layers=num_layers)

    checkpoint_pth = "direction_model.pth"
    assert os.path.exists(checkpoint_pth)
    model.load_state_dict(torch.load(checkpoint_pth))
    model = model.to(device)
    model.eval()

    loss_dir_list = []
    loss_dist_list = []

    for img_feature, camera_pos, camera_front, cylinder_dir, distance in tqdm(dataloader):
        img_feature = img_feature.to(device)
        camera_pos = camera_pos.to(device)
        camera_front = camera_front.to(device)
        out_dir, out_dist = model(img_feature, camera_pos, camera_front)
        cylinder_dir = cylinder_dir.to(device)
        distance = distance.to(device)

        loss_dir = loss_fn_dir(out_dir, cylinder_dir).item()
        loss_dist = loss_fn_dist(out_dist, distance).item()

        loss_dir_list.append(loss_dir)
        loss_dist_list.append(loss_dist)

    print("stats of direction loss")
    compute_stat(loss_dir_list)
    plot_hist(loss_dir_list, "loss_direction.png")

    print("stats of distance loss")
    compute_stat(loss_dist_list)
    plot_hist(loss_dist_list, "loss_distance.png")


def main():
    config_path = "config.json"
    config = read_json(config_path)
    test(config)


if __name__ == "__main__":
    main()
