import os.path

from torch.utils.data import DataLoader
from model import DirectionModel
from dataset import DirectionDataset
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


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

def visualize(data, img_path, bins=50):
    # 绘制直方图
    # bins 参数用来指定区间的数量
    plt.hist(data, bins=bins, edgecolor='black')

    # 添加标题和标签
    plt.title('Hist')
    plt.xlabel('value')
    plt.ylabel('number')

    # 显示图表
    plt.savefig(img_path)


@torch.inference_mode()
def test():
    device = torch.device("cuda")
    batch_size = 1
    dataset = DirectionDataset(state_dir="state_test", image_dir="image_test", cache_path="data_test.pkl")
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    loss_fn_dir = torch.nn.MSELoss()  # For direction, which is a regression task
    loss_fn_dist = torch.nn.MSELoss()  # For distance, which is also regression

    model = DirectionModel(d_input=1506,
                           d_model=256,
                           d_feedforward=1024,
                           out_channels=5,
                           num_layers=5)
    checkpoint_pth = "direction_model.pth"
    assert os.path.exists(checkpoint_pth)
    model.load_state_dict(torch.load(checkpoint_pth))
    model = model.to(device)
    model.eval()

    loss_dir_list = []
    loss_dist_list = []

    for image, camera_pos, camera_front, sphere_dir, distance in tqdm(dataloader):
        image = image.to(device)
        camera_pos = camera_pos.to(device)
        camera_front = camera_front.to(device)
        out_dir, out_dist = model(image, camera_pos, camera_front)
        sphere_dir = sphere_dir.to(device)
        distance = distance.to(device)

        loss_dir = loss_fn_dir(out_dir, sphere_dir).item()
        loss_dist = loss_fn_dist(out_dist, distance).item()

        loss_dir_list.append(loss_dir)
        loss_dist_list.append(loss_dist)

    compute_stat(loss_dir_list)
    visualize(loss_dir_list, "loss_direction.png")
    compute_stat(loss_dist_list)
    visualize(loss_dist_list, "loss_distance.png")


if __name__ == "__main__":
    test()
