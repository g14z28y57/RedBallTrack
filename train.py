import os.path
from util import read_json
from torch.utils.data import DataLoader
from model import DirectionModel
from dataset import DirectionDataset
import torch


def train(config):
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    log_every = config["training"]["log_every"]
    save_every = config["training"]["save_every"]
    lr = config["training"]["lr"]

    d_input = config["model"]["d_input"]
    d_model = config["model"]["d_model"]
    d_feedforward = config["model"]["d_feedforward"]
    out_channels = config["model"]["out_channels"]
    num_layers = config["model"]["num_layers"]

    device = torch.device("cuda")
    dataset = DirectionDataset(state_dir="state_train", image_dir="image_train", cache_path="data_train.pkl")
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    loss_fn_dir = torch.nn.MSELoss()  # For direction, which is a regression task
    loss_fn_dist = torch.nn.MSELoss()  # For distance, which is also regression

    model = DirectionModel(d_input=d_input,
                           d_model=d_model,
                           d_feedforward=d_feedforward,
                           out_channels=out_channels,
                           num_layers=num_layers)
    checkpoint_pth = "direction_model.pth"
    if os.path.exists(checkpoint_pth):
        model.load_state_dict(torch.load(checkpoint_pth))
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam is a good general-purpose optimizer
    optimizer_state_pth = "optimizer.pth"
    if os.path.exists(optimizer_state_pth):
        optimizer.load_state_dict(torch.load(optimizer_state_pth))

    count = 0

    losses_dir = []
    losses_dist = []

    for epoch in range(num_epochs):
        for image, camera_pos, camera_front, sphere_dir, distance in dataloader:
            image = image.to(device)
            camera_pos = camera_pos.to(device)
            camera_front = camera_front.to(device)
            optimizer.zero_grad()
            out_dir, out_dist = model(image, camera_pos, camera_front)
            sphere_dir = sphere_dir.to(device)
            distance = distance.to(device)

            loss_dir = loss_fn_dir(out_dir, sphere_dir)
            loss_dist = loss_fn_dist(out_dist, distance)
            total_loss = loss_dir + loss_dist
            losses_dir.append(loss_dir.item())
            losses_dist.append(loss_dist.item())
            total_loss.backward()

            if count > 0 and count % log_every == 0:
                grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += (p.grad.norm(2)) ** 2  # .norm(2) 计算 L2 范数
                grad_norm = grad_norm ** 0.5

                avg_loss_dir = sum(losses_dir) / len(losses_dir)
                losses_dir = []
                avg_loss_dist = sum(losses_dist) / len(losses_dist)
                avg_loss = avg_loss_dist + avg_loss_dir
                losses_dist = []
                print(f"epoch: {epoch}, step: {count}, loss: {avg_loss:.4f}, loss of direction: {avg_loss_dir:.4f},"
                      f" loss of distance: {avg_loss_dist:.4f}, grad_norm: {grad_norm:.4f}")

            optimizer.step()

            if count > 0 and count % save_every == 0:
                torch.save(model.state_dict(), checkpoint_pth)
                torch.save(optimizer.state_dict(), optimizer_state_pth)
                print("model saved")

            count += 1


def main():
    config_path = "config.json"
    config = read_json(config_path)
    train(config)


if __name__ == "__main__":
    main()
