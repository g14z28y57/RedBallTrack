import os.path
from util import read_json
from torch.utils.data import DataLoader
from model import VisionBackbone, DirectionModel
from dataset import DirectionDataset
import torch


def train(config):
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    log_every = config["training"]["log_every"]
    save_every = config["training"]["save_every"]
    lr = config["training"]["lr"]
    checkpoint_pth = config["training"]["model_checkpoint"]
    optimizer_state_pth = config["training"]["optimizer_checkpoint"]
    device = config["device"]
    num_data = config["training"]["num_data"]

    d_input = config["model"]["d_input"]
    d_model = config["model"]["d_model"]
    d_feedforward = config["model"]["d_feedforward"]
    out_layer = config["model"]["output_layer"]
    out_channels = config["model"]["out_channels"]
    num_layers = config["model"]["num_layers"]

    backbone = VisionBackbone(out_layer=out_layer).to(device)
    dataset = DirectionDataset(state_dir="state_train",
                               image_dir="image_train",
                               cache_path="data_train.pkl",
                               num_data=num_data,
                               image_encoder=backbone,
                               device=device)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    loss_fn_dir = torch.nn.L1Loss()  # For direction, which is a regression task
    loss_fn_dist = torch.nn.L1Loss()  # For distance, which is also regression

    model = DirectionModel(d_input=d_input,
                           d_model=d_model,
                           d_feedforward=d_feedforward,
                           out_channels=out_channels,
                           num_layers=num_layers)

    if os.path.exists(checkpoint_pth):
        model.load_state_dict(torch.load(checkpoint_pth))
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam is a good general-purpose optimizer

    if os.path.exists(optimizer_state_pth):
        optimizer.load_state_dict(torch.load(optimizer_state_pth))

    count = 0
    losses_dir = []
    losses_dist = []

    for epoch in range(num_epochs):
        for img_feature, camera_pos, camera_front, cylinder_dir, distance in dataloader:
            img_feature = img_feature.to(device)
            camera_pos = camera_pos.to(device)
            camera_front = camera_front.to(device)
            optimizer.zero_grad()
            out_dir, out_dist = model(img_feature, camera_pos, camera_front)
            cylinder_dir = cylinder_dir.to(device)
            distance = distance.to(device)

            loss_dir = loss_fn_dir(out_dir, cylinder_dir)
            loss_dist = loss_fn_dist(out_dist, distance)
            total_loss = loss_dir
            losses_dir.append(loss_dir.item())
            losses_dist.append(loss_dist.item())
            total_loss.backward()
            optimizer.step()

            if count > 0 and count % log_every == 0:
                avg_loss_dir = sum(losses_dir) / len(losses_dir)
                avg_loss_dist = sum(losses_dist) / len(losses_dist)
                print(f"epoch: {epoch}, step: {count}, direction loss: {avg_loss_dir:.4f}, distance loss: {avg_loss_dist:.4f}")
                losses_dir = []
                losses_dist = []

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
