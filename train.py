import os.path

from torch.utils.data import DataLoader
from model import DirectionModel
from dataset import DirectionDataset
import torch


def train():
    device = torch.device("cuda")
    batch_size = 64
    dataset = DirectionDataset(state_dir="state", image_dir="image")
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    loss_fn_dir = torch.nn.MSELoss()  # For direction, which is a regression task
    loss_fn_dist = torch.nn.MSELoss()  # For distance, which is also regression

    model = DirectionModel(d_input=1506,
                           d_model=256,
                           d_feedforward=1024,
                           out_channels=5,
                           num_layers=5)

    checkpoint_pth = "direction_model.pth"
    if os.path.exists(checkpoint_pth):
        model.load_state_dict(torch.load(checkpoint_pth))

    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Adam is a good general-purpose optimizer

    num_epochs = 1000
    count = 0
    log_every = 50
    save_every = 1000

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
            optimizer.step()

            if count > 0 and count % log_every == 0:
                avg_loss_dir = sum(losses_dir) / len(losses_dir)
                losses_dir = []
                avg_loss_dist = sum(losses_dist) / len(losses_dist)
                losses_dist = []
                print(f"loss of direction: {avg_loss_dir:.4f}, loss of distance: {avg_loss_dist:.4f}")

            if count > 0 and count % save_every == 0:
                torch.save(model.state_dict(), checkpoint_pth)
                print("Training finished and model saved!")

            count += 1


if __name__ == "__main__":
    train()
