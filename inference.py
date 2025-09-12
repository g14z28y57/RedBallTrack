import os
import torch
from model import VisionBackbone, DirectionModel
import cv2
import numpy as np
import random
import math
from util import compute_direction, save_json, read_json
from collect_data import take_photo


def create_model(config):
    d_input = config["model"]["d_input"]
    d_model = config["model"]["d_model"]
    d_feedforward = config["model"]["d_feedforward"]
    out_layer = config["model"]["output_layer"]
    out_channels = config["model"]["out_channels"]
    num_layers = config["model"]["num_layers"]
    device = config["device"]
    checkpoint_pth = config["training"]["model_checkpoint"]

    backbone = VisionBackbone(out_layer=out_layer).to(device)
    backbone.eval()

    model = DirectionModel(d_input=d_input,
                           d_model=d_model,
                           d_feedforward=d_feedforward,
                           out_channels=out_channels,
                           num_layers=num_layers)
    assert os.path.exists(checkpoint_pth)
    model_state = torch.load(checkpoint_pth)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    
    return backbone, model, device


@torch.inference_mode()
def step_one(backbone, model, device, img_path, camera_pos, focal_pos, camera_front, step_distance):
    img = cv2.imread(img_path) / 255.0 * 2.0 - 1.0
    img = np.transpose(img, [2, 0, 1])
    img_feature = torch.tensor(img, dtype=torch.float, device=device).unsqueeze(0)
    img_feature = backbone(img_feature)
    camera_pos_tensor = torch.tensor(camera_pos, dtype=torch.float, device=device).unsqueeze(0)
    camera_front_tensor = torch.tensor(camera_front, dtype=torch.float, device=device).unsqueeze(0)
    out_direction, distance = model(img_feature, camera_pos_tensor, camera_front_tensor)
    out_direction = out_direction.cpu().numpy().squeeze(0)

    # how to update camera pos and focal pos
    step = step_distance * out_direction
    camera_pos = camera_pos + step
    focal_pos = focal_pos + step * 2
    # camera_front = camera_front + out_direction
    # camera_front = camera_front / np.linalg.norm(camera_front)
    return camera_pos, focal_pos, out_direction
    

@torch.inference_mode()
def run(config, image_dir, state_dir, num_steps):
    # 环境配置
    plane_texture_path = "plane.jpg"
    plane_size = 20
    cylinder_radius = 1.0
    cylinder_height = 0.5
    
    center_x = random.uniform(-4, 4)
    center_y = random.uniform(-4, 4)
    start_idx = random.randint(0, 8)
    
    gap = 3.5
    cylinder_position_list = []
    for n in range(start_idx, 9):
        j, i = divmod(n, 3)
        x = center_x - gap + j * gap + random.uniform(-0.5, 0.5)
        y = center_y - gap + i * gap + random.uniform(-0.5, 0.5)
        cylinder_position = [x, y, cylinder_height * 0.5]
        cylinder_position_list.append(cylinder_position)
    
    first_pos = cylinder_position_list[0]
    
    camera_h = random.uniform(3, 15)
    camera_z = first_pos[2] + camera_h
    camera_x = first_pos[0] + random.uniform(-camera_h, camera_h)
    camera_y = first_pos[1] + 0.8 * random.uniform(-camera_h, camera_h)

    focal_pos = [camera_x, camera_y, 0]
    camera_pos = [camera_x, camera_y, camera_z]
    
    view_up_vec = [0, 1, 0]
    light_pos = [0, 0, 100]
    image_size = (640, 480)
    view_angle_deg = 90.0  # 垂直视锥角度为30度

    backbone, model, device = create_model(config)
    
    camera_pos = np.array(camera_pos)
    focal_pos = np.array(focal_pos)
    
    for idx in range(num_steps):
        img_path = os.path.join(image_dir, f"{idx}.png")

        camera_front = compute_direction(start=camera_pos, end=focal_pos)
        cylinder_dir = compute_direction(start=camera_pos, end=first_pos)

        distance = np.linalg.norm(camera_pos - np.array(first_pos)).item()

        data = {
            "cylinder_pos": first_pos,
            "focal_pos": focal_pos.tolist(),
            "camera_pos": camera_pos.tolist(),
            "camera_front": camera_front.tolist(),
            "cylinder_dir": cylinder_dir.tolist(),
            "distance": distance
        }

        json_path = os.path.join(state_dir, f"{idx}.json")
        save_json(data, json_path)

        # 调用函数来完成任务
        take_photo(cylinder_position_list=cylinder_position_list, 
                   cylinder_radius=cylinder_radius,
                   cylinder_height=cylinder_height, 
                   camera_position=camera_pos, 
                   focal_point=focal_pos, 
                   view_up=view_up_vec, 
                   view_angle=view_angle_deg, 
                   light_position=light_pos, 
                   image_size=image_size, 
                   plane_size=plane_size, 
                   plane_texture_path=plane_texture_path,
                   filename=img_path,
                   debug=False)
        
        camera_pos, focal_pos, out_direction = step_one(backbone, model, device, img_path, camera_pos, 
                                                       focal_pos, camera_front, step_distance=0.05)
        
        # print(np.linalg.norm(cylinder_dir - out_direction, ord=1))


def main(args):
    config_path = "config.json"
    config = read_json(config_path)
    image_dir = f"inference_{args.id}"
    state_dir = "inference_states"
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    num_steps = 300
    run(config, image_dir, state_dir, num_steps)
    print(args.id)
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--id", type=int)
    main(args=parser.parse_args())
