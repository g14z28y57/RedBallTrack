import pyvista as pv
import numpy as np
import random
import os
import math
from util import compute_direction, save_json

ORIGIN = np.array([0, 0, 0], dtype=float)


# 在3D场景中放置一个红色小球和相机，并拍摄一张图像。
def take_photo(
        sphere_position_list,  # sphere_position (list or tuple): 小球的中心坐标，如 [x, y, z]。
        sphere_radius,  # 小球半径
        camera_position,  # camera_position (list or tuple): 相机的位置，如 [x, y, z]。
        focal_point,
        view_up,  # view_up (list or tuple): 定义相机上方的方向向量，如 [0, 0, 1] 表示Z轴向上。
        view_angle,  # view_angle (float): 相机的垂直视锥角度，单位为度。
        light_position,  # 光源位置
        image_size,  # 图像尺寸
        plane_size,  # 地面宽度
        filename  # 输出图像的文件名
    ):

    assert len(sphere_position_list) > 0

    # 创建一个空的绘图器（plotter）
    plotter = pv.Plotter(off_screen=True, window_size=image_size)

    # 1. 创建光源
    plotter.add_light(
        pv.Light(
            position=light_position,
            focal_point=ORIGIN,
            color='white',
            intensity=1.0
        )
    )

    # 2. 创建红色小球并添加到场景
    for sphere_position in sphere_position_list:
        sphere = pv.Sphere(radius=sphere_radius, center=sphere_position, phi_resolution=360, theta_resolution=360)
        plotter.add_mesh(sphere, color='red')

    
    plane = pv.Plane(center=ORIGIN, direction=(0, 0, 1), i_size=plane_size, j_size=plane_size)
    plotter.add_mesh(plane, color='grey')

    # 3. 正确设置相机属性
    plotter.camera.position = camera_position
    plotter.camera.focal_point = focal_point
    plotter.camera.up = view_up
    plotter.camera.view_angle = view_angle
    # 3. 拍摄图像
    plotter.screenshot(filename=filename)

    print(f"图像已保存到 {filename}")
    plotter.close()


def take_photo_wrapper(idx, image_dir, state_dir):
    # --- 使用示例 ---
    # 设置参数
    plane_size = 20
    sphere_radius = 1

    start_idx = random.randint(0, 8)
    sphere_position_list = []
    for n in range(start_idx, 9):
        i, j = divmod(n, 3)
        x = -3 + i * 3 + random.uniform(-0.1, 0.1)
        y = -3 + j * 3 + random.uniform(-0.1, 0.1)
        sphere_position = [x, y, sphere_radius]
        sphere_position_list.append(sphere_position)
    
    first_sphere_pos = sphere_position_list[0]
    camera_pos = first_sphere_pos.copy()
    camera_pos[0] = camera_pos[0] + random.uniform(-4, -2)
    camera_pos[1] = camera_pos[1] + random.uniform(-4, -2)
    camera_pos[2] = 2 * sphere_radius + random.uniform(6, 8)

    focal_pos = first_sphere_pos.copy()
    focal_pos[0] = focal_pos[0] + random.uniform(-1, 1)
    focal_pos[1] = focal_pos[1] + random.uniform(-1, 1)
    focal_pos[2] = 0

    view_up_vec = [0, 0, 1]  # Z轴向上
    light_pos = [0, 0, 100]
    image_size = (640, 480)
    view_angle_deg = 60.0  # 垂直视锥角度为30度

    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, f"{idx}.png")

    camera_front = compute_direction(start=camera_pos, end=focal_pos).tolist()
    sphere_dir = compute_direction(start=camera_front, end=first_sphere_pos).tolist()

    distance = np.linalg.norm(np.array(camera_pos) - np.array(first_sphere_pos)).item()

    data = {
        "sphere_pos": sphere_position_list[0],
        "focal_pos": focal_pos,
        "camera_pos": camera_pos,
        "camera_front": camera_front,
        "sphere_dir": sphere_dir,
        "distance": distance
    }

    os.makedirs(state_dir, exist_ok=True)
    json_path = os.path.join(state_dir, f"{idx}.json")
    save_json(data, json_path)

    # 调用函数来完成任务
    take_photo(sphere_position_list=sphere_position_list,
               sphere_radius=sphere_radius,
               camera_position=camera_pos,
               focal_point=focal_pos,
               view_up=view_up_vec,
               view_angle=view_angle_deg,
               light_position=light_pos,
               image_size=image_size,
               plane_size=plane_size,
               filename=image_path)

    return True


def main(args):
    idx = args.start_index
    end_idx = idx + args.num
    while idx < end_idx:
        if take_photo_wrapper(idx=idx, image_dir=args.image_dir, state_dir=args.state_dir):
            idx += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--start_index", type=int)
    parser.add_argument("--num", type=int)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--state_dir", type=str)
    main(args=parser.parse_args())
