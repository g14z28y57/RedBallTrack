import pyvista as pv
import numpy as np
import random
import os
import math
from util import compute_direction, save_json

ORIGIN = np.array([0, 0, 0], dtype=float)


# 在3D场景中放置一个红色小球和相机，并拍摄一张图像。
def take_photo(
        sphere_position,  # sphere_position (list or tuple): 小球的中心坐标，如 [x, y, z]。
        sphere_radius,  # 小球半径
        camera_position,  # camera_position (list or tuple): 相机的位置，如 [x, y, z]。
        focal_point,  # focal_point (list or tuple): 相机看向的点，如 [x, y, z]。
        view_up,  # view_up (list or tuple): 定义相机上方的方向向量，如 [0, 0, 1] 表示Z轴向上。
        view_angle,  # view_angle (float): 相机的垂直视锥角度，单位为度。
        light_position,  # 光源位置
        image_size,  # 图像尺寸
        plane_size,  # 地面宽度
        filename  # 输出图像的文件名
    ):

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
    for i in range(5):
        for j in range(5):
            x = -9 + 3 * i + random.uniform(-0.3, 0.3)
            y = -9 + 3 * j + random.uniform(-0.3, 0.3)
            sphere = pv.Sphere(radius=sphere_radius, center=[x, y, 0], phi_resolution=360, theta_resolution=360)
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

    sphere_pos = [
        random.uniform(-10, 10),
        random.uniform(-10, 10),
        sphere_radius
    ]  # 小球在原点

    camera_pos = [
        random.uniform(6, 10),
        random.uniform(-1, 1),
        sphere_radius * 2 + random.uniform(10, 12)
    ]  # 相机位置

    focal_pt = [
        random.uniform(-2, 2),
        random.uniform(-2, 2),
        0
    ]  # 相机看向的位置

    view_up_vec = [0, 0, 1]  # Z轴向上
    light_pos = [0, 0, 100]
    image_size = (640, 480)
    view_angle_deg = 60.0  # 垂直视锥角度为30度

    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, f"{idx}.png")

    camera_front = compute_direction(start=camera_pos, end=focal_pt).tolist()
    sphere_dir = compute_direction(start=camera_front, end=sphere_pos).tolist()

    distance = np.linalg.norm(np.array(camera_pos) - np.array(sphere_pos)).item()

    data = {
        "sphere_pos": sphere_pos,
        "focal_pos": focal_pt,
        "camera_pos": camera_pos,
        "camera_front": camera_front,
        "sphere_dir": sphere_dir,
        "distance": distance
    }

    os.makedirs(state_dir, exist_ok=True)
    json_path = os.path.join(state_dir, f"{idx}.json")
    save_json(data, json_path)

    # 调用函数来完成任务
    take_photo(sphere_position=sphere_pos,
               sphere_radius=sphere_radius,
               camera_position=camera_pos,
               focal_point=focal_pt,
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
