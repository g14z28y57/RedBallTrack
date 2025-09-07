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
    sphere = pv.Sphere(radius=sphere_radius, center=sphere_position, phi_resolution=360, theta_resolution=360)
    plotter.add_mesh(sphere, color='red')

    plane = pv.Plane(center=[0, 0, -sphere_radius] , direction=(0, 0, 1), i_size=plane_size, j_size=plane_size)
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


def is_sphere_in_frustum(
    sphere_position,  # 小球中心坐标 [x, y, z]
    sphere_radius,    # 小球半径
    camera_position,  # 相机位置 [x, y, z]
    focal_point,      # 相机焦点 [x, y, z]
    view_up,          # 相机上方向向量 [x, y, z]
    view_angle,       # 垂直视场角（度）
    image_size        # 图像宽高比，默认为1200/800=1.5
):
    """
    判断小球是否在相机的视锥内。
    返回 True 如果小球（部分或全部）在视锥内，否则返回 False。
    """
    aspect_ratio = image_size[0] / image_size[1]
    # 将输入转换为 numpy 数组
    sphere_position = np.array(sphere_position, dtype=float)
    camera_position = np.array(camera_position, dtype=float)
    focal_point = np.array(focal_point, dtype=float)
    view_up = np.array(view_up, dtype=float)

    # 计算相机的朝向向量（从相机指向焦点）
    camera_direction = focal_point - camera_position
    camera_direction = camera_direction / np.linalg.norm(camera_direction)

    # 计算相机的右向量（通过叉乘）
    right_vector = np.cross(camera_direction, view_up)
    right_vector = right_vector / np.linalg.norm(right_vector)

    # 重新计算上向量，确保正交
    up_vector = np.cross(right_vector, camera_direction)
    up_vector = up_vector / np.linalg.norm(up_vector)

    # 将垂直视场角转换为弧度
    vertical_fov = np.deg2rad(view_angle)
    # 根据宽高比计算水平视场角
    horizontal_fov = vertical_fov * aspect_ratio

    # 计算小球中心到相机的向量
    sphere_to_camera = sphere_position - camera_position
    distance_to_sphere = np.linalg.norm(sphere_to_camera)

    # 如果小球在相机后面，直接返回 False
    projection = np.dot(sphere_to_camera, camera_direction)
    if projection <= 0:
        return False

    # 将小球中心向量投影到相机坐标系
    sphere_to_camera_normalized = sphere_to_camera / distance_to_sphere

    # 计算小球中心相对于相机方向的夹角
    cos_theta = np.dot(sphere_to_camera_normalized, camera_direction)
    if cos_theta <= 0:
        return False
    theta = np.arccos(cos_theta)

    # 计算小球的角半径（考虑小球半径）
    angular_radius = np.arcsin(min(sphere_radius / distance_to_sphere, 1.0))

    # 检查垂直方向是否在视锥内
    vertical_half_fov = vertical_fov / 2
    if (theta - angular_radius) > vertical_half_fov:
        return False

    # 检查水平方向
    # 投影到右向量和上向量平面，计算水平角度
    proj_right = np.dot(sphere_to_camera_normalized, right_vector)
    proj_up = np.dot(sphere_to_camera_normalized, up_vector)
    theta_horizontal = np.arctan2(proj_right, cos_theta)
    horizontal_half_fov = horizontal_fov / 2
    if abs(theta_horizontal) > (horizontal_half_fov + angular_radius):
        return False

    # 如果通过所有检查，小球在视锥内
    return True


def take_photo_wrapper(idx):
    # --- 使用示例 ---
    # 设置参数
    plane_size = 20
    sphere_pos = [random.uniform(-10, 10), random.uniform(-10, 10), 0] # 小球在原点
    sphere_radius = 1
    camera_degree = random.uniform(0, 360)
    camera_dist = random.uniform(10, 20)
    camera_x = camera_dist * math.cos(camera_degree)
    camera_y = camera_dist * math.sin(camera_degree)
    camera_h = random.uniform(5, 12)
    camera_pos = [camera_x, camera_y, camera_h]  # 相机位置
    focal_height = random.uniform(-4, 4)
    focal_pt = [0, 0, focal_height]  # 相机看向小球中心
    view_up_vec = [0, 0, 1]  # Z轴向上
    light_pos = [0, 0, 100]
    image_size = (640, 480)
    view_angle_deg = 60.0  # 垂直视锥角度为30度

    if not is_sphere_in_frustum(
        sphere_position=sphere_pos,  # 小球中心坐标 [x, y, z]
        sphere_radius=sphere_radius,  # 小球半径
        camera_position=camera_pos,  # 相机位置 [x, y, z]
        focal_point=focal_pt,  # 相机焦点 [x, y, z]
        view_up=view_up_vec,  # 相机上方向向量 [x, y, z]
        view_angle=view_angle_deg,  # 垂直视锥角度（度）
        image_size=image_size
    ):
        return False

    image_dir = "image"
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

    state_dir = "state"
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
        if take_photo_wrapper(idx=idx):
            idx += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--start_index", type=int)
    parser.add_argument("--num", type=int)
    main(args=parser.parse_args())
