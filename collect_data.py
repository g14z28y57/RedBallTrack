import pyvista as pv
import numpy as np
import random
import os
import math
from util import compute_direction, save_json

ORIGIN = [0, 0, 0]


# 在3D场景中放置一个红色小球和相机，并拍摄一张图像。
def take_photo(
        cylinder_position_list,
        cylinder_radius,
        cylinder_height,
        camera_position,  # camera_position (list or tuple): 相机的位置，如 [x, y, z]。
        focal_point,
        view_up,  # view_up (list or tuple): 定义相机上方的方向向量，如 [0, 0, 1] 表示Z轴向上。
        view_angle,  # view_angle (float): 相机的垂直视锥角度，单位为度。
        light_position,  # 光源位置
        image_size,  # 图像尺寸
        plane_size,  # 地面宽度
        plane_texture_path,  # 地板贴图
        filename,  # 输出图像的文件名
        debug=False
    ):

    assert len(cylinder_position_list) > 0

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
    for cylinder_position in cylinder_position_list:
        cylinder = pv.Cylinder(radius=cylinder_radius, 
                               height=cylinder_height,
                               center=cylinder_position,
                               resolution=360,
                               direction=(0, 0, 1),
                               capping=True)
        plotter.add_mesh(cylinder, color='red')

    plane_texture = pv.read_texture(plane_texture_path)
    plane = pv.Plane(center=ORIGIN, direction=(0, 0, 1), i_size=plane_size, j_size=plane_size)
    plotter.add_mesh(plane, texture=plane_texture, smooth_shading=True)

    # 3. 正确设置相机属性
    plotter.camera.position = camera_position
    plotter.camera.focal_point = focal_point
    plotter.camera.view_angle = view_angle
    plotter.camera.clipping_range = (0.01, 100)
    plotter.camera.up = view_up

    # 3. 拍摄图像
    plotter.screenshot(filename=filename)
    
    if debug:
        # 创建一些点来标记
        label_position_list = []
        labels = []
        for idx, cylinder_position in enumerate(cylinder_position_list):
            label_position = cylinder_position.copy()
            label_position[2] += cylinder_height
            label_position_list.append(label_position)
            labels.append(f"{idx}")
        points = np.array(cylinder_position_list)
        point_cloud = pv.PolyData(points)
        plotter.add_mesh(point_cloud, color='red', render_points_as_spheres=True, point_size=1)
        # 在这些点上添加标签
        plotter.add_point_labels(points, labels, font_size=20)
        plotter.screenshot(filename=filename.split(".")[0] + "_debug.png")

    plotter.close()


def is_cylinder_in_frustum(
    cylinder_position,  # 小球中心坐标 [x, y, z]
    cylinder_radius,    # 小球半径
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
    cylinder_position = np.array(cylinder_position, dtype=float)
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
    cylinder_to_camera = cylinder_position - camera_position
    distance_to_cylinder = np.linalg.norm(cylinder_to_camera)

    # 如果小球在相机后面，直接返回 False
    projection = np.dot(cylinder_to_camera, camera_direction)
    if projection <= 0:
        return False

    # 将小球中心向量投影到相机坐标系
    cylinder_to_camera_normalized = cylinder_to_camera / distance_to_cylinder

    # 计算小球中心相对于相机方向的夹角
    cos_theta = np.dot(cylinder_to_camera_normalized, camera_direction)
    if cos_theta <= 0:
        return False
    theta = np.arccos(cos_theta)

    # 计算小球的角半径（考虑小球半径）
    angular_radius = np.arcsin(min(cylinder_radius / distance_to_cylinder, 1.0))

    # 检查垂直方向是否在视锥内
    vertical_half_fov = vertical_fov / 2
    if (theta - angular_radius) > vertical_half_fov:
        return False

    # 检查水平方向
    # 投影到右向量和上向量平面，计算水平角度
    proj_right = np.dot(cylinder_to_camera_normalized, right_vector)
    proj_up = np.dot(cylinder_to_camera_normalized, up_vector)
    theta_horizontal = np.arctan2(proj_right, cos_theta)
    horizontal_half_fov = horizontal_fov / 2
    if abs(theta_horizontal) > (horizontal_half_fov + angular_radius):
        return False

    # 如果通过所有检查，小球在视锥内
    return True


def take_photo_wrapper(idx, image_dir, state_dir):
    # --- 使用示例 ---
    # 设置参数
    plane_size = 20
    cylinder_radius = 1.0
    cylinder_height = 0.5
    
    center_x = 4 * random.uniform(-1, 1)
    center_y = 4 * random.uniform(-1, 1)
    start_idx = random.randint(0, 8)
    
    gap = 3.5
    cylinder_position_list = []
    for n in range(start_idx, 9):
        j, i = divmod(n, 3)
        x = center_x - gap + j * gap + 0.5 * random.uniform(-1, 1)
        y = center_y - gap + i * gap + 0.5 * random.uniform(-1, 1)
        cylinder_position = [x, y, cylinder_height * 0.5]
        cylinder_position_list.append(cylinder_position)
    
    first_pos = cylinder_position_list[0]
    
    camera_h = random.uniform(1, 15)
    camera_z = first_pos[2] + camera_h
    camera_x = first_pos[0] + camera_h * random.uniform(-1, 1)
    camera_y = first_pos[1] + 0.8 * camera_h * random.uniform(-1, 1)

    delta = 0.2 * camera_h * random.uniform(-1, 1)
    theta =  math.radians(90 + 10 * random.uniform(-1, 1))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    focal_pos = [camera_x - delta * sin_theta, camera_y + delta * cos_theta, 0]
    camera_pos = [camera_x, camera_y, camera_z]
    
    view_up_vec = [cos_theta, sin_theta, 0]
    light_pos = [0, 0, 100]
    image_size = (640, 480)
    view_angle_deg = 90.0  # 垂直视锥角度为30度

    if not is_cylinder_in_frustum(
        cylinder_position=first_pos,  # 小球中心坐标 [x, y, z]
        cylinder_radius=cylinder_radius,  # 小球半径
        camera_position=camera_pos,  # 相机位置 [x, y, z]
        focal_point=focal_pos,  # 相机焦点 [x, y, z]
        view_up=view_up_vec,  # 相机上方向向量 [x, y, z]
        view_angle=view_angle_deg,  # 垂直视锥角度（度）
        image_size=image_size
    ):
        return False
    
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, f"{idx}.png")

    camera_front = [0, 0, -1]
    cylinder_dir = compute_direction(start=camera_pos, end=first_pos).tolist()
    
    # print(cylinder_dir)
    
    distance = np.linalg.norm(np.array(camera_pos) - np.array(first_pos)).item()

    data = {
        "cylinder_pos": first_pos,
        "focal_pos": focal_pos,
        "camera_pos": camera_pos,
        "camera_front": camera_front,
        "cylinder_dir": cylinder_dir,
        "distance": distance
    }

    os.makedirs(state_dir, exist_ok=True)
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
               plane_texture_path="plane.jpg",
               filename=image_path)
    
    # print(f"图像已保存到 {image_path}")

    return True


def main(args):
    idx = args.start_index
    end_idx = idx + args.num
    while idx < end_idx:
        if idx % 100 == 0:
            print(idx)
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
