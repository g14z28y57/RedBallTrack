import pyvista as pv


ORIGIN = [0, 0, 0]

def take_photo():

    # 创建一个空的绘图器（plotter）
    plotter = pv.Plotter(off_screen=True, window_size=(640, 480))

    # 1. 创建光源
    plotter.add_light(
        pv.Light(
            position=(0, 0, 100),
            focal_point=ORIGIN,
            color='white',
            intensity=1.0
        )
    )

    # 2. 创建红色小球并添加到场景
    cylinder = pv.Cylinder(radius=1.0, 
                            height=0.5,
                            center=[-5, 10, 0],
                            resolution=360,
                            direction=(0, 0, 1),
                            capping=True)
    plotter.add_mesh(cylinder, color='red')

    plane_texture = pv.read_texture("plane.jpg")
    plane = pv.Plane(center=ORIGIN, direction=(0, 0, 1), i_size=20, j_size=20)
    plotter.add_mesh(plane, texture=plane_texture, smooth_shading=True)

    # 3. 正确设置相机属性
    plotter.camera.position = [0, -10, 10]
    plotter.camera.focal_point = [0, 0, 0]
    plotter.camera.view_angle = 90
    plotter.camera.clipping_range = (0.01, 100)
    plotter.camera.up = [0, 0, 1]
    

    # 3. 拍摄图像
    plotter.show()
    # plotter.screenshot(filename=filename)
    plotter.close()
    

if __name__ == "__main__":
    take_photo()