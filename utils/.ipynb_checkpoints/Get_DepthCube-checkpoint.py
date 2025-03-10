import numpy as np
import cv2
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /root/MonoUNI/imgs
print(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)  # /root/MonoUNI
sys.path.append(ROOT_DIR)
from lib.datasets.rope3d_utils import get_objects_from_label

label_path = os.path.join(
    "/root/MonoUNI/dataset/Rope3d/label_2",
    "1632_fa2sd4a11North_420_1612431546_1612432197_1_obstacle.txt",
)
label = get_objects_from_label(label_path)
for obj in label:
    corners3d = obj.generate_corners3d()  # 得到每个物体的8个角点在相机坐标系中的坐标


# 旋转矩阵计算
def rotation_matrix(axis, angle):
    """返回旋转矩阵，旋转轴是unit vector, 旋转角度是弧度"""
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    ux, uy, uz = axis

    rotation_matrix = np.array(
        [
            [
                cos_theta + ux**2 * (1 - cos_theta),
                ux * uy * (1 - cos_theta) - uz * sin_theta,
                ux * uz * (1 - cos_theta) + uy * sin_theta,
            ],
            [
                uy * ux * (1 - cos_theta) + uz * sin_theta,
                cos_theta + uy**2 * (1 - cos_theta),
                uy * uz * (1 - cos_theta) - ux * sin_theta,
            ],
            [
                uz * ux * (1 - cos_theta) - uy * sin_theta,
                uz * uy * (1 - cos_theta) + ux * sin_theta,
                cos_theta + uz**2 * (1 - cos_theta),
            ],
        ]
    )
    return rotation_matrix


# 计算8个角点的3D坐标
def compute_3d_corners(x, y, z, length, width, height, rotation_angle):
    # 物体的8个角点位置
    corners = np.array(
        [
            [-length / 2, -width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [length / 2, width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
            [-length / 2, -width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [length / 2, width / 2, height / 2],
            [-length / 2, width / 2, height / 2],
        ]
    )

    # 旋转矩阵
    rotation = rotation_matrix(np.array([0, 0, 1]), rotation_angle)

    # 应用旋转到角点并平移
    rotated_corners = np.dot(corners, rotation.T) + np.array([x, y, z])

    return rotated_corners


# 计算平面方程的系数
def compute_plane_equation(corner1, corner2, corner3):
    # 使用3个点来计算平面方程: ax + by + cz + d = 0
    vec1 = corner2 - corner1
    vec2 = corner3 - corner1
    normal = np.cross(vec1, vec2)

    a, b, c = normal
    d = -np.dot(normal, corner1)

    return a, b, c, d


# 计算摄像机坐标系下的深度
def compute_depth(camera_matrix, pixel, plane_params):
    # 从像素坐标计算射线
    u, v = pixel
    # 反向变换像素坐标到相机坐标系
    ray = np.linalg.inv(camera_matrix).dot(np.array([u, v, 1]))  # 射线方向 (x, y, z)

    a, b, c, d = plane_params

    # 计算射线与平面的交点
    t = -(a * ray[0] + b * ray[1] + c * ray[2] + d) / (
        a * ray[0] + b * ray[1] + c * ray[2]
    )

    # 计算交点的深度
    depth = t
    return depth


# 生成深度图
def generate_depth_map(corners, camera_matrix, image_size, plane_params):
    depth_map = np.zeros(image_size)

    # 遍历图像中的每个像素
    for v in range(image_size[1]):  # H
        for u in range(image_size[0]):  # W
            depth = compute_depth(camera_matrix, (u, v), plane_params)
            depth_map[v, u] = depth

    return depth_map


# 归一化深度图
def normalize_depth_map(depth_map):
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)

    depth_map_normalized = (depth_map - min_depth) / (max_depth - min_depth)
    return depth_map_normalized


# 示例参数
x, y, z = 13.22, -3.33, -1.17  # 物体的3D位置 (x, y, z)
length, width, height = 4.19, 1.75, 1.00  # 物体的尺寸 (长, 宽, 高)
rotation_angle = 0.077  # 物体旋转角度 (弧度)

# 摄像机内参矩阵 (假设的内参矩阵，可以根据你的实际相机参数调整)
focal_length = 1000  # 焦距
cx, cy = 640, 360  # 主点
camera_matrix = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])

# 图像大小
image_size = (720, 1280)  # 假设的图像分辨率

# 计算物体的8个角点
corners = compute_3d_corners(x, y, z, length, width, height, rotation_angle)

# 计算平面方程参数
# 这里我们简单地选择前两个面来进行平面计算示范，实际情况下需要为每个面都计算平面
plane_params = compute_plane_equation(corners[0], corners[1], corners[2])

# 生成深度图
depth_map = generate_depth_map(corners, camera_matrix, image_size, plane_params)

# 归一化深度图
depth_map_normalized = normalize_depth_map(depth_map)

# 显示结果
# cv2.imshow("Depth Map", depth_map_normalized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
