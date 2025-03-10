import glob
import math
import os
import numpy as np
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

label_dir = os.path.join("/root/MonoUNI/dataset/Rope3d/label_2")
denorm_dir = os.path.join("/root/MonoUNI/dataset/Rope3d/denorm")


def quaternion_to_rotation_matrix(quaternion):
    """
    将四元数转换为旋转矩阵
    :param quaternion: 四元数，(w, x, y, z)
    :return: 3x3旋转矩阵
    """
    rotation = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    return rotation.as_matrix()


def transform_to_world_coordinates(camera_position, camera_rotation, object_position):
    """
    将相机坐标系下的物体位置转换到世界坐标系
    :param camera_position: 相机在世界坐标系中的位置 (x, y, z)
    :param camera_rotation: 相机旋转的四元数 (w, x, y, z)
    :param object_position: 相机坐标系下的物体位置 (x, y, z)
    :return: 物体在世界坐标系中的位置 (x, y, z)
    """
    # 计算旋转矩阵
    rotation_matrix = quaternion_to_rotation_matrix(camera_rotation)

    # 将物体位置转换到世界坐标系
    object_world_position = np.dot(rotation_matrix, object_position) + camera_position
    return object_world_position


def get_ground_contact_points(objects, camera_position, camera_rotation):
    """
    计算物体接触地面的坐标（在世界坐标系中）
    :param objects: 包含物体信息的列表，每个元素为一个字典 {'type': type, 'dimensions': (h, w, l), 'position': (x, y, z)}
    :param camera_position: 相机在世界坐标系中的位置 (x, y, z)
    :param camera_rotation: 相机旋转的四元数 (w, x, y, z)
    :return: 接触地面的点列表 [(x, y, z), ...]
    """
    ground_contact_points = []

    for obj in objects:
        # 获取物体在相机坐标系下的位置和尺寸
        obj_position = np.array(
            [obj["position"][0], obj["position"][1], obj["position"][2]]
        )
        obj_dimensions = np.array(
            [obj["dimensions"][0], obj["dimensions"][1], obj["dimensions"][2]]
        )

        # 将物体的相机坐标系位置转换为世界坐标系
        world_position = transform_to_world_coordinates(
            camera_position, camera_rotation, obj_position
        )

        # 计算接触地面的坐标（减去物体高度的一半）
        ground_z = (world_position[2] - obj_dimensions[0]) / 2.0
        # ground_z = 100
        ground_contact_points.append([world_position[0], world_position[1], ground_z])
        # ground_contact_points.append(obj_position)

    return np.array(ground_contact_points)


def fit_plane_to_points(points):
    R, Z = [], []
    for j in range(len(points)):
        R.append([float(points[j][0]), float(points[j][1]), 1])  # (x,y,1)
        Z.append([float(points[j][2])])  #  z

    R = np.mat(R)

    # 这是正规方法，最小化误差的平方和 power(|| R * A - Z || )
    # ==>  A = (R.T*R)的逆 * R.T * Z
    A = np.dot(np.dot(np.linalg.inv(np.dot(R.T, R)), R.T), Z)

    # 使用伪逆计算回归系数 可以计算 Moore-Penrose 伪逆，适用于 R.T * R 奇异或接近奇异的情况。
    # A = np.linalg.pinv(R) @ Z  # 或使用 np.dot(np.linalg.pinv(R), Z)

    # 使用 lstsq 直接求解最小二乘问题，内部使用更稳定和高效的算法，如 QR 分解或奇异值分解（SVD）
    # A, residuals, rank, s = np.linalg.lstsq(R, Z, rcond=None)
    A = np.array(A, dtype="float32").flatten()

    a, b, d = A
    C = -1.0 / math.sqrt(a * a + b * b + 1)
    A, B, D = -a * C, -b * C, -d * C
    return f"{A} {B} {C} {D}"


def transform_plane_to_camera_coordinates(
    plane_coeffs, camera_position, camera_rotation, point
):
    """
    将地面平面方程从世界坐标系转换到相机坐标系
    :param plane_coeffs: 世界坐标系下的平面方程系数 [a, b, c, d]
    :param camera_position: 平移矩阵 相机在世界坐标系中的位置 (x, y, z)
    :param camera_rotation: 旋转矩阵 相机旋转的四元数 (w, x, y, z)
    :param point：这里是相机坐标系下的点(x, y, z)
    :return: 相机坐标系下的平面方程系数 [a', b', c', d']
    """
    # 计算旋转矩阵
    rotation_matrix = quaternion_to_rotation_matrix(camera_rotation)

    # 计算平面法向量的旋转
    normal_world = np.array([plane_coeffs[0], plane_coeffs[1], plane_coeffs[2]])
    normal_camera = np.dot(rotation_matrix.T, normal_world)

    point = np.array(point)
    # point_prime = rotation_matrix @ point + camera_position  # 不需要转换了
    d = -normal_camera.dot(point)
    (
        a,
        b,
        c,
    ) = normal_camera

    return a, b, c, d


def main():

    name_list = os.listdir(label_dir)
    for i in tqdm(range(len(name_list)), desc="Compute denorm"):
        name = name_list[i].split(".")[0]
        label_path = os.path.join(label_dir, f"{name}.txt")
        with open(label_path) as f:
            label_info = f.readlines()
        vehcel = {}
        vehcels = []
        points = []
        for line in label_info:
            label = line.strip().split(" ")
            # print(label)
            vehcel["type"] = label[0]
            if float(label[8]) != 0 and float(label[9]) != 0 and float(label[10]) != 0:
                # vehcel["dimensions"] = np.array(
                #     (float(label[8]), float(label[9]), float(label[10])),
                #     dtype=np.float32,
                # )
                # print(vehcel["dimensions"])
                vehcel["position"] = np.array(
                    (float(label[11]), float(label[12]), float(label[13])),
                    dtype=np.float32,
                )
                points.append([float(label[11]), float(label[12]), float(label[13])])
                vehcels.append(vehcel)
        points = np.array(points)

        ####  之前以为标签文件中的目标位置是指目标的中心，后面发现是底部中间，所以都算错了  #####
        # # 计算接触地面点
        # ground_contact_points = get_ground_contact_points(
        #     vehcels, camera_position, camera_rotation
        # )

        # # 拟合地平面
        # plane_coeffs_world = fit_plane_to_points(ground_contact_points)
        # print("地平面方程 (世界坐标系):", plane_coeffs_world)

        # # 将地平面从世界坐标系转换到相机坐标系
        # plane_coeffs_camera = transform_plane_to_camera_coordinates(
        #     plane_coeffs_world, camera_position, camera_rotation, vehcels[0]["position"]
        # )
        # print(f"地平面方程{i} (相机坐标系):", plane_coeffs_camera)
        ####################################################################################

        # 地面点即标签中的点
        if (
            len(points) < 3
        ):  # 当标签中的物体数量少于3个，无法拟合地平面，直接用上一张图像的平面
            name = name_list[i].split(".")[0]
            denorm_path = os.path.join(denorm_dir, f"{name}.txt")
            os.system(f"touch {denorm_path} && chmod 777 {denorm_path}")
            with open(denorm_path, "w") as f:
                f.write(plane_coeffs_camera)
            continue
        plane_coeffs_camera = fit_plane_to_points(points)
        # print(f"地平面方程{i}: ", plane_coeffs_camera)

        denorm_path = os.path.join(denorm_dir, f"{name}.txt")
        os.system(f"touch {denorm_path} && chmod 777 {denorm_path}")
        with open(denorm_path, "w") as f:
            f.write(plane_coeffs_camera)


if __name__ == "__main__":
    if os.path.exists(denorm_dir):
        os.system(f"rm -r {denorm_dir}")
    os.mkdir(denorm_dir)
    main()
