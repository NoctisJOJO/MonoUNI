import json
import os
import cv2
import numpy as np
import math
import sys

# import config
import glob as gb

# import pdb
# import yaml
import shutil
from pyquaternion import Quaternion
from tqdm import tqdm
import yaml

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# pwd = os.getcwd()
# img_dir = ".\\datasets\\testing\\image"
# label_dir = ".\\datasets\\training\\label"
# calib_dir = ".\\datasets\\testing\\calib"
# denorm_dir = ".\\datasets\\testing\\denorm_1"
# save_dir = ".\\datasets\\testing\\visualize_original"
img_dir = os.path.join("/root/MonoUNI/dataset/Rope3d/image_2")
Gtlabel_dir = os.path.join(
    "/root/MonoUNI/dataset/Rope3d/label_2_4cls_filter_with_roi_for_eval"
)
Detlabel_dir = os.path.join("/root/MonoUNI/output/rope3d_eval/data_filter")
calib_dir = os.path.join("/root/MonoUNI/dataset/Rope3d/calib")
extrinsic = os.path.join("/root/MonoUNI/dataset/Rope3d/extrinsics")
denorm_dir = os.path.join("/root/MonoUNI/dataset/Rope3d/denorm")
save_dir = os.path.join("/root/MonoUNI/dataset/Rope3d/finle")

color_list = {
    "car": (0, 0, 255),
    "truck": (0, 255, 255),
    "van": (255, 0, 255),
    "bus": (255, 255, 0),
    "cyclist": (0, 128, 128),
    "motorcyclist": (128, 0, 128),
    "tricyclist": (128, 128, 0),
    "pedestrian": (0, 128, 255),
    "barrow": (255, 0, 128),
}


class Data:
    def __init__(
        self,
        obj_type="unset",
        truncation=-1,
        occlusion=-1,
        obs_angle=-10,
        x1=-1,
        y1=-1,
        x2=-1,
        y2=-1,
        w=-1,
        h=-1,
        l=-1,
        X=-1000,
        Y=-1000,
        Z=-1000,
        yaw=-10,
        score=-1000,
        detect_id=-1,
        vx=0,
        vy=0,
        vz=0,
    ):
        """init object data"""
        self.obj_type = obj_type
        self.truncation = truncation
        self.occlusion = occlusion
        self.obs_angle = obs_angle
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = w
        self.h = h
        self.l = l
        self.X = X
        self.Y = Y
        self.Z = Z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.yaw = yaw
        self.score = score
        self.ignored = False
        self.valid = False
        self.detect_id = detect_id

    def __str__(self):
        """str"""
        attrs = vars(self)
        return "\n".join("%s: %s" % item for item in attrs.items())


# 从label文件中加载检测内容
def load_detect_data(filename):
    data = []
    with open(filename) as infile:
        index = 0
        for line in infile:
            # (objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
            line = line.strip()
            fields = line.split(" ")
            t_data = Data()
            # get fields from table
            t_data.obj_type = fields[
                0
            ].lower()  # object type [car, pedestrian, cyclist, ...]
            t_data.truncation = float(fields[1])  # truncation [0..1]
            t_data.occlusion = int(float(fields[2]))  # occlusion  [0,1,2]
            t_data.obs_angle = float(fields[3])  # observation angle [rad]
            t_data.x1 = int(float(fields[4]))  # left   [px]
            t_data.y1 = int(float(fields[5]))  # top    [px]
            t_data.x2 = int(float(fields[6]))  # right  [px]
            t_data.y2 = int(float(fields[7]))  # bottom [px]
            t_data.h = float(fields[8])  # height [m]
            t_data.w = float(fields[9])  # width  [m]
            t_data.l = float(fields[10])  # length [m]
            t_data.X = float(fields[11])  # X [m]
            t_data.Y = float(fields[12])  # Y [m]
            t_data.Z = float(fields[13])  # Z [m]
            t_data.yaw = float(fields[14])  # yaw angle [rad]
            if len(fields) >= 16:
                t_data.score = float(fields[15])  # detection score
            else:
                t_data.score = 1
            t_data.detect_id = index
            data.append(t_data)
            index = index + 1
    return data


# 读取v2x denorm文件内容
def load_denorm_data(denormfile):
    text_file = open(denormfile, "r")
    for line in text_file:
        parsed = line.split("\n")[0].split(" ")
        if parsed is not None and len(parsed) > 3:
            de_norm = []
            de_norm.append(float(parsed[0]))
            de_norm.append(float(parsed[1]))
            de_norm.append(float(parsed[2]))
    text_file.close()
    return np.array(de_norm)


# V2X-Seq 标签文件中的三维目标框信息(pos,dim,ry)在虚拟LiDAR坐标系下，所以需要转至相机坐标系
def transfer_Lidar2Camera(Lidar_pos, R, T, ry):
    """Args:
        Lidar_pos: 虚拟LiDAR坐标系下的物体位置
        R: 虚拟LiDAR坐标系到相机坐标系的旋转矩阵
        T: 虚拟LiDAR坐标系到相机坐标系的平移
        ry3d(在V2X-Seq中输入ry在虚拟LiDAR坐标系下): 虚拟LiDAR坐标系下，物体绕Z轴旋转到x轴的角度；相机坐标系下物体绕y轴旋转到x轴的角度

    Returns:
        camera_pos: 相机坐标系下的物体位置
        rot_camera_res: 相机坐标系下的物体角度（绕y轴旋转至x轴）
    """
    Lidar_pos = np.array(Lidar_pos).T
    R = np.array(R)
    T = np.array(T).flatten()
    ###   camera_pos = Lidar_pos.dot(R) + T  注意点积顺序，这里是 (3,1) * (3，3)，计算错误
    # camera_pos = R.dot(Lidar_pos) + T       同下，为正确计算
    camera_pos = R @ Lidar_pos + T

    theta_Lidar = np.matrix(data=[math.cos(ry), math.sin(ry), 0]).reshape(3, 1)
    theta_Camera = R @ theta_Lidar
    # 因为在相机坐标系下是绕y旋转，所以 x-z 平面, 函数输入的y是z轴的向量
    rot_camera_res = math.atan2(theta_Camera[2], theta_Camera[0])

    return camera_pos, rot_camera_res


# 相机坐标系到地面坐标系的转换 获得地面坐标系是为了获得8个角点的坐标
def compute_c2g_trans(de_norm):
    ground_z_axis = de_norm
    cam_xaxis = np.array([1.0, 0.0, 0.0])
    ground_x_axis = cam_xaxis - cam_xaxis.dot(ground_z_axis) * ground_z_axis
    ground_x_axis = ground_x_axis / np.linalg.norm(ground_x_axis)
    ground_y_axis = np.cross(ground_z_axis, ground_x_axis)
    ground_y_axis = ground_y_axis / np.linalg.norm(ground_y_axis)
    c2g_trans = np.vstack([ground_x_axis, ground_y_axis, ground_z_axis])  # (3, 3)
    return c2g_trans


# 读取相机校准文件P2
def read_kitti_cal(calfile):
    text_file = open(calfile, "r")
    for line in text_file:
        parsed = line.split("\n")[0].split(" ")
        # cls x y w h occ x y w h ign ang
        if parsed is not None and parsed[0] == "P2:":
            p2 = np.zeros([4, 4], dtype=float)
            p2[0, 0] = parsed[1]
            p2[0, 1] = parsed[2]
            p2[0, 2] = parsed[3]
            p2[0, 3] = parsed[4]
            p2[1, 0] = parsed[5]
            p2[1, 1] = parsed[6]
            p2[1, 2] = parsed[7]
            p2[1, 3] = parsed[8]
            p2[2, 0] = parsed[9]
            p2[2, 1] = parsed[10]
            p2[2, 2] = parsed[11]
            p2[2, 3] = parsed[12]
            p2[3, 3] = 1
    text_file.close()
    return p2


# 从外参文件中获得旋转矩阵和平移矩阵
# def load_transfer_data(extrifile):
#     cfg = yaml.load(open(extrifile, "r"), Loader=yaml.Loader)
#     R = cfg["transform"]["rotation"]
#     t = cfg["transform"]["translation"]
#     return R, t


# 获得像素坐标系下的8个角点
def project_3d_ground(p2, Object_center, w3d, h3d, l3d, ry3d, c2g_trans, isCenter=True):
    """
    Args:
        p2 (nparray): 相机内参矩阵 不带畸变（3*3），带畸变（3*4），试了带畸变的有问题
        Object_center: 标签文件中物体的三维坐标点(x,y,z)，可能是底部中心，可能是正中心，取决于数据集
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: 虚拟LiDAR坐标系下，物体绕Z轴旋转到x轴的角度；相机坐标系下物体绕y轴旋转到x轴的角度
        c2g_trans: camera_to_ground translation
    Return:
        verts2d: 8个角点在像素坐标系的坐标
        verts3d: 8个角点在相机坐标系的坐标
    """
    Object_center_Ground = c2g_trans.dot(
        Object_center
    )  # 将相机坐标系中的物体坐标转换为地面坐标系
    Object_center_Ground = np.array(Object_center_Ground)
    Object_center_Ground = Object_center_Ground.reshape((3, 1))

    # 将相机坐标系中的物体航向角转换为地面坐标系
    theta0 = np.matrix(data=[math.cos(ry3d), 0, -math.sin(ry3d)]).reshape(3, 1)
    theta0 = c2g_trans[:3, :3] @ theta0
    yaw_world_res = math.atan2(theta0[1], theta0[0])

    g2c_trans = np.linalg.inv(c2g_trans)
    verts2d, verts3d = get_camera_3d_8points_g2c(
        w3d,
        h3d,
        l3d,
        yaw_world_res,
        Object_center_Ground,
        g2c_trans,
        p2,
        isCenter,
    )
    verts2d = np.array(verts2d)
    verts3d = np.array(verts3d)
    return verts2d, verts3d


def get_camera_3d_8points_g2c(
    w3d, h3d, l3d, yaw_ground, center_ground, g2c_trans, p2, isCenter
):
    """
    Args:
        w3d: width of object
        h3d: height of object
        l3d: length of object
        yaw_ground: yaw angle in world coordinate
        center_ground: the center or the bottom-center of the object in world-coord
        g2c_trans: ground2camera / world2camera transformation
        p2: projection matrix of size 4x3 (camera intrinsics)
        isCenter: 标签文件中物体的三维坐标
            1: 物体的正中心,
            0: 物体的底部中心
    Returns:
        _corners_2d_: 像素坐标系下物体的8个角点
    """
    ground_r = np.matrix(
        [
            [math.cos(yaw_ground), -math.sin(yaw_ground), 0],
            [math.sin(yaw_ground), math.cos(yaw_ground), 0],
            [0, 0, 1],
        ]
    )  # 地面坐标系下物体的旋转矩阵
    w = w3d
    l = l3d
    h = h3d

    if isCenter:  # True
        corners_3d_ground = np.matrix(
            [
                [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
            ]
        )
    else:  # bottom center, ground: z axis is up
        corners_3d_ground = np.matrix(
            [
                [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                [0, 0, 0, 0, h, h, h, h],
            ]
        )

    corners_3d_ground = np.matrix(ground_r) * np.matrix(corners_3d_ground) + np.matrix(
        center_ground
    )  # [3, 8] 地面坐标系下物体的8个角点坐标
    if g2c_trans.shape[0] == 4:  # world2camera transformation
        ones = np.ones(8).reshape(1, 8).tolist()
        corners_3d_cam = g2c_trans * np.matrix(corners_3d_ground.tolist() + ones)
        corners_3d_cam = corners_3d_cam[:3, :]
    else:  # only consider the rotation 相机坐标系下物体的8个角点坐标
        corners_3d_cam = np.matrix(g2c_trans) * corners_3d_ground  # [3, 8]

    pt = p2[:3, :3] * corners_3d_cam
    corners_2d = pt / pt[2]
    corners_2d_all = corners_2d.reshape(-1)  # 像素坐标系下物体的8个角点坐标
    # if True in np.isnan(corners_2d_all):
    #     print("Invalid projection")
    #     return None, corners_2d_all

    corners_2d = corners_2d[0:2].T.tolist()  # (8,2)
    corners_3d_cam = corners_3d_cam.T  # (8,3)
    for i in range(8):
        corners_2d[i][0] = int(corners_2d[i][0])
        corners_2d[i][1] = int(corners_2d[i][1])
    return corners_2d, corners_3d_cam


# 显示2D和3D框
def show_box_with_roll(name_list, thresh=0.5):
    for name in tqdm(name_list, desc="Draw BBox"):
        name = name.strip()
        name = name.split("/")
        name = name[-1].split(".")[0]
        img_path = os.path.join(img_dir, "%s.jpg" % (name))
        groundtruth_file = os.path.join(Gtlabel_dir, f"{name}.txt")
        detect_file = os.path.join(Detlabel_dir, "%s.txt" % (name))
        gtresult = load_detect_data(groundtruth_file)
        detresult = load_detect_data(detect_file)
        denorm_file = os.path.join(denorm_dir, "%s.txt" % (name))
        de_norm = load_denorm_data(denorm_file)  # [ax+by+cz+d=0]
        c2g_trans = compute_c2g_trans(de_norm)

        calfile = os.path.join(calib_dir, "%s.txt" % (name))
        p2 = read_kitti_cal(calfile)
        # extrifile = os.path.join(extrinsic, "%s.yaml" % (name))
        # R, T = load_transfer_data(extrifile)

        img = cv2.imread(img_path)
        h, w, c = img.shape

        # 真值
        for result_index in range(len(gtresult)):
            gt = gtresult[result_index]
            # if gt.score < thresh:
            #     continue
            # if gt.obj_type not in color_list.keys():
            #     continue
            # color_type = color_list[gt.obj_type]
            cv2.rectangle(img, (gt.x1, gt.y1), (gt.x2, gt.y2), (255, 255, 255), 1)
            if gt.w <= 0.05 and gt.l <= 0.05 and gt.h <= 0.05:  # 无效的annos
                continue
            # obj_center,gt.yaw = transfer_Lidar2Camera([gt.X, gt.Y, gt.Z], R, T,gt.yaw)
            gtverts2d, gtverts3d = project_3d_ground(
                p2,
                np.array([gt.X, gt.Y, gt.Z]),
                gt.w,
                gt.h,
                gt.l,
                gt.yaw,
                c2g_trans,
                False,
            )
            # if gtverts2d is None:
            #     continue
            # verts3d = verts3d.astype(np.int32)
            # draw projection
            cv2.line(img, tuple(gtverts2d[2]), tuple(gtverts2d[1]), (0, 0, 255), 2)
            cv2.line(img, tuple(gtverts2d[1]), tuple(gtverts2d[0]), (0, 0, 255), 2)
            cv2.line(img, tuple(gtverts2d[0]), tuple(gtverts2d[3]), (0, 0, 255), 2)
            cv2.line(img, tuple(gtverts2d[2]), tuple(gtverts2d[3]), (0, 0, 255), 2)
            cv2.line(img, tuple(gtverts2d[7]), tuple(gtverts2d[4]), (0, 0, 255), 2)
            cv2.line(img, tuple(gtverts2d[4]), tuple(gtverts2d[5]), (0, 0, 255), 2)
            cv2.line(img, tuple(gtverts2d[5]), tuple(gtverts2d[6]), (0, 0, 255), 2)
            cv2.line(img, tuple(gtverts2d[6]), tuple(gtverts2d[7]), (0, 0, 255), 2)
            cv2.line(img, tuple(gtverts2d[7]), tuple(gtverts2d[3]), (0, 0, 255), 2)
            cv2.line(img, tuple(gtverts2d[1]), tuple(gtverts2d[5]), (0, 0, 255), 2)
            cv2.line(img, tuple(gtverts2d[0]), tuple(gtverts2d[4]), (0, 0, 255), 2)
            cv2.line(img, tuple(gtverts2d[2]), tuple(gtverts2d[6]), (0, 0, 255), 2)
        # 检测值
        for result_index in range(len(detresult)):
            det = detresult[result_index]
            cv2.rectangle(img, (det.x1, det.y1), (det.x2, det.y2), (0, 0, 0), 1)
            detverts2d, detverts3d = project_3d_ground(
                p2,
                np.array([det.X, det.Y, det.Z]),
                det.w,
                det.h,
                det.l,
                det.yaw,
                c2g_trans,
                False,
            )
            # if detverts2d is None:
            #     continue
            cv2.line(img, tuple(detverts2d[2]), tuple(detverts2d[1]), (0, 255, 0), 2)
            cv2.line(img, tuple(detverts2d[1]), tuple(detverts2d[0]), (0, 255, 0), 2)
            cv2.line(img, tuple(detverts2d[0]), tuple(detverts2d[3]), (0, 255, 0), 2)
            cv2.line(img, tuple(detverts2d[2]), tuple(detverts2d[3]), (0, 255, 0), 2)
            cv2.line(img, tuple(detverts2d[7]), tuple(detverts2d[4]), (0, 255, 0), 2)
            cv2.line(img, tuple(detverts2d[4]), tuple(detverts2d[5]), (0, 255, 0), 2)
            cv2.line(img, tuple(detverts2d[5]), tuple(detverts2d[6]), (0, 255, 0), 2)
            cv2.line(img, tuple(detverts2d[6]), tuple(detverts2d[7]), (0, 255, 0), 2)
            cv2.line(img, tuple(detverts2d[7]), tuple(detverts2d[3]), (0, 255, 0), 2)
            cv2.line(img, tuple(detverts2d[1]), tuple(detverts2d[5]), (0, 255, 0), 2)
            cv2.line(img, tuple(detverts2d[0]), tuple(detverts2d[4]), (0, 255, 0), 2)
            cv2.line(img, tuple(detverts2d[2]), tuple(detverts2d[6]), (0, 255, 0), 2)
        cv2.imwrite("%s/%s.jpg" % (save_dir, name), img)


if __name__ == "__main__":
    name_list = os.listdir(Detlabel_dir)
    # print(name_list)
    if os.path.exists(save_dir):
        os.system(f"rm -r {save_dir}")
    os.mkdir(save_dir)
    show_box_with_roll(name_list)
