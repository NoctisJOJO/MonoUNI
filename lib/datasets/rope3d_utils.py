import numpy as np
import cv2
import pdb

################  Object3D  ##################


def get_objects_from_label(label_file):
    with open(label_file, "r") as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(" ")
        self.src = line
        self.cls_type = label[0]  # 障碍物类别
        self.trucation = float(label[1])  # 障碍物阶段
        self.occlusion = float(
            label[2]
        )  # 障碍物遮挡 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])  # alpha，物体的观察角度，范围：-pi~pi，
        # （在相机坐标系下，以相机原点为中心，相机原点到物体中心的连线为半径，将物体绕相机y轴旋转至相机z轴，此时物体方向与相机x轴的夹角）
        # 二维边界框，xmin,ymin,xmax,ymax
        self.box2d = np.array(
            (float(label[4]), float(label[5]), float(label[6]), float(label[7])),
            dtype=np.float32,
        )
        self.h = float(label[8])  # 三维物体的尺寸 长宽高
        self.w = float(label[9])
        self.l = float(label[10])
        # 物体中心三维位置，x,y,z
        self.pos = np.array(
            (float(label[11]), float(label[12]), float(label[13])), dtype=np.float32
        )
        self.dis_to_cam = np.linalg.norm(self.pos)
        self.ry = float(label[14])  # 3维物体的空间方向 rotation_y，
        # 在相机坐标系下，物体的全局方向角（物体前进方向与相机坐标系x轴的夹角），范围：-pi~pi。
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_obj_level()

    def get_obj_level(self):  # 物体检测的难度级别
        height = (
            float(self.box2d[3]) - float(self.box2d[1]) + 1
        )  # ymax - ymin + 1 (+1确保高度为正值)

        if self.trucation == -1:
            self.level_str = "DontCare"
            return 0

        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = "Easy"
            return 1  # Easy
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = "Moderate"
            return 2  # Moderate
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = "Hard"
            return 3  # Hard
        else:
            self.level_str = "UnKnown"
            return 4

    def generate_corners3d(self):
        """
        1.初始化物体的8个角点(物体自身为坐标系)：根据物体的尺寸 l、h 和 w，在局部坐标系下定义物体的角点。
        2.构造旋转矩阵：根据物体的Y轴旋转角 ry，构造绕Y轴旋转的矩阵 R。
        3.应用旋转矩阵：将角点通过旋转矩阵进行旋转，使它们符合物体的实际朝向。
        4.平移到相机坐标系：将旋转后的角点平移到相机坐标系，最终得到物体的8个角点在相机坐标系中的坐标
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]  #  所以物体的位置是指 物体底部中间
        # y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array(
            [
                [np.cos(self.ry), 0, np.sin(self.ry)],
                [0, 1, 0],
                [-np.sin(self.ry), 0, np.cos(self.ry)],
            ]
        )
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T  # (3,3) * (3 * 8) = (3 * 8) -> (8 * 3)
        corners3d = corners3d + self.pos
        return corners3d

    def to_bev_box2d(self, oblique=True, voxel_size=0.1):
        """
        :param bev_shape: (2) for bev shape (h, w), => (y_max, x_max) in image
        :param voxel_size: float, 0.1m  每个体素的大小，默认为0.1米
        :param oblique: 是否使用斜视角生成边界框
        :return: box2d (4, 2)/ (4) in image coordinate
        """
        Object3d.MIN_XZ = [0, 0]
        Object3d.BEV_SHAPE = [1080, 500]
        if oblique:
            corners3d = self.generate_corners3d()
            xz_corners = corners3d[
                0:4, [0, 2]
            ]  # 取前4个角点的X和Z坐标（忽略Y轴）（XZ平面）
            box2d = np.zeros((4, 2), dtype=np.int32)
            # 计算边界框的X坐标，将实际的X坐标减去最小X值 Object3d.MIN_XZ[0]，并除以体素大小 voxel_size，再转换为离散的整型坐标
            box2d[:, 0] = ((xz_corners[:, 0] - Object3d.MIN_XZ[0]) / voxel_size).astype(
                np.int32
            ) + 150
            # 计算边界框的Z坐标，操作类似，但还需要考虑鸟瞰图的Y轴反转
            box2d[:, 1] = (
                Object3d.BEV_SHAPE[0]
                - 1
                - ((xz_corners[:, 1] - Object3d.MIN_XZ[1]) / voxel_size).astype(
                    np.int32
                )
            )
            box2d[:, 0] = np.clip(
                box2d[:, 0], 0, Object3d.BEV_SHAPE[1]
            )  # 将边界框的X和Z坐标限制在鸟瞰图的范围内，
            box2d[:, 1] = np.clip(
                box2d[:, 1], 0, Object3d.BEV_SHAPE[0]
            )  # 防止坐标超出图像边界
        else:
            # 当 oblique=False 时，不使用物体的3D角点，而是根据物体的中心位置 self.pos 和尺寸 self.l, self.w 直接生成2D边界框
            box2d = np.zeros(4, dtype=np.int32)
            # discrete_center = np.floor((self.pos / voxel_size)).astype(np.int32)
            cu = np.floor((self.pos[0] - Object3d.MIN_XZ[0]) / voxel_size).astype(
                np.int32
            )
            cv = (
                Object3d.BEV_SHAPE[0]
                - 1
                - ((self.pos[2] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            )
            half_l, half_w = int(self.l / voxel_size / 2), int(self.w / voxel_size / 2)
            box2d[0], box2d[1] = cu - half_l, cv - half_w
            box2d[2], box2d[3] = cu + half_l, cv + half_w

        return box2d

    def to_str(self):
        print_str = (
            "%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f"
            % (
                self.cls_type,
                self.trucation,
                self.occlusion,
                self.alpha,
                self.box2d,
                self.h,
                self.w,
                self.l,
                self.pos,
                self.ry,
            )
        )
        return print_str

    def to_kitti_format(self):
        kitti_str = (
            "%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f"
            % (
                self.cls_type,
                self.trucation,
                int(self.occlusion),
                self.alpha,
                self.box2d[0],
                self.box2d[1],
                self.box2d[2],
                self.box2d[3],
                self.h,
                self.w,
                self.l,
                self.pos[0],
                self.pos[1],
                self.pos[2],
                self.ry,
            )
        )
        return kitti_str


class Denorm(object):
    """这段代码的功能是实现从相机坐标系和地面坐标系之间的转换，以及进行一些数学计算来确定相机和物体中心之间的关系。
    它通过加载一个文件中的数值来初始化地面的法向量，然后根据该法向量计算相机与地面坐标系之间的转换矩阵。"""

    def __init__(self, denorm_file):
        text_file = open(denorm_file, "r")
        for line in text_file:
            parsed = line.split("\n")[0].split(
                " "
            )  # 0.008187128 -0.975265 -0.2208872 7.23388195038
            # α β γ d  ; αx + βy + γz + d = 0 ; 平面的法向量 n = (α, β, γ)
            if parsed is not None and len(parsed) > 3:  # 这里只用前三个
                de_norm = []
                de_norm.append(float(parsed[0]))  # 0.008187128
                de_norm.append(float(parsed[1]))
                de_norm.append(float(parsed[2]))
        text_file.close()
        self.de_norm = np.array(de_norm)
        self.pitch_tan = float((self.de_norm[2] * 1.0 / self.de_norm[1]))
        self.pitch = float(np.arctan(self.pitch_tan))  # 俯仰角 θ = arctan(γ / β)
        self.pitch_cos = float(np.cos(self.pitch))
        self.pitch_sin = float(np.sin(self.pitch))
        ground_z_axis = self.de_norm  # 地平面法向量，在数据集中为单位向量
        cam_xaxis = np.array([1.0, 0.0, 0.0])  # 相机坐标系中的X轴向量
        ground_x_axis = cam_xaxis - cam_xaxis.dot(ground_z_axis) * ground_z_axis
        # cam_xaxis.dot(ground_z_axis) 是 cam_xaxis 在 ground_z_axis 方向上的投影，这个值表示 cam_xaxis 在 ground_z_axis 上的分量大小。
        # cam_xaxis.dot(ground_z_axis) * ground_z_axis 是 cam_xaxis 在 ground_z_axis 方向的投影向量。该向量表示相机X轴向量在地面Z轴方向上的分量。
        # cam_xaxis - cam_xaxis.dot(ground_z_axis) * ground_z_axis 的作用是将 cam_xaxis 在 ground_z_axis 上的分量去掉，得到一个垂直于 ground_z_axis 的向量。
        # 结果：该向量仍然位于原平面内，并且与 ground_z_axis 垂直。
        ground_x_axis = ground_x_axis / np.linalg.norm(
            ground_x_axis
        )  # 归一化，变成单位向量 结果：与地面法向量ground_z垂直的地面X轴
        ground_y_axis = np.cross(
            ground_z_axis, ground_x_axis
        )  # 外积\叉乘  得到与两个向量垂直的向量
        ground_y_axis = ground_y_axis / np.linalg.norm(ground_y_axis)
        # 这个矩阵就是，在相机坐标系下，地面坐标轴的三个向量。  其结果就是，相机坐标系->地面坐标系的转换（旋转）矩阵
        self.c2g_trans = np.vstack(
            [ground_x_axis, ground_y_axis, ground_z_axis]
        )  # (3, 3)
        self.g2c_trans = np.linalg.inv(self.c2g_trans)

    def ground_center2object_center(
        self, camera_pos, h
    ):  # camera_pos是物体中心在相机坐标系下的坐标
        ground_pos = np.matmul(
            self.c2g_trans, camera_pos
        )  # 将相机坐标系转换为地面坐标系
        ground_pos[2, 0] += h / 2  # 修正地面坐标中的Z轴，考虑物体高度的一半
        camera_center = np.matmul(
            self.g2c_trans, ground_pos
        )  # 将修正后的地面坐标系转换回相机坐标系
        return camera_center

    def object_center2ground_center(self, camera_pos, h):
        ground_pos = np.matmul(self.c2g_trans, camera_pos)
        ground_pos[2, 0] -= h / 2
        camera_center = np.matmul(self.g2c_trans, ground_pos)
        return camera_center


###################  calibration  ###################


def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()
    # eg:  P2: 2183.375019 0.000000 940.590363 0 0.000000 2329.297332 567.568513 0 0.000000 0.000000 1.000000 0
    obj = (
        lines[0].strip().split(" ")[1:]
    )  # strip()去除首尾的空格，split以空格为分界，[1:]跳过第一个
    P2 = np.array(obj, dtype=np.float32)
    # obj = lines[3].strip().split(' ')[1:]
    # P3 = np.array(obj, dtype=np.float32)
    # obj = lines[4].strip().split(' ')[1:]
    # R0 = np.array(obj, dtype=np.float32)
    # obj = lines[5].strip().split(' ')[1:]
    # Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {"P2": P2.reshape(3, 4)}  # 字典
    # [[2.1833750e+03 0.0000000e+00 9.4059039e+02 0.0000000e+00]   [fx   0  cx tx]
    # [0.0000000e+00 2.3292974e+03 5.6756854e+02 0.0000000e+00]    [ 0   fy cy ty]
    # [0.0000000e+00 0.0000000e+00 1.0000000e+00 0.0000000e+00]]   [ 0   0  1  0 ]


class Calibration(object):
    def __init__(self, calib_file):
        if isinstance(calib_file, str):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib["P2"]  # 3 x 4
        # self.R0 = calib['R0']  # 3 x 3
        # self.V2C = calib['Tr_velo2cam']  # 3 x 4
        # self.C2V = self.inverse_rigid_trans(self.V2C)

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        笛卡尔坐标(Cartesian Coordinates)-->齐次坐标(Homogeneous Cordinates)
        pts 是点云存储的格式（文件后缀），还有las,pcd,xyz,obj
        :param pts: (N, 3 or 2)  点云坐标 (N,3) / (N,2)
        :return pts_hom: (N, 4 or 3) 齐次坐标 (N,4) / (N,3) 在原有基础上加一个维度，方便矩阵运算(平移和旋转)
        """
        # np.hstack 是水平拼接函数，这里将 pts 和一列全为 1 的列向量拼接在一起，
        # 使得输出 pts_hom 比输入 pts 多一列（即齐次坐标中的第四个分量为1）。
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_rect(self, pts_lidar):
        """
        校正坐标(rectified coordinates)
        雷达坐标系-->相机坐标系
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        # V2C ，LiDAR坐标系到相机坐标系的转换矩阵  RO 相机校准矩阵
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_lidar(self, pts_rect):
        """相机坐标系-->雷达坐标系"""
        pts_ref = np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_rect)))
        pts_ref = self.cart_to_hom(pts_ref)  # nx4
        return np.dot(pts_ref, np.transpose(self.C2V))

    def rect_to_img(self, pts_rect):
        """
        相机坐标系——>图像坐标系
        :param pts_rect: (N, 3) 相机坐标系下的三维点
        :return pts_img: (N, 2) 图像坐标系下的二维点
        """
        # 将三维点转换为齐次坐标 eg (x,y,z)-->(x,y,z,1) 方便矩阵运算(投影变换)
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N,4)

        # 使用相机内参矩阵 self.P2.T 进行投影 eg (x,y,z,1)-->(xf+zCx, yf+ZCy,z)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)  # (N,3)

        # 这一步是透视投影，除以深度 pts_rect_hom[:, 2] 将二位投影坐标 标准化（去除深度导致的近大远小）
        # eg (xf+zCx, yf+ZCy,z)-->(xf/z+Cx,yf/z+Cy,1)==(u,v,1)
        if pts_rect_hom[:, 2].all() != 0:
            pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        else:
            pts_img = pts_2d_hom[:, 0:2]

        # 如果相机不在原点（有偏移），则得到的深度需要减去偏移量，得到真正的深度
        pts_rect_depth = (
            pts_2d_hom[:, 2] - self.P2.T[3, 2]
        )  # depth in rect camera coord

        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        将图像平面上的点反投影到相机坐标系中的三维空间点
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1
        )
        return pts_rect

    def depthmap_to_rect(self, depth_map):
        """
        将深度图中的每个像素反投影到三维空间中，得到这些像素对应的三维坐标
        :param depth_map: (H, W), depth_map
        :return:
        """
        x_range = np.arange(0, depth_map.shape[1])  # 获取图像宽度的索引范围
        y_range = np.arange(0, depth_map.shape[0])  # 获取图像高度的索引范围

        # 生成网格索引，x_indxs 和 y_indxs 对应每个像素的x 和y 坐标
        # x_indxs和y_indxs 是两个二维数组， eg 如果图像大小是3*3
        # x_indxs=[0 1 2]  y_indxs=[0 0 0]
        #         [0 1 2]          [1 1 1]
        #         [0 1 2]          [2 2 2]
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)

        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)  # 拉平
        depth = depth_map[
            y_idxs, x_idxs
        ]  # 根据x_idxs, y_idxs从深度图中提取出每个像素的深度值，存入一维数组depth
        pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)  # 像素坐标-->相机坐标
        return pts_rect, x_idxs, y_idxs

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate(
            (corners3d, np.ones((sample_num, 8, 1))), axis=2
        )  # (N,8,3)+(N,8,1)=(N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)  # 投影到二维图像平面

        x, y = (
            img_pts[:, :, 0] / img_pts[:, :, 2],
            img_pts[:, :, 1] / img_pts[:, :, 2],
        )  # /Z 见上面的投影函数; x和y为(N,8)
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)  # 最大值和最小值
        x2, y2 = np.max(x, axis=1), np.max(
            y, axis=1
        )  # 也可以看成目标框左上角(x1,y1)和右下角(x2,y2)

        boxes = np.concatenate(
            (
                x1.reshape(-1, 1),
                y1.reshape(-1, 1),
                x2.reshape(-1, 1),
                y2.reshape(-1, 1),
            ),
            axis=1,
        )
        boxes_corner = np.concatenate(
            (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2
        )  # (N,8,2) 得到每个角点在图像平面中的坐标

        return boxes, boxes_corner

    def camera_dis_to_rect(self, u, v, d):
        """
        Can only process valid u, v, d, which means u, v can not beyond the image shape, reprojection error 0.02
        :param u: (N)
        :param v: (N)
        :param d: (N), the distance between camera and 3d points, d^2 = x^2 + y^2 + z^2
        这里的 d 代表的是相机到3D点的欧氏距离，及相机光心(相机原点)到3D点的直线距离
        在很多应用中（如激光雷达、深度相机），可能会直接测量出物体到相机的实际欧氏距离 d，而不只是 z 轴上的深度值。
        这种情况下，d 更能反映物体在三维空间中的真实位置，而不是仅仅沿着光轴的深度。
        因此，代码中先使用 d 计算 x 和 y，然后再求出 z，是为了从给定的实际距离 d 反推完整的三维坐标。
        :return:
        """
        assert self.fu == self.fv, "%.8f != %.8f" % (self.fu, self.fv)  # 确保焦距一致性
        # 这是在计算图像坐标与主点之间的距离（像素空间中的距离），加上焦距的平方。
        # 实际上，这是用于计算相机成像的“有效距离”，即从相机光心到图像平面上该像素点的距离。
        fd = np.sqrt((u - self.cu) ** 2 + (v - self.cv) ** 2 + self.fu**2)
        x = ((u - self.cu) * d) / fd + self.tx
        y = ((v - self.cv) * d) / fd + self.ty
        z = np.sqrt(d**2 - x**2 - y**2)
        pts_rect = np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1
        )
        return pts_rect

    def inverse_rigid_trans(self, Tr):
        """Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
        """
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(
            Tr[0:3, 0:3]
        )  # 旋转矩阵是正交矩阵，正交矩阵的转置矩阵就是逆矩阵
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    def alpha2ry(self, alpha, u):
        """
        Get rotation_y by alpha + theta - 180
        alpha : Observation angle of object, ranging [-pi..pi]
        x : Object center x to the camera center (x-W/2), in pixels
        rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
        """
        ry = alpha + np.arctan2(u - self.cu, self.fu)

        if ry > np.pi:
            ry -= 2 * np.pi
        if ry < -np.pi:
            ry += 2 * np.pi

        return ry

    def ry2alpha(self, ry, u):
        alpha = ry - np.arctan2(u - self.cu, self.fu)

        if alpha > np.pi:
            alpha -= 2 * np.pi
        if alpha < -np.pi:
            alpha += 2 * np.pi

        return alpha

    def flip(self, img_size):
        # print(self.P2)  # 相机内参
        self.P2[0, 2] = img_size[0] - self.P2[0, 2]
        self.cu = self.P2[0, 2]
        # print(self.P2)
        # wsize = 4
        # hsize = 2
        # p2ds = (np.concatenate([np.expand_dims(np.tile(np.expand_dims(np.linspace(0,img_size[0],wsize),0),[hsize,1]),-1),\
        #                         np.expand_dims(np.tile(np.expand_dims(np.linspace(0,img_size[1],hsize),1),[1,wsize]),-1),
        #                         np.linspace(2,78,wsize*hsize).reshape(hsize,wsize,1)],-1)).reshape(-1,3)
        # p3ds = self.img_to_rect(p2ds[:,0:1],p2ds[:,1:2],p2ds[:,2:3])
        # p3ds[:,0]*=-1
        # p2ds[:,0] = img_size[0] - p2ds[:,0]

        # #self.P2[0,3] *= -1
        # cos_matrix = np.zeros([wsize*hsize,2,7])
        # cos_matrix[:,0,0] = p3ds[:,0]
        # cos_matrix[:,0,1] = cos_matrix[:,1,2] = p3ds[:,2]
        # cos_matrix[:,1,0] = p3ds[:,1]
        # cos_matrix[:,0,3] = cos_matrix[:,1,4] = 1
        # cos_matrix[:,:,-2] = -p2ds[:,:2]
        # cos_matrix[:,:,-1] = (-p2ds[:,:2]*p3ds[:,2:3])
        # new_calib = np.linalg.svd(cos_matrix.reshape(-1,7))[-1][-1]
        # new_calib /= new_calib[-1]

        # new_calib_matrix = np.zeros([4,3]).astype(np.float32)
        # new_calib_matrix[0,0] = new_calib_matrix[1,1] = new_calib[0]
        # new_calib_matrix[2,0:2] = new_calib[1:3]
        # new_calib_matrix[3,:] = new_calib[3:6]
        # new_calib_matrix[-1,-1] = self.P2[-1,-1]
        # self.P2 = new_calib_matrix.T
        # self.cu = self.P2[0, 2]
        # self.cv = self.P2[1, 2]
        # self.fu = self.P2[0, 0]
        # self.fv = self.P2[1, 1]
        # self.tx = self.P2[0, 3] / (-self.fu)
        # self.ty = self.P2[1, 3] / (-self.fv)

    def affine_transform(self, img_size, trans):
        """该函数通过将一系列图像像素点反投影到三维空间，构建线性方程组，并通过奇异值分解求解，最终生成新的相机内参矩阵。
        它通过仿射变换改变了图像坐标系，并根据这些变化重新推导了新的相机内参，使得相机能够适应图像的几何变换。"""
        wsize = 4
        hsize = 2
        random_depth = np.linspace(2, 78, wsize * hsize).reshape(
            hsize, wsize, 1
        )  # 深度从2~78均匀分布，分成w*h份
        """np.linspace(0, img_size[0], wsize)：在图像宽度范围内均匀采样 wsize 个点，生成x方向的坐标。
        np.linspace(0, img_size[1], hsize)：在图像高度范围内均匀采样 hsize 个点，生成y方向的坐标。
        通过 np.tile 和 np.expand_dims，生成了 (hsize, wsize, 1) 维度的 x 和 y 坐标矩阵，并与 random_depth 合并，形成一个 (hsize, wsize, 3) 的矩阵，
        表示采样点的 (x, y, z) 值。
        通过 reshape(-1, 3) 将这个矩阵拉平，变成一个 (wsize*hsize, 3) 的矩阵，即8个 (x, y, z) 三维坐标点。"""
        p2ds = (
            np.concatenate(
                [
                    np.expand_dims(
                        np.tile(
                            np.expand_dims(np.linspace(0, img_size[0], wsize), 0),
                            [hsize, 1],
                        ),
                        -1,
                    ),
                    np.expand_dims(
                        np.tile(
                            np.expand_dims(np.linspace(0, img_size[1], hsize), 1),
                            [1, wsize],
                        ),
                        -1,
                    ),
                    random_depth,
                ],
                -1,
            )
        ).reshape(-1, 3)
        p3ds = self.img_to_rect(p2ds[:, 0:1], p2ds[:, 1:2], p2ds[:, 2:3])  # 2d->3d
        p2ds[:, :2] = np.dot(
            np.concatenate([p2ds[:, :2], np.ones([wsize * hsize, 1])], -1), trans.T
        )  # 仿射变换
        # cos_matrix 是一个用于表示线性约束的矩阵，它的大小是 (wsize*hsize, 2, 7)，即每个图像点有两个约束方程，每个方程有7个变量。
        # 这个矩阵的每一行表示一个点的约束方程。通过设定矩阵中的值，表达了相机内参（如焦距、偏移等）与图像坐标和三维点坐标之间的关系。
        cos_matrix = np.zeros([wsize * hsize, 2, 7])
        # cos_matrix[:, 0, 0] = p3ds[:, 0]：将 p3ds 的 x 坐标赋值给约束矩阵中第一个方程的第一个元素。
        # cos_matrix[:, 1, 0] = p3ds[:, 1]：将 p3ds 的 y 坐标赋值给约束矩阵中第二个方程的第一个元素。
        # 其他行列是构建仿射变换的约束。
        cos_matrix[:, 0, 0] = p3ds[:, 0]
        cos_matrix[:, 0, 1] = cos_matrix[:, 1, 2] = p3ds[:, 2]
        cos_matrix[:, 1, 0] = p3ds[:, 1]
        cos_matrix[:, 0, 3] = cos_matrix[:, 1, 4] = 1
        cos_matrix[:, :, -2] = -p2ds[:, :2]
        cos_matrix[:, :, -1] = -p2ds[:, :2] * p3ds[:, 2:3]
        # 使用 np.linalg.svd 对 cos_matrix 进行奇异值分解，将矩阵变换为一个低秩矩阵，从而提取出新的相机内参。
        # 通过 [-1][-1] 选取奇异值最小的解，即求解后的相机内参。
        # 然后对结果进行归一化，使得内参矩阵的最后一个值为1。
        new_calib = np.linalg.svd(cos_matrix.reshape(-1, 7))[-1][-1]
        new_calib /= new_calib[-1]

        new_calib_matrix = np.zeros([4, 3]).astype(np.float32)
        new_calib_matrix[0, 0] = new_calib_matrix[1, 1] = new_calib[0]
        new_calib_matrix[2, 0:2] = new_calib[1:3]
        new_calib_matrix[3, :] = new_calib[3:6]
        new_calib_matrix[-1, -1] = self.P2[-1, -1]
        return new_calib_matrix.T
        # return new_calib_matrix.T
        # print('{}-->{}'.format(ori_size,tar_size))
        # print(new_calib_matrix.T)
        # print(np.abs(p3ds[:,:2] - self.img_to_rect(p2ds[:,0:1],p2ds[:,1:2],p2ds[:,2:3])[:,:2]).max())
        # assert(np.abs(p3ds[:,:2] - self.img_to_rect(p2ds[:,0:1],p2ds[:,1:2],p2ds[:,2:3])[:,:2]).max()<1e-10)


###################  affine trainsform  ###################


def get_dir(src_point, rot_rad):
    """旋转后的向量"""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)  # 获得旋转角度的正弦值和余弦值
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn  # 旋转后的x
    src_result[1] = src_point[0] * sn + src_point[1] * cs  # 旋转后的y

    return src_result


def get_3rd_point(a, b):
    direct = a - b  # 从b到a的方向向量
    return b + np.array(
        [-direct[1], direct[0]], dtype=np.float32
    )  # 后面的意思是将direct逆时针旋转九十度
    # 最后得到等腰直角三角形的斜边


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):  # 1表示仿射变换的逆矩阵
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array(
            [scale, scale], dtype=np.float32
        )  # 将scale转换为NumPy数组格式，确保是长度为2的数组[sx,sy]，即缩放比例

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180  # 转换成弧度
    src_dir = get_dir([0, src_w * -0.5], rot_rad)  # 原图像的方向向量
    dst_dir = np.array([0, dst_w * -0.5], np.float32)  # 目标图像的方向向量

    src = np.zeros(
        (3, 2), dtype=np.float32
    )  # 存储源图像和目标图像的三个关键点坐标，用于得出仿射变换矩阵
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift  # 中心点
    src[1, :] = center + src_dir + scale_tmp * shift  # 旋转后的方向点
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]  # 目标图像中心点
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])  # 计算得到第三个点
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))  # 仿射变换矩阵
        trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        return trans, trans_inv
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def affine_transform(pt, t):
    """进行仿射变换"""
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def roty(t):
    """绕y轴旋转"""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def compute_box_3d(obj, calib):
    """Takes an object and a projection matrix (P) and projects the 3d
    bounding box into the image plane.
    Returns:
        corners_2d: (8,2) array in left image coord.
        corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)  # 获得旋转矩阵

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    # y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))  # 相对位置
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.pos[0]
    # 绝对位置
    corners_3d[1, :] = corners_3d[1, :] + obj.pos[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.pos[2]

    return np.transpose(corners_3d)


if __name__ == "__main__":
    from lib.datasets.rope3d import Rope3D

    cfg = {"random_flip": 0.0, "random_crop": 0.0, "scale": 0.4, "shift": 0.1}
    dataset = Rope3D("../../data", "train", cfg)

    # calib testing
    # we project center fo 3D objects to image plane
    index = 1
    calib = dataset.get_calib(index)
    objects = dataset.get_label(index)
    for object in objects:
        print("object.to_kitti_format():\n", object.to_kitti_format())
        object.pos[0] *= 1
        center_3d = object.pos + [
            0,
            -object.h / 2,
            0,
        ]  # real 3D center object是底部位置，向上半个高度即中心
        center_3d = center_3d.reshape(-1, 3)  # (N, 3)
        center_3d_projected, depth = calib.rect_to_img(center_3d)
        box2d = object.box2d
        center_2d = [(box2d[0] + box2d[2]) / 2, (box2d[1] + box2d[3]) / 2]  # 2d框的中点
        # 3d中心点，2d中心点，3d中心点的投影。需要注意的是，后二者并不相同，投影考虑到了透视、深度等，更为准确
        print(
            "3D center/2D center/projected 3D center:",
            center_3d,
            center_2d,
            center_3d_projected,
        )
        print(
            "alpha ---> ry ", object.alpha, calib.alpha2ry(object.alpha, center_2d[0])
        )
        break
