import cv2
import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .bfm import ParametricFaceModel
from utils import util
from utils.nvdiffrast import MeshRenderer
import trimesh
from scipy.io import savemat


class AvatarFaceRecon(BaseModel):
    """
    重建的主体类
    """

    def __init__(self, opt):
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel

        # self.visual_names = ['output_vis']
        self.model_names = ['net_recon']
        self.parallel_names = self.model_names + ['renderer']

        self.net_recon = networks.define_net_recon(
            net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=opt.init_path
        )

        self.facemodel = ParametricFaceModel(
            bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            default_name=opt.bfm_model
            # is_train=self.isTrain, default_name=opt.bfm_model
        )

        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(2 * opt.center)
        )

    def set_input(self, input):
        """
        解析input，获取输入数据
        :param input: 字典类型
        :return:
        """
        self.input_img = input['imgs'].to(self.device)
        self.gt_lm = input['lms'].to(self.device) if 'lms' in input else None
        # self.atten_mask = input['msks'].to(self.device) if 'msks' in input else None
        # self.trans_m = input['M'].to(self.device) if 'M' in input else None
        # self.image_paths = input['im_paths'] if 'im_paths' in input else None

    def forward(self):
        output_coeff = self.net_recon(self.input_img)
        self.facemodel.to(self.device)

        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(output_coeff)

        #self.pred_mask, _, self.pred_face = self.renderer(
        #    self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)

        self.pred_coeffs_dict = self.facemodel.split_coeff(output_coeff)

    def optimize_parameters(self):
        self.forward()
        # self.compute_losses()

    def compute_visuals(self):
        with torch.no_grad():
            input_img_numpy = 255. * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
            output_vis = self.pred_face * self.pred_mask + (1 - self.pred_mask) * self.input_img
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()

            if self.gt_lm is not None:
                gt_lm_numpy = self.gt_lm.cpu().numpy()
                pred_lm_numpy = self.pred_lm.detach().cpu().numpy()
                output_vis_numpy = util.draw_landmarks(output_vis_numpy_raw, gt_lm_numpy, 'b')
                output_vis_numpy = util.draw_landmarks(output_vis_numpy, pred_lm_numpy, 'r')

                output_vis_numpy = np.concatenate((input_img_numpy,
                                                   output_vis_numpy_raw, output_vis_numpy), axis=-2)
            else:
                output_vis_numpy = np.concatenate((input_img_numpy,
                                                   output_vis_numpy_raw), axis=-2)

            self.output_vis = torch.tensor(
                output_vis_numpy / 255., dtype=torch.float32
            ).permute(0, 3, 1, 2).to(self.device)

    def umeyama(self, X, Y):
        """
        Estimates the Sim(3) transformation between `X` and `Y` point sets.
        Estimates c, R and t such as c * R @ X + t ~ Y.
        Parameters
        ----------
        X : numpy.array
            (m, n) shaped numpy array. m is the dimension of the points,
            n is the number of points in the point set.
        Y : numpy.array
            (m, n) shaped numpy array. Indexes should be consistent with `X`.
            That is, Y[i] must be the point corresponding to X[i].

        Returns
        -------
        c : float
            Scale factor.
        R : numpy.array
            (3, 3) shaped rotation matrix.
        t : numpy.array
            (3, 1) shaped translation vector.
        """
        mu_x = X.mean(axis=1).reshape(-1, 1)
        mu_y = Y.mean(axis=1).reshape(-1, 1)
        var_x = np.square(X - mu_x).sum(axis=0).mean()
        cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
        U, D, VH = np.linalg.svd(cov_xy)
        S = np.eye(X.shape[0])
        if np.linalg.det(U) * np.linalg.det(VH) < 0:
            S[-1, -1] = -1
        c = np.trace(np.diag(D) @ S) / var_x
        R = U @ S @ VH
        t = mu_y - c * R @ mu_x
        return c, R, t

    def transformMesh(self, vertics, p0, p1, p2):

        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p0 - p2)
        p = (a + b + c) / 2
        s = np.sqrt(p * (p - a) * (p - b) * (p - c))
        h = 2 * s / a
        X = np.array([p0, p1, p2])
        Y = np.array([-np.sqrt(c * c - h * h), h, 0,
                      np.sqrt(b * b - h * h), h, 0,
                      0, 0, 0]).reshape(3, 3)
        X = X.T
        Y = Y.T

        c_estimated, R_estimated, t_estimated = self.umeyama(X, Y)
        # print( c_estimated, R_estimated, t_estimated )

        matrix = np.array([
            [1, 0, 0],
            [0, np.cos(-np.pi / 12), np.sin(-np.pi / 12)],
            [0, -np.sin(-np.pi / 12), np.cos(-np.pi / 12)]
        ])
        new_vertices = c_estimated * R_estimated @ vertics.T + t_estimated
        new_vertices = matrix.T @ new_vertices
        return new_vertices.T

    def write_obj_with_colors(self, obj_name, vertices, triangles, colors):
        """ Save 3D face model with texture represented by colors.
        Args:
            obj_name: str
            vertices: shape = (nver, 3)
            triangles: shape = (ntri, 3)
            colors: shape = (nver, 3)
        """
        triangles = triangles.copy()

        triangles += 1  # meshlab start with 1
        triangles = triangles[:, [2, 1, 0]]
        if obj_name.split('.')[-1] != 'obj':
            obj_name = obj_name + '.obj'

        # write obj
        with open(obj_name, 'w') as f:

            # write vertices & colors
            for i in range(vertices.shape[0]):
                # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
                s = 'v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2])
                f.write(s)

            # write f: ver ind/ uv ind
            [k, ntri] = triangles.shape
            for i in range(triangles.shape[0]):
                # s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
                s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
                f.write(s)

    def save_mesh(self, name):
        """
        保存mesh模型文件
        :param name: 保存的文件路径
        :return: None
        """
        recon_shape = self.pred_vertex  # get reconstructed shape
        recon_shape[..., -1] = 10 - recon_shape[..., -1]  # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        recon_color = self.pred_color
        recon_color = recon_color.cpu().numpy()[0]
        tri = self.facemodel.face_buf.cpu().numpy()

        #pred_face = self.pred_face.cpu().numpy().squeeze().transpose(1, 2, 0) * 255.
        #pred_face = cv2.cvtColor(pred_face.astype(np.uint8), cv2.COLOR_RGB2BGR)

        #output_vis = self.pred_face * self.pred_mask + (1 - self.pred_mask) * self.input_img
        #output_vis = output_vis.detach().cpu().numpy().squeeze().transpose(1, 2, 0) * 255.
        #output_vis = cv2.cvtColor(output_vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
        #cv2.imwrite(name.replace(".obj", ".png"), output_vis)


        meshMat = dict()
        meshMat['vertices'] = recon_shape
        meshMat['faces'] = tri
        meshMat['colors'] = np.clip(255. * recon_color, 0, 255).astype(np.uint8)

        keysIndex = np.array([16644, 16888, 16467, 16264, 32244, 32939, 33375, 33654, 33838, 34022,
                              34312, 34766, 35472, 27816, 27608, 27208, 27440, 28111, 28787, 29177,
                              29382, 29549, 30288, 30454, 30662, 31056, 31716, 8161, 8177, 8187,
                              8192, 6515, 7243, 8204, 9163, 9883, 2215, 3886, 4920, 5828,
                              4801, 3640, 10455, 11353, 12383, 14066, 12653, 11492, 5522, 6025,
                              7495, 8215, 8935, 10395, 10795, 9555, 8836, 8236, 7636, 6915,
                              5909, 7384, 8223, 9064, 10537, 8829, 8229, 7629])

        keys = recon_shape[keysIndex]
        new_vertices = self.transformMesh(recon_shape, keys[19], keys[24], keys[51])

        ver_colors = np.clip(255. * recon_color, 0, 255).astype(np.uint8)
        self.write_obj_with_colors(name, np.squeeze(new_vertices), np.squeeze(tri),
                                   np.squeeze(np.clip(255. * recon_color, 0, 255).astype(np.uint8)))

        sss1 = name.replace(".obj", "_mesh.mat")
        # savemat(sss1, meshMat)

        # 保存顶点颜色
        mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri,
                               vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8))
        mesh.export(name)

    def save_coeff(self, name):
        """
        保存几何模型参数
        :param name: 保存的文件路径
        :return: None
        """
        pred_coeffs = {key: self.pred_coeffs_dict[key].cpu().numpy() for key in self.pred_coeffs_dict}
        pred_lm = self.pred_lm.cpu().numpy()
        pred_lm = np.stack([pred_lm[:, :, 0], self.input_img.shape[2] - 1 - pred_lm[:, :, 1]],
                           axis=2)  # transfer to image coordinate
        pred_coeffs['lm68'] = pred_lm
        savemat(name, pred_coeffs)
