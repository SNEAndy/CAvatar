# Function: 3D Face Reconstruction
# Authors: 蔡永辉
from __future__ import print_function

import os
import time
import warnings

import cv2
import torch

from face_detection.detect import init_retinaface, face_keypoints_detect
from models import create_model
from opt import get_opt
# from utils.data_prepare import face_keypoints_detect
from utils.load_BFM import load_lm3d
from utils.preprocess import get_data_path, read_data
from utils.visualizer import Visualizer

warnings.filterwarnings("ignore")


def reconstruction(opt, im, lm_5, lm3d_std, model, visualizer, img_name="output"):
    tic = time.time()
    im_tensor, lm_tensor = read_data(im, lm_5, lm3d_std)
    data = {
        'imgs': im_tensor,
        'lms': lm_tensor
    }
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()  # get image results
    visualizer.display_current_results(visuals, 0, save_results=True, count=0, name=img_name, add_image=False)

    model.save_mesh(os.path.join(visualizer.img_dir, img_name + '.obj'))  # 保存重建后的Mesh模型
    # model.save_coeff(os.path.join(visualizer.img_dir, img_name + '.mat'))  # 保存重建后的参数
    print('reconstruction time: {:.4f}s\n'.format(time.time() - tic))


def inference(opt):
    # step 1: 设置代码运行的硬件：GPU型号
    device = torch.device(int(opt.gpu_ids))
    # print("****device: ", device)
    torch.cuda.set_device(device=device)

    # step 2: 创建并初始化模型
    model = create_model(opt)

    # step 3: 加载模型到GPU
    cfg, face_keypoints_net = init_retinaface(opt, device)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = Visualizer(opt)

    # step 3: 获取测试数据
    lm3d_std = load_lm3d(opt.bfm_folder)

    # while True:
    print(opt.image_path)
    im, lm_5 = face_keypoints_detect(opt, cfg, opt.image_path, face_keypoints_net, device)

    # step 4: 执行推理

    if len(lm_5) > 0:
        img_name = os.path.basename(opt.image_path).split('.')[0]
        reconstruction(opt, im, lm_5, lm3d_std, model, visualizer, img_name=img_name)
        # reconstruction(opt, im, lm_5, lm3d_std, model, visualizer, img_name=opt.obj_name)
    else:
        raise Exception("Error!!!No face detection for reconstruction.")

    # time.sleep(100)     # 测试重建结束后是否还会占用较大显存


if __name__ == '__main__':
    # opt = get_opt()
    # if not os.path.exists(opt.results_dir):
    #     os.makedirs(opt.results_dir)
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    #
    # inference(opt)

    opt = get_opt()
    file_list = os.listdir(opt.inputDir)
    for file in file_list:
        imgs_list = [img for img in os.listdir(os.path.join(opt.inputDir, file)) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img in imgs_list:
            opt.image_path = os.path.join(opt.inputDir, file, img)
            opt.results_dir = os.path.join(opt.outputDir, file, os.path.splitext(img)[0])
            if not os.path.exists(opt.results_dir):
                os.makedirs(opt.results_dir)
            os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

            inference(opt)
    
