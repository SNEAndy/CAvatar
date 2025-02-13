from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from face_detection.data import cfg_mnet, cfg_re50
from face_detection.layers.functions.prior_box import PriorBox
from face_detection.utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from face_detection.models.retinaface import RetinaFace
from face_detection.utils.box_utils import decode, decode_landm
import time


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    """ Old style model is stored with all names of parameters sharing common prefix 'module.' """
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    # print('Loading face detection model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def init_retinaface(opt, device):
    torch.set_grad_enabled(False)
    cfg = None
    if opt.network == "mobile0.25":
        cfg = cfg_mnet
    elif opt.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, opt.trained_model, opt.cpu)
    net.eval()
    # print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    net = net.to(device)
    return cfg, net


def face_keypoints_detect(args, cfg, im_path, net, device):
    resize = 1

    # testing begin
    tic = time.time()
    # image_path = "./curve/test.jpg"
    # image_path = args.image_path
    image_path = im_path
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 高分辨率的图片做人脸检测，误检率较高，因此需要根据分辨率大小做下采样操作
    img_copy = img_raw.copy()
    img_height, img_width, _ = img_copy.shape
    print('The shape of image: {}x{}'.format(img_width, img_height))
    if 2000 < max(img_height, img_width) < 3000:
        resize_value = 2
    elif 3000 < max(img_height, img_width) < 4000:
        resize_value = 3
    else:
        resize_value = 4

    if resize_value > 1:
        img_copy = cv2.resize(img_copy, (int(img_width / resize_value), int(img_height / resize_value)))

    img = np.float32(img_copy)
    # img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    # net forward
    loc, conf, landms = net(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    # show image
    lm_5 = []
    if args.save_image:
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            b = list(map(lambda item: item * resize_value, b))
            # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

            cx = b[0]
            cy = b[1] + 12
            # cv2.putText(img_raw, text, (cx, cy),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)        # left eye
            # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)      # right eye
            # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)     # nose
            # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)      # left mouth
            # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)      # right mouth
            lm_5 = np.array([
                    [b[5], b[6]],
                    [b[7], b[8]],
                    [b[9], b[10]],
                    [b[11], b[12]],
                    [b[13], b[14]]
                ]).astype(np.float32)

        # save image

        # name = "test.jpg"
        name = args.face_detection_output
        # cv2.imwrite(name, img_raw)

    print('face detection time: {:.4f}s'.format(time.time() - tic))

    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    return img_rgb, lm_5
