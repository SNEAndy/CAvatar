import argparse
import os
import cv2
from mtcnn import MTCNN


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_path', type=str, default='./data/custom', help='folder to store test images')
    return parser.parse_args()


def face_keypoints_detect(data_path):
    save_path = os.path.join(data_path, 'detections')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    imgs_path = [os.path.join(data_path, i) for i in sorted(os.listdir(data_path)) if i.endswith('png') or i.endswith('jpg')]
    detector = MTCNN()

    for img_path in imgs_path:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        res = detector.detect_faces(img)

        txt_path = os.path.join(save_path, os.path.basename(img_path).replace('.png', '.txt').replace('.jpg', '.txt'))
        # print(txt_path)
        f = open(txt_path, 'w')

        left_eye = str(res[0]['keypoints']['left_eye'][0]) + ' ' + str(res[0]['keypoints']['left_eye'][1]) + '\n'
        f.write(left_eye)

        right_eye = str(res[0]['keypoints']['right_eye'][0]) + ' ' + str(res[0]['keypoints']['right_eye'][1]) + '\n'
        f.write(right_eye)

        nose = str(res[0]['keypoints']['nose'][0]) + ' ' + str(res[0]['keypoints']['nose'][1]) + '\n'
        f.write(nose)

        left_mouth = str(res[0]['keypoints']['mouth_left'][0]) + ' ' + str(res[0]['keypoints']['mouth_left'][1]) + '\n'
        f.write(left_mouth)

        right_mouth = str(res[0]['keypoints']['mouth_right'][0]) + ' ' + str(
            res[0]['keypoints']['mouth_right'][1]) + '\n'
        f.write(right_mouth)

        f.close()
    del detector


if __name__ == '__main__':
    data_path = get_opt().imgs_path

    face_keypoints_detect(data_path)
