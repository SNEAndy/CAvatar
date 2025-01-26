#!/bin/sh

# 输入的是单张图片路径
#IMAGE_PATH="./data/custom/clj_1206_2.jpg"
#OBJ_NAME='clj_1206_2'                                          # 保存的obj文件名字前缀
#RESULTS_DIR="./infer_results"                           # folder to save the inference results
#FACE_DETECTION_OUTPUT=$RESULTS_DIR/$OBJ_NAME".jpg"      # output for face detection
#GPU_IDS="0"

IMAGE_PATH="./data/corner_cases/corner_case_1.jpeg"
OBJ_NAME='corner_cases_1'                                          # 保存的obj文件名字前缀
RESULTS_DIR="./infer_results/$OBJ_NAME"                           # folder to save the inference results
FACE_DETECTION_OUTPUT=$RESULTS_DIR/$OBJ_NAME".jpg"      # output for face detection
GPU_IDS="0"

python main.py \
    --image_path=$IMAGE_PATH \
    --results_dir=$RESULTS_DIR \
    --gpu_ids=$GPU_IDS \
    --face_detection_output=$FACE_DETECTION_OUTPUT \
    --obj_name=$OBJ_NAME

## Option 2.输入的是文件夹路径
#images_folder='/home/yons/project/migu_release/migu_Avatar_3DFaceRecon/avatar_inference_3.0.0/data/test'
#images_name=`ls ${images_folder}/`
## shellcheck disable=SC2068
## shellcheck disable=SC2039
## shellcheck disable=SC2034
#for img_name in ${images_name[@]};do
#  IMAGE_PATH=$images_folder/$img_name
#  RESULTS_DIR="./infer_results/test"
#  FACE_DETECTION_OUTPUT=$RESULTS_DIR/$img_name
#  GPU_IDS="0"
#  python inference.py \
#    --image_path=$IMAGE_PATH \
#    --results_dir=$RESULTS_DIR \
#    --face_detection_output=$FACE_DETECTION_OUTPUT \
#    --gpu_ids=$GPU_IDS
#done
