IMAGE_PATH="./imgs/cyh_right.jpg"
TRAINED_MODEL="./weights/Resnet50_Final.pth"  # or "./weights/mobilenet0.25_Final.pth"
NETWORK="resnet50"                            # or "mobile0.25"
OUTPUT_PATH="output.jpg"
#IMAGE_PATH="/home/yons/project/migu_release/migu_Avatar_3DFaceRecon/avatar_inference_3.0.1/infer_results/corner/image1/image1_revi.jpg"
#TRAINED_MODEL="./weights/Resnet50_Final.pth"  # or "./weights/mobilenet0.25_Final.pth"
#NETWORK="resnet50"                            # or "mobile0.25"
#OUTPUT_PATH="/home/yons/project/migu_release/migu_Avatar_3DFaceRecon/avatar_inference_3.0.1/infer_results/corner/image1/retina_image1.jpg"

#python detect.py \
python detect_old.py \
  --image_path ${IMAGE_PATH} \
  --trained_model ${TRAINED_MODEL} \
  --network ${NETWORK} \
  --output ${OUTPUT_PATH}
