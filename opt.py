import argparse


def get_opt():
    parser = argparse.ArgumentParser()

    # test data parameters
    parser.add_argument('--inputDir', default='./3', help='Input image path for inference')
    parser.add_argument('--outputDir', default='./infer_results', help='output dir for inference')
    parser.add_argument('--image_path', default='data/custom/cyh.jpg', help='Input image path for inference')
    #parser.add_argument('--img_folder', type=str, default='./data/custom', help='folder to store test images')
    parser.add_argument('--results_dir', type=str, default="./infer_results", help='folder to save inference results')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0')
    parser.add_argument('--obj_name', type=str, default=None, help='name for obj file')
    # parser.add_argument('--obj_name', type=str, default='cyh', help='name for obj file')

    # face detection parameters
    # parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
    parser.add_argument('--trained_model', default='./face_detection/weights/Resnet50_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--face_detection_output', default='./infer_results/output.jpg', help='Path for output')
    parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')

    # model parameters
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--model_name', type=str, default='Migu_FaceReconModel.pth')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--use_ddp', type=bool, default=False, help='whether use distributed data parallel')

    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'],
                        help='network structure')
    parser.add_argument('--init_path', type=str, default='checkpoints/resnet50-0676ba61.pth')
    parser.add_argument('--use_last_fc', type=bool, default=False)
    parser.add_argument('--bfm_folder', type=str, default='BFM')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)
    # sadtalker parameters
    parser.add_argument("--driven_audio", default='./infer_result/audio/1/index.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./infer_results/picture/1/index.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./infer', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0, help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2, help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256, help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1., help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer', type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer', type=str, default=None, help="background enhancer, [realesrgan]")

    parser.add_argument("--face3dvis", default=False, help="generate 3d face and 3d landmarks")
    parser.add_argument("--still", action="store_true",
                        help="can crop back to the original videos for the full body aniamtion")
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'],
                        help="how to preprocess the images")

    parser.add_argument("--old_version", action="store_true", help="use the pth other than safetensor version")
    #==============================================
    parser.set_defaults(focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.)

    return parser.parse_args()
