#!/usr/bin/env python3
import datetime
import random
import shutil
import os, time
import sys
from opt import get_opt
from inference import inference
from argparse import ArgumentParser
import numpy as np
sys.path.append("F:/projects/project/SadTalker")
import torchaudio
from fireredtts.fireredtts import FireRedTTS
from SadTalker.src.utils.preprocess import CropAndExtract
from SadTalker.src.test_audio2coeff import Audio2Coeff
from SadTalker.src.facerender.animate import AnimateFromCoeff
from SadTalker.src.generate_batch import get_data
from SadTalker.src.generate_facerender_batch import get_facerender_data
from SadTalker.src.utils.init_path import init_path
def create_log_file(filePath,time1,time2):
    logPath = os.path.join(filePath,'output.log')
    with open(logPath, 'w') as f:
        f.write(f"{time1},{time2}")
        f.close()
def TryWriteFileTxt(path,content):
    try:
        with open(path, "w") as f:
            f.write(content)
            f.close()
            return 0
    except:
        return 1
def create_directory(dirPath):
    try:
        os.makedirs(dirPath)
        print(f"{dirPath} created.")
    except FileExistsError:
        print(f"{dirPath} already exists.")

###################################################
# voice clone model
def voice_clone_load(model_path, device: str):
    tts = FireRedTTS(
        config_path="configs/config_24k.json",
        pretrained_path=model_path,
        device=device
    )
    return tts


def voice_clone_main(net, prompt_text, prompt_audio, lang):
    assert lang in ['zh', 'en']
    with open(prompt_text, 'r', encoding='utf-8') as fin:
        text = fin.read()
        fin.close()

    rec_wavs = net.synthesize(
        prompt_wav=prompt_audio,  # "examples/prompt_1.wav",
        text=text,
        lang=lang,
    )
    return rec_wavs

if __name__ == '__main__':
    # # 输入文件夹
    # inputDir = sys.argv[1]
    # #输出文件夹
    # outputDir = sys.argv[2]
    opt = get_opt()
    inputDir = opt.inputDir
    outputDir = opt.outputDir
    print("input path: ",inputDir)
    device = 'cuda'
    # load vc model
    try:
        t_load_1 = time.time()
        tts = voice_clone_load("./pretrained_models", device=device)  # user model
        t_load_2 = time.time()
        print("load vc model in {:.2f}s".format(t_load_2 - t_load_1))
    except Exception as e:
        print("load model failed!")
        raise RuntimeError(e)

    try:
        if os.path.exists(outputDir):
           shutil.rmtree(outputDir)
        create_directory(outputDir)

        picOutPaths = []
        picInputPaths=[]
        textInputPaths = []   # add in text paths
        audioInputPaths = []   # add in audio paths
        audioOutPaths = []
        videoOutPaths = []
        timestringCounts=[]
        for i in range(1, 11):
            path = os.path.join(outputDir, f'picture/{i}')
            picOutPaths.append(path)
            picInputPaths.append(os.path.join(inputDir, f'picture/{i}'))
        # print("picInputPaths: ", picInputPaths)
        # print("picOutPaths: ", picOutPaths)
        for i in range(1, 4):
            path = os.path.join(outputDir, f'audio/{i}')
            audioInputPaths.append(os.path.join(inputDir, f'audio/{i}/index.wav'))
            textInputPaths.append(os.path.join(inputDir, f'text/{i}/index.txt'))
            audioOutPaths.append(path)
        for i in range(1, 4):
            path = os.path.join(outputDir, f'video/{i}')
            videoOutPaths.append(path)

        arrayIndex = 0
        curProgressIndex = 1
        finish = False
        while (not finish):
                current_start_timestamp = int(datetime.datetime.now().timestamp())
                current_end_timestamp = current_start_timestamp + random.randint(10, 30)

                match curProgressIndex:
                    case 1:
                        #pta
                        showIndex = arrayIndex
                        curPath = picOutPaths[showIndex]
                        os.makedirs(curPath, exist_ok=True)
                        picFileNameSrc = os.path.join(picInputPaths[showIndex], f'index.png')

                        #************* 执行形象生成 *************#
                        opt.image_path = picFileNameSrc             # 每张图像的路径
                        opt.results_dir = curPath    # 对每张图像执行形象生成的输出路径（包括index.obj、output.log和index.png）
                        # print("opt.image_path: ", opt.image_path)
                        # print("opt.results_dir: ", opt.results_dir)
                        inference(opt)
                        # ************* 形象生成结束 *************#

                        # 日志记录每张图片形象生成的时间戳
                        current_end_timestamp = int(datetime.datetime.now().timestamp())
                        create_log_file(curPath, current_start_timestamp, current_end_timestamp)
                        #改名照片
                        src_file = os.path.join(curPath, f'index.png')
                        new_file = os.path.join(curPath, f'{current_end_timestamp}_1.png')
                        if os.path.isfile(src_file):
                            # Rename the file
                            os.rename(src_file, new_file)
                            print(f'Renamed "{src_file}" to "{new_file}".')
                        else:
                            print(f'The file "{src_file}" does not exist.')

                        if arrayIndex == len(picOutPaths)-1:
                            arrayIndex = -1
                            curProgressIndex += 1
                    case 2:
                        # 语音
                        curPath = audioOutPaths[arrayIndex]
                        in_aud = audioInputPaths[arrayIndex]
                        in_txt = textInputPaths[arrayIndex]
                        os.makedirs(curPath, exist_ok=True)
                        # here add net code for voice clone
                        try:
                            data_ch = voice_clone_main(tts, prompt_text=in_txt, prompt_audio=in_aud, lang='zh')
                            data_en = voice_clone_main(tts, prompt_text=in_txt, prompt_audio=in_aud, lang='en')
                            data_ch_tensor=data_ch.detach().cpu()
                            data_en_tensor=data_en.detach().cpu()
                            data_ch = data_ch.detach().cpu().numpy().tobytes()
                            data_en = data_en.detach().cpu().numpy().tobytes()

                            current_end_timestamp = int(datetime.datetime.now().timestamp())
                            timestringCounts.append(current_end_timestamp)
                            create_log_file(curPath, current_start_timestamp, current_end_timestamp)
                            data1 = os.path.join(curPath, f'{current_end_timestamp}_chinese.data')
                            data2 = os.path.join(curPath, f'{current_end_timestamp}_english.data')
                            wav_save_path1=os.path.join(curPath,f'{current_end_timestamp}_chinese.wav')
                            wav_save_path2 = os.path.join(curPath, f'{current_end_timestamp}_english.wav')
                            torchaudio.save(wav_save_path1,data_ch_tensor,24000)
                            torchaudio.save(wav_save_path2,data_ch_tensor, 24000)
                            # save result
                            with open(data1, 'wb') as f:
                                f.write(data_ch)
                            with open(data2, 'wb') as f:
                                f.write(data_en)

                        except Exception as e:
                            print("infer voice clone fail, cause {}".format(e))

                        print(f"模拟声音复刻生成，组{arrayIndex + 1}")
                        if arrayIndex == len(audioOutPaths) - 1:
                            arrayIndex = -1
                            curProgressIndex += 1
                    case 3:
                        #视频，请实现播报视频生成
                        print(f"请完成播报视频生成能力")
                        current_root_path = os.path.split(sys.argv[0])[0]
                        tmp_path=os.path.join(opt.outputDir,"audio")
                        exitPaths=os.listdir(tmp_path)
                        for i in range(len(exitPaths)):

                        # pic_path = opt.source_image
                        # audio_path = opt.driven_audio
                            pic_path=os.path.join(picInputPaths[i],"index.png")
                            audio_path=os.path.join(tmp_path,str(i+1)+"/"+str(timestringCounts[i])+"_chinese.wav")

                            # save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
                            save_dir = os.path.join(opt.outputDir, "video/"+str(i+1)+'/')
                            os.makedirs(save_dir, exist_ok=True)
                            pose_style = opt.pose_style
                            batch_size = opt.batch_size
                            input_yaw_list = opt.input_yaw
                            input_pitch_list = opt.input_pitch
                            input_roll_list = opt.input_roll
                            ref_eyeblink = opt.ref_eyeblink
                            ref_pose = opt.ref_pose
                            current_end_timestamp = int(datetime.datetime.now().timestamp())

                            create_log_file(save_dir, current_start_timestamp, current_end_timestamp)
                            sadtalker_paths = init_path(opt.checkpoint_dir, os.path.join(current_root_path, 'SadTalker/src/config'),
                                                        opt.size, opt.old_version, opt.preprocess)
                            preprocess_model = CropAndExtract(sadtalker_paths, device)

                            audio_to_coeff = Audio2Coeff(sadtalker_paths, device)

                            animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

                            # crop image and extract 3dmm from image
                            first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
                            os.makedirs(first_frame_dir, exist_ok=True)
                            print('3DMM Extraction for source image')
                            first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path,
                                                                                                   first_frame_dir,
                                                                                                   opt.preprocess, \
                                                                                                   source_image_flag=True,
                                                                                                   pic_size=opt.size)
                            if first_coeff_path is None:
                                print("Can't get the coeffs of the input")


                            if ref_eyeblink is not None:
                                ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
                                ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
                                os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
                                print('3DMM Extraction for the reference video providing eye blinking')
                                ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(ref_eyeblink,
                                                                                          ref_eyeblink_frame_dir,
                                                                                          opt.preprocess,
                                                                                          source_image_flag=False)
                            else:
                                ref_eyeblink_coeff_path = None

                            if ref_pose is not None:
                                if ref_pose == ref_eyeblink:
                                    ref_pose_coeff_path = ref_eyeblink_coeff_path
                                else:
                                    ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
                                    ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
                                    os.makedirs(ref_pose_frame_dir, exist_ok=True)
                                    print('3DMM Extraction for the reference video providing pose')
                                    ref_pose_coeff_path, _, _ = preprocess_model.generate(ref_pose, ref_pose_frame_dir,
                                                                                          opt.preprocess,
                                                                                          source_image_flag=False)
                            else:
                                ref_pose_coeff_path = None

                            # audio2ceoff
                            batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path,
                                             still=opt.still)
                            coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

                            # 3dface render
                            if opt.face3dvis:
                                from SadTalker.src.face3d.visualize import gen_composed_video

                                gen_composed_video(opt, device, first_coeff_path, coeff_path, audio_path,
                                                   os.path.join(save_dir, '3dface.mp4'))

                            # coeff2video
                            data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                                                       batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                                       expression_scale=opt.expression_scale, still_mode=opt.still,
                                                       preprocess=opt.preprocess, size=opt.size)

                            result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                                                 enhancer=opt.enhancer,
                                                                 background_enhancer=opt.background_enhancer,
                                                                 preprocess=opt.preprocess, img_size=opt.size)
                            new_path=os.path.join(save_dir,str(current_end_timestamp)+'_1.mp4')
                            shutil.move(result, new_path )
                            print('The generated video is named:', new_path)

                            # if not opt.verbose:
                            #     shutil.rmtree(save_dir)
                            #

                        print(f"模拟程序完成end.data")
                        endData = os.path.join(outputDir, 'end.data')
                        TryWriteFileTxt(endData,"1")
                        finish = True


                arrayIndex += 1
                time.sleep(0.1)  # 每隔2秒创建一个文件
        print("生成完成！")

    except KeyboardInterrupt:
            print("程序已停止。")
