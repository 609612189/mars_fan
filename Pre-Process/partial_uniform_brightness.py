'''
局部平均光照。
一个局部的gamma变换。
[论文阅读] LCC-NLM(局部颜色校正, 非线性mask)_local colour correction using nonlinear masking-CSDN博客
https://blog.csdn.net/j05073094/article/details/119595637
论文：Local color correction using non-linear masking "python"。
'''

import datetime
import numpy as np
import cv2
import os
import time

def test():
    inputDir = r"D:\Desktop\graduation_project(mars_fan)\Python_clip_code+coco_data\run_code_data\test_uniform_brightness2\input"
    outputDir=r"D:\Desktop\graduation_project(mars_fan)\Python_clip_code+coco_data\run_code_data\test_uniform_brightness2"

    radius=128
    outputDir = os.path.join(outputDir, "output-radius=" + str(radius))
    makedir(outputDir)

    for filepath, dirnames, filenames in os.walk(inputDir):
        for filename in filenames:
            if os.path.splitext(filename)[1]=='.png':
                input_image_path=os.path.join(filepath, filename)
                image_gray = cv2.imread(input_image_path,cv2.IMREAD_UNCHANGED)
                image_gray=image_gray.astype(np.float64)
                image_gray_inv = 255 - image_gray
                mask = cv2.blur(image_gray_inv, (radius, radius))
                mask_output_image_path = os.path.join(outputDir, "mask-"+filename)
                cv2.imwrite(mask_output_image_path, mask)
                lcc = 255 * ((image_gray / 255)**(2**((128 - mask) / 128)))
                output_image_path = os.path.join(outputDir, filename)
                cv2.imwrite(output_image_path, lcc)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def uniform_brightness(inputDir,outputDir,radius):
    for filepath, dirnames, filenames in os.walk(inputDir):
        for filename in filenames:
            if os.path.splitext(filename)[1]=='.png':
                input_image_path=os.path.join(filepath, filename)
                image_gray = cv2.imread(input_image_path,cv2.IMREAD_UNCHANGED)
                image_gray=image_gray.astype(np.float64)
                image_gray_inv = 255 - image_gray
                mask = cv2.blur(image_gray_inv, (radius, radius))
                # mask_output_image_path = os.path.join(outputDir, "mask-"+filename)
                # cv2.imwrite(mask_output_image_path, mask)
                lcc = 255 * ((image_gray / 255)**(2**((128 - mask) / 128)))
                output_image_path = os.path.join(outputDir, filename)
                cv2.imwrite(output_image_path, lcc)


def all_uniform_brightness(inputDir,outputDir,radius):
    input_labelTrainDir = os.path.join(inputDir, "train")
    input_labelValDir = os.path.join(inputDir, "val")
    input_labelTestDir = os.path.join(inputDir, "test")

    output_labelTrainDir = os.path.join(outputDir, "train")
    output_labelValDir = os.path.join(outputDir, "val")
    output_labelTestDir = os.path.join(outputDir, "test")

    makedir(output_labelTrainDir)
    makedir(output_labelValDir)
    makedir(output_labelTestDir)

    uniform_brightness(input_labelTrainDir,output_labelTrainDir,radius)
    uniform_brightness(input_labelValDir,output_labelValDir,radius)
    uniform_brightness(input_labelTestDir,output_labelTestDir,radius)

if __name__ == '__main__':
    time_start = time.time()
    inputDir = r"D:\Desktop\graduation_project(mars_fan)\dataset\20240425-2\testCenterClip6(keypoints)_3-25mpx-1280\images"
    outputDir = r"D:\Desktop\graduation_project(mars_fan)\dataset\20240425-2\testCenterClip6(keypoints)_3-25mpx-1280"
    radius=256
    outputDir = os.path.join(outputDir, "1280-uniform_brightness_images2-radius=" + str(radius))
    makedir(outputDir)
    all_uniform_brightness(inputDir,outputDir,radius)
    print("结束:", str(datetime.timedelta(seconds=time.time() - time_start)))

    # test()

