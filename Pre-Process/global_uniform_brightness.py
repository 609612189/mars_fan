'''
全局平均光照。
多个图片亮度和色度归一化处理 https://blog.csdn.net/qq_36638362/article/details/109311353
YOLOv8融入低照度图像增强算法---传统算法篇-CSDN博客  https://yolov5.blog.csdn.net/article/details/137045228 。
这个方法是gamma矫正。就是给每个像素归一化后，乘以gamma次方。
'''

'''
Math.pow() 方法是Java 中用于计算幂运算的函数。 这个方法接受两个参数，第一个参数是底数（base），第二个参数是指数（exponent）。 返回结果是底数的指数次幂。
numpy.clip :给定一个区间，区间之外的值将被剪裁到区间边缘。例如，如果指定间隔[0, 1] ，则小于 0 的值将变为 0，大于 1 的值将变为 1。
Opencv学习-LUT函数 https://blog.csdn.net/anjisi/article/details/53899222
'''
import datetime
import numpy as np
import cv2
import os
import time

def test():
    inputDir = r"D:\Desktop\graduation_project(mars_fan)\Python_clip_code+coco_data\run_code_data\test_uniform_brightness\input"
    outputDir=r"D:\Desktop\graduation_project(mars_fan)\Python_clip_code+coco_data\run_code_data\test_uniform_brightness\output"
    for filepath, dirnames, filenames in os.walk(inputDir):
        for filename in filenames:
            if os.path.splitext(filename)[1]=='.png':
                input_image_path=os.path.join(filepath, filename)
                image_gray = cv2.imread(input_image_path,cv2.IMREAD_UNCHANGED)
                Gamma = np.log(128.0 / 255.0) / np.log(cv2.mean(image_gray)[0] / 255.0)
                lookUpTable = np.zeros((1, 256), np.uint8) # 1行256列
                for i in range(256):
                    lookUpTable[0, i] = np.clip(pow(i / 255.0, Gamma) * 255.0, 0, 255)
                image_gray = cv2.LUT(image_gray, lookUpTable)
                output_image_path = os.path.join(outputDir, filename)
                cv2.imwrite(output_image_path, image_gray)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def uniform_brightness(inputDir,outputDir):
    for filepath, dirnames, filenames in os.walk(inputDir):
        for filename in filenames:
            if os.path.splitext(filename)[1]=='.png':
                input_image_path=os.path.join(filepath, filename)
                image_gray = cv2.imread(input_image_path,cv2.IMREAD_UNCHANGED)
                Gamma = np.log(128.0 / 255.0) / np.log(cv2.mean(image_gray)[0] / 255.0)
                lookUpTable = np.zeros((1, 256), np.uint8) # 1行256列
                for i in range(256):
                    lookUpTable[0, i] = np.clip(pow(i / 255.0, Gamma) * 255.0, 0, 255)
                image_gray = cv2.LUT(image_gray, lookUpTable)
                output_image_path = os.path.join(outputDir, filename)
                cv2.imwrite(output_image_path, image_gray)

def all_uniform_brightness(inputDir,outputDir):
    input_labelTrainDir = os.path.join(inputDir, "train")
    input_labelValDir = os.path.join(inputDir, "val")
    input_labelTestDir = os.path.join(inputDir, "test")

    output_labelTrainDir = os.path.join(outputDir, "train")
    output_labelValDir = os.path.join(outputDir, "val")
    output_labelTestDir = os.path.join(outputDir, "test")

    makedir(output_labelTrainDir)
    makedir(output_labelValDir)
    makedir(output_labelTestDir)

    uniform_brightness(input_labelTrainDir,output_labelTrainDir)
    uniform_brightness(input_labelValDir,output_labelValDir)
    uniform_brightness(input_labelTestDir,output_labelTestDir)

if __name__ == '__main__':
    time_start = time.time()
    inputDir = r"D:\Desktop\graduation_project(mars_fan)\dataset\20240425-2\testCenterClip6(keypoints)_3-25mpx-1280\images"
    outputDir = r"D:\Desktop\graduation_project(mars_fan)\dataset\20240425-2\testCenterClip6(keypoints)_3-25mpx-1280\uniform_brightness_images"
    makedir(outputDir)
    all_uniform_brightness(inputDir,outputDir)
    print("结束:", str(datetime.timedelta(seconds=time.time() - time_start)))
    # test()
