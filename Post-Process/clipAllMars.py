'''
把火星的所有CTX图片，裁剪成可以被训练好的Yolo模型预测的图片大小，就是25m/px，1280px。
'''
import glob
from osgeo import gdal,gdalconst,ogr
import warnings
from PIL import Image
import numpy as np
import os
import time
import datetime

time_start = time.time()
warnings.filterwarnings("ignore")

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def tif2png(outputDir):
    ImgTifDir = outputDir + "/Img/tif"
    ImgPngDir = outputDir + "/Img/png"
    makedir(ImgPngDir)
    tifPathList = glob.glob(ImgTifDir + '\\' + "*.tif")
    for tifPath in tifPathList:
        image = Image.open(tifPath)  # 打开tiff图像
        pngPath=ImgPngDir+"/"+os.path.splitext(os.path.basename(tifPath))[0]+".png"
        image.save(pngPath)  # 更改后缀名，保存png图像

def clipAllMars(rasterPath,outputDir,clipSize,startPoint):
    raster = gdal.Open(rasterPath)
    i, j = startPoint
    Move = clipSize
    end_row , end_col =  raster.RasterYSize, raster.RasterXSize # 栅格数据的宽度（X方向的像素数），栅格数据的高度（Y方向的像素数）
    transform = raster.GetGeoTransform()
    geo_startX=transform[0] # 左上角X坐标
    geo_startY=transform[3] # 左上角Y坐标
    pixelWidth = transform[1]
    pixelHeight = -transform[5]
    spaceResolution = round(pixelWidth)

    ImgTifDir = outputDir + "/Img/tif"
    makedir(ImgTifDir)

    while i < end_row:
        while j < end_col:
            # (xmin,ymin)是左下角点，(xmax,ymax)是右上角点。因为ArcGis以左下角为原点，横轴x轴，竖轴y轴。
            xmin = geo_startX + j * pixelWidth # 左下角x
            xmax = geo_startX + (j + clipSize) * pixelWidth # 右上角x
            ymin = geo_startY - (i + clipSize) * pixelHeight # 左下角y
            ymax = geo_startY - i * pixelHeight # 右上角y
            # 命名：空间分辨率pixelWidth_空间分辨率pixelHeight_左下角点x_左下角点y_右上角点x_右上角点y。不能用"-"，因为点坐标有负号。
            fileName=str(pixelWidth)+"_"+str(pixelHeight)+"_"+str(xmin)+"_"+str(ymin)+"_"+str(xmax)+"_"+str(ymax)
            gdal.Warp(ImgTifDir + "/" + fileName + '.tif', raster,
                      outputBounds=[xmin, ymin, xmax, ymax])
            j += Move
        i += Move
        j = 0

if __name__ == '__main__':
    rasterPath=r"G:\graduation_project(mars_fan)\CTX\run_code_data\clipAllMars\input\25-meter_per_pixel_latitude[-50,50].vrt"
    outputDir=r"G:\graduation_project(mars_fan)\CTX\run_code_data\clipAllMars\output"
    clipSize = 1280
    startPointArray = np.array([
        [0, 0],
        [0, clipSize / 2],
        [clipSize / 2, 0],
        [clipSize / 2, clipSize / 2]
    ])
    clipAllMars(rasterPath,outputDir,clipSize,startPointArray[0])
    tif2png(outputDir)
    print("结束:", str(datetime.timedelta(seconds=time.time() - time_start)))


