'''
懒的整理了，仅供参考。
5m/px降采样到25m/px、50m/px、100m/px。
'''

from osgeo import gdal, gdalconst, ogr
import os
import glob
import numpy as np
import math
import time
import datetime


'''
osgeo.gdal module — GDAL documentation  https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.Warp
gdalwarp — GDAL documentation  https://gdal.org/programs/gdalwarp.html
GTiff格式的可选项，比如options=["TILED=YES", "COMPRESS=LZW"]：GTiff -- GeoTIFF File Format — GDAL documentation  https://gdal.org/drivers/raster/gtiff.html#raster-gtiff 
[gdal-dev] Fwd: Re: Request for clarification on SPARSE_OK=TRUE and empty tiles relation  https://lists.osgeo.org/pipermail/gdal-dev/2016-May/044429.html
warp可以裁剪图像（使用shapefile文件裁剪）、重采样重投影图像（设置参数，分辨率是多少）、拼接图像。
不管driver.Create()有没有["SPARSE_OK=True"]，都会创建一个大小为0的图像，而不会直接把图像都填充上像素0。
试了这个方法，还是无法使NoData数据没有像素。解决此问题的一种方法是创建一个适当大小的空稀疏目标GeoTIFF，然后将每个输入数据集gdalwrp到其中（-wo SKIP_NOSOURCE=YES）。

'''
def test_gdalwarp(path):
    start_time = time.time()

    os.chdir(path)
    if os.path.exists('warp_mosaiced_image.tif'):
        os.remove('warp_mosaiced_image.tif')
    # in_files = glob.glob("*.TIF")

    in_files=[]
    for filepath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if os.path.splitext(filename)[1]=='.tif':
                in_files.append(os.path.join(filepath, filename))
    #貌似还需要设置dstSRS，目标的空间参考系，不然得到的空间参考系和原图不一致。merge和Mosaic_all()，得到的空间参考系都和CTX马赛克不一样。
    gdal.Warp('warp_mosaiced_image.tif',in_files,creationOptions =["SPARSE_OK=True","COMPRESS=LZW","BIGTIFF=YES"],warpOptions=["SKIP_NOSOURCE=YES"])# warpOptions=["SKIP_NOSOURCE=YES"] warpOptions=["INIT_DEST=[5]"]

    end_time = time.time()
    print(str(datetime.timedelta(seconds=end_time - start_time)))
    return 0

'''
21秒把47420×47420的图降采样到23710×23710，就是除以2，且LZW压缩。
9秒把47420×47420的图降采样到11855×11855，就是除以4，且LZW压缩。
2秒把47420×47420的图降采样到5927×5927，就是除以8，且LZW压缩。

'''
def testResizeImg(srcPath, dstPath,divisor):
    start_time = time.time()

    fileName=srcPath.split("\\")[-1]
    dataset = gdal.Open(srcPath, gdalconst.GA_ReadOnly)
    # MurrayLab_CTX_V01_E-036_N-28_Mosaic.tif：47420×47420。47420÷2÷2=11855
    rows = dataset.RasterYSize
    cols = dataset.RasterXSize
    gdal.Warp(dstPath+'/resize_'+str(divisor)+"_"+fileName,srcPath,
              resampleAlg ="near",width =cols / divisor, height =rows / divisor,creationOptions =["SPARSE_OK=True","COMPRESS=LZW","BIGTIFF=YES"],warpOptions=["SKIP_NOSOURCE=YES"])# warpOptions=["SKIP_NOSOURCE=YES"] warpOptions=["INIT_DEST=[5]"]

    end_time = time.time()
    print(str(datetime.timedelta(seconds=end_time - start_time)))

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

'''
47420×47420，5m/px，重采样到：
÷5，9484×9484，25m/px
÷10，4742×4742，50m/px
÷20，2371×2371，100m/px
'''
def resizeImg(rootDir):
    start_time = time.time()
    Dir5=rootDir+"/5-meter_per_pixel/"
    Dir25=rootDir+"/25-meter_per_pixel/"
    Dir50=rootDir+"/50-meter_per_pixel/"
    Dir100=rootDir+"/100-meter_per_pixel/"
    makedir(Dir5)
    makedir(Dir25)
    makedir(Dir50)
    makedir(Dir100)

    divisorList=[5,10,20]
    meter_per_pixel_List=[25,50,100]

    Dir100FileList=[]
    for filepath, dirnames, filenames in os.walk(Dir100):
        for fileName in filenames:
            Dir100FileList.append(fileName.replace("100-","")) # 不能用fileName.split("-")[1]，因为有这种名字 100-MurrayLab_CTX_V01_E-016_N44_Mosaic.tif，得到"MurrayLab_CTX_V01_E"。
    for filepath, dirnames, filenames in os.walk(Dir5):
        for fileName in filenames:
            if fileName not in Dir100FileList:
                srcPath=os.path.join(filepath, fileName)
                dataset = gdal.Open(srcPath, gdalconst.GA_ReadOnly)
                rows = dataset.RasterYSize
                cols = dataset.RasterXSize
                gdal.Warp(Dir25 + str(meter_per_pixel_List[0]) + "-" + fileName, srcPath,
                          resampleAlg="near", width=cols / divisorList[0], height=rows / divisorList[0])
                gdal.Warp(Dir50 + str(meter_per_pixel_List[1]) + "-" + fileName, srcPath,
                          resampleAlg="near", width=cols / divisorList[1], height=rows / divisorList[1])
                gdal.Warp(Dir100 + str(meter_per_pixel_List[2]) + "-" + fileName, srcPath,
                          resampleAlg="near", width=cols / divisorList[2], height=rows / divisorList[2])
                del dataset
            #     print("Y:",fileName)
            # else:
            #     print("N:",fileName)

    end_time = time.time()
    print(str(datetime.timedelta(seconds=end_time - start_time)))



'''
47420×47420，5m/px，重采样到：
÷5，9484×9484，25m/px
÷10，4742×4742，50m/px
÷20，2371×2371，100m/px
'''
def filterResizeImg(rootDir):
    start_time = time.time()
    Dir5=rootDir+"/5-meter_per_pixel/"
    Dir25=rootDir+"/25-meter_per_pixel/"
    Dir50=rootDir+"/50-meter_per_pixel/"
    Dir100=rootDir+"/100-meter_per_pixel/"
    makedir(Dir5)
    makedir(Dir25)
    makedir(Dir50)
    makedir(Dir100)

    divisorList=[5,10,20]
    meter_per_pixel_List=[25,50,100]

    Dir100FileList=[]
    for filepath, dirnames, filenames in os.walk(Dir100):
        for fileName in filenames:
            # srcPath = os.path.join(filepath, fileName)
            # dataset = gdal.Open(srcPath, gdalconst.GA_ReadOnly)
            # sub_img=dataset.ReadAsArray(0,0, dataset.RasterXSize,dataset.RasterYSize)
            # if np.count_nonzero(sub_img)<1:
            #     Dir100FileList.append(fileName.replace("100-",""))
            # del sub_img
            # del dataset
            Dir100FileList.append(fileName.replace("100-", ""))

    for filepath, dirnames, filenames in os.walk(Dir5):
        for fileName in filenames:
            if fileName not in Dir100FileList:
                srcPath=os.path.join(filepath, fileName)
                dataset = gdal.Open(srcPath, gdalconst.GA_ReadOnly)
                rows = dataset.RasterYSize
                cols = dataset.RasterXSize
                gdal.Warp(Dir25 + str(meter_per_pixel_List[0]) + "-" + fileName, srcPath,
                          resampleAlg="near", width=cols / divisorList[0], height=rows / divisorList[0])
                gdal.Warp(Dir50 + str(meter_per_pixel_List[1]) + "-" + fileName, srcPath,
                          resampleAlg="near", width=cols / divisorList[1], height=rows / divisorList[1])
                gdal.Warp(Dir100 + str(meter_per_pixel_List[2]) + "-" + fileName, srcPath,
                          resampleAlg="near", width=cols / divisorList[2], height=rows / divisorList[2])
                del dataset
                print(fileName)
            #     print("Y:",fileName)
            # else:
            #     print("N:",fileName)

    end_time = time.time()
    print(str(datetime.timedelta(seconds=end_time - start_time)))


'''
47420分解出的因子：10，2，2371。2371是素数，无法再分解。
47420×47420，5m/px，重采样到：
÷5，9484×9484，25m/px
÷10，4742×4742，50m/px
÷20，2371×2371，100m/px

÷40，200m/px
÷100，500m/px
÷200，1000m/px
'''
def filterResizeImg2(rootDir):
    start_time = time.time()
    Dir5=rootDir+"/5-meter_per_pixel/"
    makedir(Dir5)
    Dir200=rootDir+"/200-meter_per_pixel/"
    makedir(Dir200)
    Dir500=rootDir+"/500-meter_per_pixel/"
    makedir(Dir500)
    Dir1000=rootDir+"/1000-meter_per_pixel/"
    makedir(Dir1000)

    divisorList=[40,100,200]
    meter_per_pixel_List=[200,500,1000]

    # Dir100FileList=[]
    # for filepath, dirnames, filenames in os.walk(Dir100):
    #     for fileName in filenames:
    #         srcPath = os.path.join(filepath, fileName)
    #         dataset = gdal.Open(srcPath, gdalconst.GA_ReadOnly)
    #         sub_img=dataset.ReadAsArray(0,0, dataset.RasterXSize,dataset.RasterYSize)
    #         if np.count_nonzero(sub_img)<1:
    #             Dir100FileList.append(fileName.replace("100-",""))
    #         del sub_img
    #         del dataset

    for filepath, dirnames, filenames in os.walk(Dir5):
        for fileName in filenames:
            srcPath=os.path.join(filepath, fileName)
            dataset = gdal.Open(srcPath, gdalconst.GA_ReadOnly)
            rows = dataset.RasterYSize
            cols = dataset.RasterXSize
            gdal.Warp(Dir200 + str(meter_per_pixel_List[0]) + "-" + fileName, srcPath,
                      resampleAlg="near", width=cols / divisorList[0], height=rows / divisorList[0])
            gdal.Warp(Dir500 + str(meter_per_pixel_List[1]) + "-" + fileName, srcPath,
                      resampleAlg="near", width=cols / divisorList[1], height=rows / divisorList[1])
            gdal.Warp(Dir1000 + str(meter_per_pixel_List[2]) + "-" + fileName, srcPath,
                      resampleAlg="near", width=cols / divisorList[2], height=rows / divisorList[2])
            del dataset
            print(fileName)
            #     print("Y:",fileName)
            # else:
            #     print("N:",fileName)

    end_time = time.time()
    print(str(datetime.timedelta(seconds=end_time - start_time)))



if __name__ == '__main__':
    # 步骤1：测试降采样
    # srcPath=r"F:\graduation_project(mars_fan)\CTX\CTX_tif\test_Mosaic2NewRaster\MurrayLab_CTX_V01_E-036_N-28_Mosaic.tif"
    # dstPath=r"F:\graduation_project(mars_fan)\CTX\CTX_tif\test_Mosaic2NewRaster"
    # divisor=8 # 降采样倍数。比如divisor=2，那么width =cols/2,height =rows/2。
    # testResizeImg(srcPath, dstPath,divisor)

    # 步骤2、降采样代码：
    # rootDir=r"G:\new\graduation_project(mars_fan)\CTX\allCTX"
    # resizeImg(rootDir)

    # 步骤3：降采样后是黑色的，就是没降采样成功的，都重新降采样。
    # rootDir=r"G:\graduation_project(mars_fan)\CTX\allCTX"
    # filterResizeImg(rootDir)

    # 步骤4：得到200m/px，500m/px，1000m/px的图。
    # rootDir=r"G:\graduation_project(mars_fan)\CTX\allCTX"
    # filterResizeImg2(rootDir)

    # 步骤5
    rootDir=r"G:\graduation_project(mars_fan)\CTX\allCTX2"
    filterResizeImg(rootDir)
    # filterResizeImg2(rootDir)