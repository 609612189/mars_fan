import shapefile
import glob
import math
from osgeo import gdal, gdalconst, ogr
import warnings
import datetime
from PIL import Image
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import os
import time

'''
图片名：background-空间分辨率_x坐标_y坐标。
随机坐标点，然后裁剪，每个空间分辨率都要裁剪，不同分辨率的裁剪坐标不要一样，如果裁剪到扇，就把图片都删除掉。
裁剪完成后，分配到train val test。
ctx.tif
temp.img
ctx.png
'''
# def clipBackground(rasterPathList,backgroundDir,shapefilePath,backgroundCount,clipSize):
#
#     # rasterList=[]
#     # for rasterPath in rasterPathList:
#     #     raster = gdal.Open(rasterPath)
#     #     rasterList.append(raster)
#
#     ImgTifDir=backgroundDir+"/Img/tif/"
#     makedir(ImgTifDir)
#
#     count=int(backgroundCount/4)
#     haveClipCount=0
#     # for raster in rasterList:
#     for rasterPath in rasterPathList:
#         raster = gdal.Open(rasterPath)
#         haveClipCount = 0
#         while haveClipCount<int(backgroundCount/4):
#             transform = raster.GetGeoTransform()
#             pixelWidth = transform[1]
#             pixelHeight = -transform[5]
#             spaceResolution = round(pixelWidth)
#             # x在[-100 00000,100 00000]，y在[-300 0000,200 0000]
#             xmin=-10000000+20000000*random.uniform(0,1)
#             ymin=-3000000+5000000*random.uniform(0,1)
#             fileName="background" + "-" + str(spaceResolution) +"_"+str(xmin)+"_"+str(ymin)
#             gdal.Warp(ImgTifDir + "/" +fileName + '.tif', raster,
#                       outputBounds=[xmin, ymin, xmin + clipSize * pixelWidth,
#                                     ymin + clipSize * pixelHeight])
#             rasterClip = gdal.Open(ImgTifDir + "/" + fileName+ '.tif')
#
#             target_ds=ShpToRaster(rasterClip, shapefilePath, backgroundDir,fileName)
#             sub_label = target_ds.ReadAsArray(0, 0, target_ds.RasterXSize, target_ds.RasterYSize)
#             use_pixel_num = np.count_nonzero(sub_label)
#
#             del target_ds
#             del sub_label
#             sub_img = rasterClip.ReadAsArray(0, 0, rasterClip.RasterXSize, rasterClip.RasterYSize)
#             del rasterClip
#             if use_pixel_num > 0 or np.count_nonzero(sub_img)<1: # 说明有扇 or 图裁剪到没像素的区域。
#                 while True:
#                     try:
#                         os.remove(ImgTifDir+"/"+fileName + '.tif')
#                         break
#                     except PermissionError:
#                         time.sleep(1)
#                         print("等待1秒")
#                 while True:
#                     try:
#                         os.remove(backgroundDir+"/temp/"+"temp_"+fileName+".img")
#                         break
#                     except PermissionError:
#                         time.sleep(1)
#                         print("等待1秒")
#             else:
#                 haveClipCount=haveClipCount+1
#             del sub_img
#         del raster


time_start = time.time()
warnings.filterwarnings("ignore")


# 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def defineLog(log_path):
    makedir(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + time.strftime("%Y-%m-%d#%H-%M-%S", time.localtime()) + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)


'''
给shp文件新增'id_label'字段，然后给每行shp面的'id_label'字段赋值：1、2、3、4、5....。[1,1501]，1501个扇。
'''
def CreatShapeID(inshp):
    in_ds = ogr.Open(inshp, True)  # False - read only, True - read/write
    in_layer = in_ds.GetLayer(0)  # 这里的in_ds就是.shp文件，相当于in_ds.GetLayerByIndex(0)。总共就1个Layer，Layer里存了5个shp面。
    in_lydefn = in_layer.GetLayerDefn()  # GetLayerDefn得到关于Layer的一些架构信息，比如有几个字段，字段名是什么。

    name = 'id_label'
    # print(dir(in_lydefn))
    for i in range(in_lydefn.GetFieldCount()):
        if in_lydefn.GetFieldDefn(i).GetName() == name:
            return

    field = ogr.FieldDefn(name, ogr.OFTInteger)
    field.SetWidth(8)
    field.SetPrecision(3)
    in_layer.CreateField(field)

    for i, feature in enumerate(in_layer):
        feature.SetField(name, i + 1)
        in_layer.SetFeature(feature)
        # print(i, ":set is ok ")
    del in_ds


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def showImgAndCategory(Showdir, imgPath, labelList, imgName):
    makedir(Showdir)
    img = cv2.imread(imgPath)
    for label in labelList[1:]:
        boxCenter_x = label[0]
        boxCenter_y = label[1]
        boxWidth = label[2]
        boxHeight = label[3]
        point_x = int(label[4])
        point_y = int(label[5])
        segment_points=np.array(label[7:], dtype=np.int32).reshape(-1,1,2)
        xmin=int(boxCenter_x-boxWidth/2)
        xmax=int(boxCenter_x+boxWidth/2)
        ymin=int(boxCenter_y-boxHeight/2)
        ymax=int(boxCenter_y+boxHeight/2)
        img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 6) # 这个方法中，xmin,ymin,xmax,ymax必须是整数。
        img=cv2.polylines(img, [segment_points], True, (0, 255, 0), 2)
        # cv.circle(img, point, point_size, point_color, thickness)
        img = cv2.circle(img, (point_x, point_y), 1, (0, 0, 255), 10)
    plt.imsave(Showdir + '/' + str(imgName) + '.png',img)

'''
geo的地理坐标系。
y
^
|
|
0————————>x              
yolo标注的坐标系：
0————————>x 
|
⬇
y                  
'''
def testCenterClip(shapefilePath,rasterPathList,outputDir,clipSize,dimFan):
    raster = gdal.Open(rasterPathList[0])
    ImgTifDir = outputDir + "/Img/tif"
    makedir(ImgTifDir)

    labelList = []
    haveCliped = set()
    fanId=0

    spaceResolution_list = [5, 25, 50, 100, 500, 1000]
    divisorList = [1, 5, 10, 20, 100, 200]
    file1 = shapefile.Reader(shapefilePath)
    shapes = file1.shapes()
    records = file1.records()
    for shape in shapes:
        print(fanId)
        if fanId in haveCliped or fanId in dimFan:
            fanId=fanId+1
            continue
        # if fanId<1481:
        #     fanId=fanId+1
        #     continue
        array=shape.bbox
        shape_xmin=shape.bbox[0]
        shape_ymin=shape.bbox[1]
        shape_xmax=shape.bbox[2]
        shape_ymax=shape.bbox[3]
        shape_xRange=shape_xmax-shape_xmin
        shape_yRange=shape_ymax-shape_ymin

        transform = raster.GetGeoTransform()
        pixelWidth = transform[1]
        pixelHeight = -transform[5]
        spaceResolution = round(pixelWidth)  # 四舍五入，浮点数变整数。
        #如果5次后，还是做不到，没有重复的裁剪，就扩大spaceResolution。
        if shape_xRange * shape_yRange <= math.pow(spaceResolution_list[1] * clipSize *0.1, 2):
            spaceResolution=spaceResolution_list[0]
            pixelWidth=pixelWidth*divisorList[0]
            pixelHeight=pixelHeight*divisorList[0]
            # 就在5m / px上裁剪。
        elif shape_xRange<spaceResolution_list[1]*clipSize and shape_yRange<spaceResolution_list[1]*clipSize:
            spaceResolution = spaceResolution_list[1]
            pixelWidth=pixelWidth*divisorList[1]
            pixelHeight=pixelHeight*divisorList[1]
        elif shape_xRange<spaceResolution_list[2]*clipSize and shape_yRange<spaceResolution_list[2]*clipSize:
            spaceResolution = spaceResolution_list[2]
            pixelWidth=pixelWidth*divisorList[2]
            pixelHeight=pixelHeight*divisorList[2]

        whileCount=5
        while True: # 达到do-while的效果。
            # random.random()随机范围：[0,1)。random.uniform(a,b)随机范围：[a,b]，浮点数。
            # (xmin,ymin)是左下角点，(xmax,ymax)是右上角点。因为ArcGis以左下角为原点，横轴x轴，竖轴y轴。
            xmin = (shape_xmax - clipSize * pixelWidth) + random.uniform(0.01, 0.99) * (
                        shape_xmin - (shape_xmax - clipSize * pixelWidth))
            ymin = (shape_ymax - clipSize * pixelHeight) + random.uniform(0.01, 0.99) * (
                        shape_ymin - (shape_ymax - clipSize * pixelHeight))
            xmax = xmin + clipSize * pixelWidth
            ymax = ymin + clipSize * pixelHeight
            whileCount=whileCount-1
            if isClip(file1,dimFan,xmin,ymin,xmax,ymax) or whileCount==0:
                break

        '''
        tempList=
        [
            文件名，
            [目标框中心坐标x，目标框中心坐标y，目标框宽，目标框高，点1x，点1y]
            [目标框中心坐标x，目标框中心坐标y，目标框宽，目标框高，点1x，点1y]
        ]
        '''

        tempList = []
        fanList=[]
        for i in range(len(records)):
            if i in dimFan:
                continue
            geo_point_x = records[i]["point_x"]
            geo_point_y = records[i]["point_y"]
            shape_xmin_temp = shapes[i].bbox[0]
            shape_ymin_temp = shapes[i].bbox[1]
            shape_xmax_temp = shapes[i].bbox[2]
            shape_ymax_temp = shapes[i].bbox[3]
            shape_xRange_temp=shape_xmax_temp-shape_xmin_temp
            shape_yRange_temp=shape_ymax_temp-shape_ymin_temp

            inner_xmin=max(xmin,shape_xmin_temp)
            inner_ymin=max(ymin,shape_ymin_temp)
            inner_xmax=min(xmax,shape_xmax_temp)
            inner_ymax=min(ymax,shape_ymax_temp)
            inner=clamp(inner_xmax-inner_xmin,0)*clamp(inner_ymax-inner_ymin,0)


            # 点坐标在截图范围内，目标框在截图范围内。目标框大小不是小目标。
            if xmin<geo_point_x and geo_point_x<xmax and ymin<geo_point_y and geo_point_y<ymax\
                    and inner>=shape_xRange_temp*shape_yRange_temp*0.8: # 如果裁剪到80%，就认为裁剪到扇了，不再次裁剪。

                if shape_xmin_temp < xmin:
                    shape_xmin_temp = xmin
                if shape_xmax_temp > xmax:
                    shape_xmax_temp = xmax
                if shape_ymin_temp < ymin:
                    shape_ymin_temp = ymin
                if shape_ymax_temp > ymax:
                    shape_ymax_temp = ymax

                geo_boxWidth = shape_xmax_temp - shape_xmin_temp
                geo_boxHeight = shape_ymax_temp - shape_ymin_temp
                geo_boxCenter_x = shape_xmin_temp + geo_boxWidth / 2
                geo_boxCenter_y = shape_ymin_temp + geo_boxHeight / 2

                boxWidth=geo_boxWidth/pixelWidth
                boxHeight=geo_boxHeight/pixelHeight
                boxCenter_x=(geo_boxCenter_x-xmin)/pixelWidth
                boxCenter_y=clipSize-((geo_boxCenter_y-ymin)/pixelHeight)

                point_x = (geo_point_x-xmin)/pixelWidth
                point_y = clipSize-((geo_point_y-ymin)/pixelHeight)

                segmentPointList = []
                for point in shapes[i].points:
                    geo_segment_point_x=point[0]
                    geo_segment_point_y=point[1]
                    if geo_segment_point_x < xmin:
                        geo_segment_point_x = xmin
                    if geo_segment_point_x > xmax:
                        geo_segment_point_x = xmax
                    if geo_segment_point_y < ymin:
                        geo_segment_point_y = ymin
                    if geo_segment_point_y > ymax:
                        geo_segment_point_y = ymax
                    segment_point_x = (geo_segment_point_x - xmin) / pixelWidth
                    segment_point_y = clipSize - ((geo_segment_point_y - ymin) / pixelHeight)
                    segmentPointList.append(segment_point_x)
                    segmentPointList.append(segment_point_y)

                # 2*clipSize。因为要写2，但是后面要除以1280归一化。
                tempList.append([boxCenter_x,boxCenter_y,boxWidth,boxHeight,point_x,point_y,2.0*clipSize]+segmentPointList)
                fanList.append(i)

        if len(fanList)!=0:
            haveCliped.update(set(fanList))

            fanListString=""
            for temp in fanList:
                fanListString=fanListString+str(temp)+"_"
            fanListString=fanListString.rstrip("_")

            fileName=str(fanId)+"_"+str(spaceResolution)+"-"+fanListString
            tempList.insert(0,fileName+".txt")
            labelList.append(tempList)

            gdal.Warp(ImgTifDir+"/"+fileName+ '.tif', raster, resampleAlg="cubic",
                  outputBounds=[xmin,ymin,xmax,ymax],width=clipSize,height=clipSize)

            # 打印图像
            ShowDir = outputDir + '/Show'
            showImgAndCategory(ShowDir, ImgTifDir+"/"+fileName+ '.tif', tempList, "show-" + fileName)

        fanId=fanId+1

    LabelDir = outputDir + "/Label"
    makedir(LabelDir)
    for label in labelList:
        labelFileName=label[0]
        labelFilePath=os.path.join(LabelDir,labelFileName)
        f = open(labelFilePath,"w")
        fileContent=""
        for labelContent in label[1:]:
            fileContent=fileContent+str(0) # 0是类别0，即火星扇这个类别
            for count in labelContent:
                fileContent=fileContent+" "
                fileContent=fileContent+str(count/clipSize) # yolo要求，标准化为 0 到 1 之间。
            # fileContent=fileContent+" 2" # 可见性标志 v 定义为 v=0：未标记，v=1：标记但不可见，v=2：标记且可见。
            fileContent=fileContent+"\n"

        f.write(fileContent)
        f.close()


def clamp(number,min):
    if number<min:
        return min
    else:
        return number


def isClip(file1,dimFan,xmin,ymin,xmax,ymax):
    shapes = file1.shapes()
    records = file1.records()
    for i in range(len(records)):
        if i in dimFan:
            continue
        shape_xmin_temp = shapes[i].bbox[0]
        shape_ymin_temp = shapes[i].bbox[1]
        shape_xmax_temp = shapes[i].bbox[2]
        shape_ymax_temp = shapes[i].bbox[3]
        shape_xRange_temp = shape_xmax_temp - shape_xmin_temp
        shape_yRange_temp = shape_ymax_temp - shape_ymin_temp

        inner_xmin = max(xmin, shape_xmin_temp)
        inner_ymin = max(ymin, shape_ymin_temp)
        inner_xmax = min(xmax, shape_xmax_temp)
        inner_ymax = min(ymax, shape_ymax_temp)
        inner = clamp(inner_xmax - inner_xmin, 0) * clamp(inner_ymax - inner_ymin, 0)

        if inner==0 or abs(inner-shape_xRange_temp*shape_yRange_temp)<0.001:
            continue
        else:
            return False
    return True

'''
roboflow不支持上传tif文件。所以得把tif转成png格式。
'''
def tif2png(outputDir):
    ImgTifDir = outputDir + "/Img/tif"
    ImgPngDir = outputDir + "/Img/png"
    makedir(ImgPngDir)
    tifPathList = glob.glob(ImgTifDir + '\\' + "*.tif")
    for tifPath in tifPathList:
        image = Image.open(tifPath)  # 打开tiff图像
        pngPath=ImgPngDir+"/"+os.path.splitext(os.path.basename(tifPath))[0]+".png"
        image.save(pngPath)  # 更改后缀名，保存png图像


def splitTrainValTest(ROOT_DIR):
    IMAGE_DIR = os.path.join(ROOT_DIR, "Img/png")
    ANNOTATION_DIR = os.path.join(ROOT_DIR, "Label")
    SHOW_DIR = os.path.join(ROOT_DIR, "Show")

    files = os.listdir(IMAGE_DIR)  # 读入文件夹
    fileCount = len(files)  # 统计文件夹中的文件个数
    trainCount=int(fileCount * 0.8)
    valCount = int(fileCount * 0.2)  # int()函数向下取整
    testCount = fileCount - trainCount - valCount
    print("Img截图总数: ", fileCount)
    print("训练集、验证集、测试集数量: ", trainCount, valCount, testCount)

    imageDir=os.path.join(ROOT_DIR,"result\multi\images")
    labelDir=os.path.join(ROOT_DIR,"result\multi\labels")
    imageTrainDir=os.path.join(imageDir,"train")
    imageValDir=os.path.join(imageDir,"val")
    imageTestDir=os.path.join(imageDir,"test")
    labelTrainDir=os.path.join(labelDir,"train")
    labelValDir=os.path.join(labelDir,"val")
    labelTestDir=os.path.join(labelDir,"test")
    makedir(imageDir)
    makedir(labelDir)
    makedir(imageTrainDir)
    makedir(imageValDir)
    makedir(imageTestDir)
    makedir(labelTrainDir)
    makedir(labelValDir)
    makedir(labelTestDir)
    showDir=os.path.join(ROOT_DIR,"result\show")
    showTrainDir=os.path.join(showDir,"train")
    showValDir=os.path.join(showDir,"val")
    showTestDir=os.path.join(showDir,"test")
    makedir(showTrainDir)
    makedir(showValDir)
    makedir(showTestDir)

    unallocatedList=[]
    # [[fileName,[fanList]],[],[],[]]
    for filepath, dirnames, filenames in os.walk(IMAGE_DIR):
        for fileName in filenames:
            fanList=os.path.splitext(fileName)[0].split("-")[1].split("_")
            tempList=[]
            tempList.append(fileName)
            tempList.append(fanList)
            unallocatedList.append(tempList)
    random.shuffle(unallocatedList)

    trainFanSet=set()
    trainList = unallocatedList[0:int(fileCount * 0.05)]
    unallocatedList=unallocatedList[int(fileCount * 0.05):]
    for tempList in trainList:
        trainFanSet.update(tempList[1])

    addImg(unallocatedList, trainList, trainFanSet)
    while(len(trainList)<trainCount):
        trainFanSet.update(unallocatedList[0][1])
        trainList.append(unallocatedList[0])
        unallocatedList.remove(unallocatedList[0])
        addImg(unallocatedList, trainList, trainFanSet)

    valFanSet=set()
    valList=unallocatedList[0:int(fileCount * 0.05)]
    unallocatedList=unallocatedList[int(fileCount * 0.05):]
    for tempList in valList:
        valFanSet.update(tempList[1])
    addImg(unallocatedList, valList, valFanSet)
    while(len(valList)<valCount):
        valFanSet.update(unallocatedList[0][1])
        valList.append(unallocatedList[0])
        unallocatedList.remove(unallocatedList[0])
        addImg(unallocatedList, valList, valFanSet)

    testFanSet=set()
    testList=unallocatedList
    for tempList in testList:
        testFanSet.update(tempList[1])
    print("实际上，训练集、验证集、测试集数量: ", len(trainList), len(valList), len(testList))
    print("训练集的扇_FID：",sorted(trainFanSet))
    print("验证集的扇_FID：",sorted(valFanSet))
    print("测试集的扇_FID：",sorted(testFanSet))

    haveClipFanSet=set()
    haveClipFanSet.update(trainFanSet)
    haveClipFanSet.update(valFanSet)
    haveClipFanSet.update(testFanSet)
    print("haveClipFanCount:",str(len(haveClipFanSet)))
    print("haveClipFan:",str(sorted(haveClipFanSet)))

    allFan=set()
    for fanId in range(0, 1501): # range(0, 1501)得到[0,1500]，但是是数字，不是字符串。
        allFan.add(str(fanId))
    notClipFanSet=allFan-haveClipFanSet
    print("notClipFanCount:"+str(len(notClipFanSet)))
    print("notClipFanSet_FID:"+str(sorted(notClipFanSet)))

    trainImgFilenameSet=set()
    valImgFilenameSet=set()
    testImgFilenameSet=set()

    trainLabelSet=set()
    valLabelSet=set()
    testLabelSet=set()

    for tempList in trainList:
        trainImgFilenameSet.add(tempList[0])
        trainLabelSet.add(os.path.splitext(tempList[0])[0]+".txt")
    for tempList in valList:
        valImgFilenameSet.add(tempList[0])
        valLabelSet.add(os.path.splitext(tempList[0])[0]+".txt")
    for tempList in testList:
        testImgFilenameSet.add(tempList[0])
        testLabelSet.add(os.path.splitext(tempList[0])[0]+".txt")


    for filepath, dirnames, filenames in os.walk(IMAGE_DIR):
        for filename in filenames:
            if filename in trainImgFilenameSet:
                shutil.copy(os.path.join(filepath, filename), imageTrainDir)
            elif filename in valImgFilenameSet:
                shutil.copy(os.path.join(filepath, filename), imageValDir)
            elif filename in testImgFilenameSet:
                shutil.copy(os.path.join(filepath, filename), imageTestDir)

    for filepath, dirnames, filenames in os.walk(ANNOTATION_DIR):
        for filename in filenames:
            if filename in trainLabelSet:
                shutil.copy(os.path.join(filepath, filename), labelTrainDir)
            elif filename in valLabelSet:
                shutil.copy(os.path.join(filepath, filename), labelValDir)
            elif filename in testLabelSet:
                shutil.copy(os.path.join(filepath, filename), labelTestDir)

    for filepath, dirnames, filenames in os.walk(SHOW_DIR):
        for filename in filenames:
            tempFilename=filename.replace("show-","")
            if tempFilename in trainImgFilenameSet:
                shutil.copy(os.path.join(filepath, filename), showTrainDir)
            elif tempFilename in valImgFilenameSet:
                shutil.copy(os.path.join(filepath, filename), showValDir)
            elif tempFilename in testImgFilenameSet:
                shutil.copy(os.path.join(filepath, filename), showTestDir)

def addImg(unallocatedList,locatedList,locatedFanSet): # 不需要返回，因为函数里List、Set的改变会影响函数外的List、Set。
    flag = True
    while flag:
        flag = False
        for tempList in unallocatedList:
            tempFanList = tempList[1]
            if (len(set(tempFanList) & locatedFanSet) > 0):
                if (len(set(tempFanList) - locatedFanSet) > 0):
                    flag = True
                locatedFanSet.update(tempFanList)
                locatedList.append(tempList)
                unallocatedList.remove(tempList)


'''
剪切并粘贴文件，粘贴的地方如果有同名文件，就直接覆盖掉同名文件。相当于删除dstPath处的该文件，再把srcPath粘贴过去。
'''
from shutil import move,copy
def copyAndPasteFile(srcPathSet,dstDir):
    for srcPath in srcPathSet:
        copy(srcPath, dstDir)

def splitBackgroundTrainValTest(ROOT_DIR,backgroundDir):
    IMAGE_DIR = os.path.join(backgroundDir, "Img/png")
    backgroundImgPathMap={}

    for filepath, dirnames, filenames in os.walk(IMAGE_DIR):
        for fileName in filenames:
            prefixName=fileName.split("_")[0]
            if backgroundImgPathMap.get(prefixName)!=None:
                backgroundImgPathMap.get(prefixName).append(os.path.join(filepath,fileName))
            else:
                tempList=[os.path.join(filepath,fileName)]
                backgroundImgPathMap[prefixName]=tempList

    trainPathSet=set()
    valPathSet=set()
    testPathSet=set()
    for backgroundImgPathList in backgroundImgPathMap.values():
        random.shuffle(backgroundImgPathList)
        trainPathSet.update(backgroundImgPathList[0:int(len(backgroundImgPathList)*0.6)])
        valPathSet.update(backgroundImgPathList[int(len(backgroundImgPathList)*0.6):int(len(backgroundImgPathList)*0.8)])
        testPathSet.update(backgroundImgPathList[int(len(backgroundImgPathList)*0.8):])

    print("Img截图总数: ", len(trainPathSet)+ len(valPathSet)+ len(testPathSet))
    print("训练集、验证集、测试集数量: ", len(trainPathSet), len(valPathSet), len(testPathSet))

    resultPath = os.path.join(ROOT_DIR, "result")
    trainPath = os.path.join(resultPath, "train")
    trainImgPath = os.path.join(trainPath, "Img")
    valPath = os.path.join(resultPath, "val")
    valImgPath = os.path.join(valPath, "Img")
    testPath = os.path.join(resultPath, "test")
    testImgPath = os.path.join(testPath, "Img")
    makedir(trainImgPath)
    makedir(valImgPath)
    makedir(testImgPath)

    copyAndPasteFile(trainPathSet,trainImgPath)
    copyAndPasteFile(valPathSet,valImgPath)
    copyAndPasteFile(valPathSet,valImgPath)

if __name__ == '__main__':

    # 步骤1：使用shapefile文件和raster，来裁剪1280×1280的图像。
    inputDir=r"G:\graduation_project(mars_fan)\CTX\run_code_data\testCenterClip6(keypoints)\input"
    rasterPathList = glob.glob(inputDir + '\\' + "*.vrt")
    shapefilePath = glob.glob(inputDir + '\\' + "*.shp")[0]
    outputDir=r"G:\graduation_project(mars_fan)\CTX\run_code_data\testCenterClip6(keypoints)\output\my"
    makedir(outputDir)
    log_path=outputDir + '/logs/'
    defineLog(log_path)
    random.seed(10)

    clipSize=1280

    # 模糊扇FID
    dimFan={195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,537,800,801,1120,1124,1127,1287,1483,1486,1487,1495}

    print("inputDir:",inputDir)
    print("rasterPathList:",rasterPathList)
    print("shapefilePath:",shapefilePath)
    print("clipSize:",clipSize)

    print("Step 2: ClipRasterAndShapefile:", str(datetime.timedelta(seconds=time.time() - time_start)))
    testCenterClip(shapefilePath,rasterPathList,outputDir,clipSize,dimFan)
    tif2png(outputDir)

    print("Step 3: splitTrainValTest:",str(datetime.timedelta(seconds=time.time() - time_start)))
    splitTrainValTest(outputDir)

    # print("Step 4: clipBackground:", str(datetime.timedelta(seconds=time.time() - time_start)))
    # files = os.listdir(outputDir+r"\Img\tif")   # 读入文件夹
    # backgroundCount=int(len(files)*0.05)
    # del files
    #
    # backgroundDir=outputDir+"/background/"
    # makedir(backgroundDir)
    # clipBackground(rasterPathList,backgroundDir,shapefilePath,backgroundCount,clipSize)
    # tif2png(backgroundDir)
    # splitBackgroundTrainValTest(outputDir,backgroundDir)

    print("结束:", str(datetime.timedelta(seconds=time.time() - time_start)))
