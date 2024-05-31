'''
就是拿训练好的模型，在 用clipAllMars.py裁剪全火星CTX得到的25m/px、1280px的图片 上跑，得到全火星上，框、点的预测结果。
框用框的IoU来去重，点也根据框的IoU来去重。最后用得到的预测框、点的结果，生成框的shapefile文件、点的shapefile文件。
预测出来的的框、点的shapefile的坐标系，和https://www.sciencedirect.com/science/article/pii/S2352340922006886 下载的冲积扇的数据集完全一致。
'''

# 如果是用detect、pose、segment这些官方有的任务，可以直接用下面这个导包。
from ultralytics import YOLO


'''
multi模型，有些框会有，预测框没问题，预测框对应的预测点预测到整张图的最左上角点上，预测点不正确且预测点不在框内的问题。还没找到原因。
盲猜是点置信度太低，就没预测出点来，然后给点(x,y)填写了个默认值(0,0)。
pose模型，没遇到这个问题。大概因为pose的模型，就没预测点的置信度，因为我pose数据集那里，就(x,y,v)，我只有(x,y)，没写置信度。
'''
# # 如果是用multi任务，用下面这个导包。路径里是下面这个pr的代码：https://github.com/ultralytics/ultralytics/pull/5219。
# import sys
# sys.path.insert(0,r'D:\Desktop\graduation_project(mars_fan)\my_model\yolov8-related\temp37-multi-task-尝试改网络的各种模块。\ultralytics')
# from ultralytics import YOLO

import shapefile
import glob
import math
from osgeo import gdal, gdalconst, ogr
import warnings
import sys
import os
import time
import datetime
import torch

time_start = time.time()
warnings.filterwarnings("ignore")

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

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
    log_file_name = log_path +"/" + 'log-' + time.strftime("%Y-%m-%d#%H-%M-%S", time.localtime()) + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)

def clamp(number, min):
    if number < min:
        return min
    else:
        return number

def modelPredict(modelPath, imgDir, oldBoxShapefilePath,imgsz,confidence,iou): # ,oldPointShapefilePath,distanceIsOnePoint
    model = YOLO(modelPath)
    device = torch.device("cuda:0")
    results = model.predict(source=imgDir, save=False, imgsz=imgsz, conf=confidence, iou=iou, stream=True,
                            device=device)

    geoBoxList = []
    geoPointList=[]

    # 处理结果列表
    for result in results:
        # boxes = result.boxes  # 边界框输出的 Boxes 对象
        # probs = result.probs  # 分类输出的 Probs 对象
        # masks = result.masks  # 分割掩码输出的 Masks 对象
        # keypoints = result.keypoints  # 姿态输出的 Keypoints 对象

        path = result.path
        # 命名：空间分辨率pixelWidth_空间分辨率pixelHeight_左下角点x_左下角点y_右上角点x_右上角点y。
        tempList = os.path.splitext(os.path.basename(path))[0].split("_")
        tempList = [float(i) for i in tempList]
        pixelWidth = tempList[0]
        pixelHeight = tempList[1]
        geo_xmin = tempList[2]
        geo_ymin = tempList[3]
        geo_xmax = tempList[4]
        geo_ymax = tempList[5]

        boxes = result.boxes  # 边界框输出的 Boxes 对象
        keypoints = result.keypoints

        for xyxy in boxes.xyxy:
            if (len(xyxy) == 0):
                continue

            box_left_top_x = float(xyxy[0])  # 左上角x
            box_left_top_y = float(xyxy[1])  # 左上角y
            box_right_bottom_x = float(xyxy[2])  # 右下角x
            box_right_bottom_y = float(xyxy[3])  # 右下角y
            print("path:" + str(path) + "，非geo的，[左上角x，左上角y，右下角x，右下角y]：", [box_left_top_x,box_left_top_y,box_right_bottom_x,box_right_bottom_y])

            box_left_bottom_x = box_left_top_x
            box_left_bottom_y = box_right_bottom_y
            box_right_top_x = box_right_bottom_x
            box_right_top_y = box_left_top_y

            # 左下角，右下角，右上角，左上角。
            geo_box_left_bottom_x = geo_xmin + box_left_bottom_x * pixelWidth
            geo_box_left_bottom_y = geo_ymax - box_left_bottom_y * pixelHeight
            geo_box_right_bottom_x = geo_xmin + box_right_bottom_x * pixelWidth
            geo_box_right_bottom_y = geo_ymax - box_right_bottom_y * pixelHeight
            geo_box_right_top_x = geo_xmin + box_right_top_x * pixelWidth
            geo_box_right_top_y = geo_ymax - box_right_top_y * pixelHeight
            geo_box_left_top_x = geo_xmin + box_left_top_x * pixelWidth
            geo_box_left_top_y = geo_ymax - box_left_top_y * pixelHeight

            # [左下角x，左下角y，右下角x，右下角y，右上角x，右上角y，左上角x，左上角y]。
            nowBox = [geo_box_left_bottom_x, geo_box_left_bottom_y, geo_box_right_bottom_x, geo_box_right_bottom_y,
                      geo_box_right_top_x, geo_box_right_top_y, geo_box_left_top_x, geo_box_left_top_y]
            geoBoxList.append(nowBox)
            print("path:" + str(path) + "，geo的，[左下角x，左下角y，右下角x，右下角y，右上角x，右上角y，左上角x，左上角y]:", nowBox)

        if keypoints!=None:
            for xy in keypoints.xy:
                if(len(xy)==0):
                    continue
                point_x=float(xy[0][0])
                point_y=float(xy[0][1])
                print("path:"+str(path)+"，非geo的，point_x:"+str(point_x)+"，point_y:"+str(point_y))

                geo_point_x=geo_xmin+point_x*pixelWidth
                geo_point_y=geo_ymax-point_y*pixelHeight
                newPoint=[geo_point_x,geo_point_y]
                geoPointList.append(newPoint)
                print("path:"+str(path)+"，geo的，geo_point_x:"+str(geo_point_x)+"，geo_point_y:"+str(geo_point_y))

    print("去重前的geoBoxList:", geoBoxList)
    print("去重前的geoPointList:",geoPointList)

    # 新框和新框去重。点也对应去重。
    eps = 1e-7
    fid1 = 0
    while fid1 < len(geoBoxList):
        fid1_newBox_xmin = geoBoxList[fid1][0]
        fid1_newBox_ymin = geoBoxList[fid1][1]
        fid1_newBox_xmax = geoBoxList[fid1][4]
        fid1_newBox_ymax = geoBoxList[fid1][5]
        fid1_newBox_xRange = abs(fid1_newBox_xmax - fid1_newBox_xmin)
        fid1_newBox_yRange = abs(fid1_newBox_ymax - fid1_newBox_ymin)
        fid2 = fid1 + 1
        while fid2 < len(geoBoxList):
            fid2_newBox_xmin = geoBoxList[fid2][0]
            fid2_newBox_ymin = geoBoxList[fid2][1]
            fid2_newBox_xmax = geoBoxList[fid2][4]
            fid2_newBox_ymax = geoBoxList[fid2][5]
            fid2_newBox_xRange = abs(fid2_newBox_xmax - fid2_newBox_xmin)
            fid2_newBox_yRange = abs(fid2_newBox_ymax - fid2_newBox_ymin)

            intersection_xmin = max(fid1_newBox_xmin, fid2_newBox_xmin)
            intersection_ymin = max(fid1_newBox_ymin, fid2_newBox_ymin)
            intersection_xmax = min(fid1_newBox_xmax, fid2_newBox_xmax)
            intersection_ymax = min(fid1_newBox_ymax, fid2_newBox_ymax)
            intersection = clamp(intersection_xmax - intersection_xmin, 0) * clamp(
                intersection_ymax - intersection_ymin, 0)
            union = fid1_newBox_xRange * fid1_newBox_yRange + fid2_newBox_xRange * fid2_newBox_yRange - intersection + eps

            if intersection == 0 or intersection / union < iou:
                fid2 = fid2 + 1
            else:
                del geoBoxList[fid2]
                if len(geoPointList)-1>=fid2: # 防止不预测点的任务比如segment会出现报错："IndexError: list assignment index out of range"。
                    del geoPointList[fid2]

        fid1 = fid1 + 1

    # 新框和老框去重。点也对应去重。
    eps = 1e-7
    file1 = shapefile.Reader(oldBoxShapefilePath)
    shapes = file1.shapes()
    for shape in shapes:
        oldBox_xmin = shape.bbox[0]
        oldBox_ymin = shape.bbox[1]
        oldBox_xmax = shape.bbox[2]
        oldBox_ymax = shape.bbox[3]
        oldBox_xRange = abs(oldBox_xmax - oldBox_xmin)
        oldBox_yRange = abs(oldBox_ymax - oldBox_ymin)

        fid = 0
        while fid < len(geoBoxList):
            newBox_xmin = geoBoxList[fid][0]  # 左下角x
            newBox_ymin = geoBoxList[fid][1]  # 左下角y
            newBox_xmax = geoBoxList[fid][4]  # 右上角x
            newBox_ymax = geoBoxList[fid][5]  # 右上角y
            newBox_xRange = abs(newBox_xmax - newBox_xmin)
            newBox_yRange = abs(newBox_ymax - newBox_ymin)

            intersection_xmin = max(newBox_xmin, oldBox_xmin)
            intersection_ymin = max(newBox_ymin, oldBox_ymin)
            intersection_xmax = min(newBox_xmax, oldBox_xmax)
            intersection_ymax = min(oldBox_ymax, newBox_ymax)
            intersection = clamp(intersection_xmax - intersection_xmin, 0) * clamp(
                intersection_ymax - intersection_ymin, 0)
            union = oldBox_xRange * oldBox_yRange + newBox_xRange * newBox_yRange - intersection + eps

            if intersection == 0 or intersection / union < iou:
                fid = fid + 1
            else:
                del geoBoxList[fid]
                if len(geoPointList) - 1 >= fid:  # 防止不预测点的任务比如segment会出现报错："IndexError: list assignment index out of range"。
                    del geoPointList[fid]

    # # 下面是不根据框之间的IoU来去重点，而是直接根据点之间的距离来去重 点。
    # # 新点和新点去重。
    # fid1=0
    # while fid1<len(geoPointList):
    #     fid1_geo_point_x=geoPointList[fid1][0]
    #     fid1_geo_point_y=geoPointList[fid1][1]
    #     fid2=fid1+1
    #     while fid2<len(geoPointList):
    #         fid2_geo_point_x = geoPointList[fid2][0]
    #         fid2_geo_point_y = geoPointList[fid2][1]
    #         if(math.sqrt(math.pow((fid1_geo_point_x-fid2_geo_point_x),2)+math.pow((fid1_geo_point_y-fid2_geo_point_y),2))<distanceIsOnePoint):
    #             del geoPointList[fid2]
    #         else:
    #             fid2=fid2+1
    #     fid1=fid1+1
    #
    # # 新点和老点去重。
    # file1 = shapefile.Reader(oldPointShapefilePath)
    # shapes = file1.shapes()
    # for shape in shapes: # 每个shape里一个shape.points，格式是[[x,y]]
    #     oldPoint_x = shape.points[0][0]
    #     oldPoint_y = shape.points[0][1]
    #     fid=0
    #     while fid<len(geoPointList):
    #         newPoint_x=geoPointList[fid][0]
    #         newPoint_y=geoPointList[fid][1]
    #         if (math.sqrt(math.pow((oldPoint_x - newPoint_x), 2) + math.pow(
    #                 (oldPoint_y - newPoint_y), 2)) < distanceIsOnePoint):
    #             del geoPointList[fid]
    #         else:
    #             fid=fid+1

    return geoBoxList,geoPointList


def createBoxShapefile(boxShapefilePath, geoBoxList):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.CreateDataSource(boxShapefilePath)
    layer = ds.CreateLayer('box', geom_type=ogr.wkbPolygon)

    fieldDefn = ogr.FieldDefn('id_label', ogr.OFTInteger)
    fieldDefn.SetWidth(8)
    fieldDefn.SetPrecision(3)
    layer.CreateField(fieldDefn)

    featureDefn = layer.GetLayerDefn()
    feature = ogr.Feature(featureDefn)

    for i in range(0, len(geoBoxList)):
        outring = ogr.Geometry(ogr.wkbLinearRing)
        outring.AddPoint(geoBoxList[i][0], geoBoxList[i][1])
        outring.AddPoint(geoBoxList[i][2], geoBoxList[i][3])
        outring.AddPoint(geoBoxList[i][4], geoBoxList[i][5])
        outring.AddPoint(geoBoxList[i][6], geoBoxList[i][7])
        outring.AddPoint(geoBoxList[i][0], geoBoxList[i][1])
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(outring)

        feature.SetGeometry(polygon)
        feature.SetField('id_label', i)
        layer.CreateFeature(feature)

    ds.Destroy()

def createPointShapefile(pointShapefilePath,geoPointList):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.CreateDataSource(pointShapefilePath)
    layer = ds.CreateLayer('point', geom_type=ogr.wkbPoint)

    fieldDefn = ogr.FieldDefn('id_label', ogr.OFTInteger)
    fieldDefn.SetWidth(8)
    fieldDefn.SetPrecision(3)
    layer.CreateField(fieldDefn)

    featureDefn = layer.GetLayerDefn()
    feature = ogr.Feature(featureDefn)

    for i in range(0,len(geoPointList)):
        point = ogr.Geometry(ogr.wkbPoint)
        point.SetPoint(0, geoPointList[i][0], geoPointList[i][1])
        feature.SetGeometry(point)
        feature.SetField('id_label', i)
        layer.CreateFeature(feature)

    ds.Destroy()


if __name__ == '__main__':
    print(sys.path)
    # distanceIsOnePoint=6000 # 2个点距离多少米被认为是同1个点。
    iou = 0.05  # 2个框iou是多少，会被认为是同1个物体。
    confidence = 0.6
    imgsz = 1280

    outputDir = r"G:\graduation_project(mars_fan)\CTX\run_code_data\modelPredict_box_and_point\output"
    outputDir = outputDir + '/' + "imgsz" + str(imgsz) + "_confidence" + str(confidence) + "_iou" + str(iou) #+ "_distanceIsOnePoint"+str(distanceIsOnePoint)
    makedir(outputDir)
    defineLog(outputDir)

    modelPath = r"G:\graduation_project(mars_fan)\CTX\run_code_data\modelPredict_box_and_point\input\temp37-17的4、yolov8n-seg-C2f_DWRSeg-p7-1-C2f.yaml。box-mAP50=0.511。\best.pt"
    # modelPath=r"G:\graduation_project(mars_fan)\CTX\run_code_data\modelPredict\input\yolov8s-pose-mAP50box=0.239-mAP50pose=0.64-25mpx-1280\best.pt"
    # modelPath = r"G:\graduation_project(mars_fan)\CTX\run_code_data\modelPredict_box_and_point\input\temp37的5，yolov8n-multi-p7-1.yaml。数据：multi(1280-uniform_brightness_images)。box-mAP50=0.45，pose-mAP50=0.555。\best.pt"
    # modelPath = r"G:\graduation_project(mars_fan)\CTX\run_code_data\modelPredict_box_and_point\input\temp37-2的6、kobj=0，yolov8n-multi-p7-1.yaml。\best.pt"

    # imgDir=r"G:\graduation_project(mars_fan)\CTX\run_code_data\clipAllMars\output\Img\25mpx\png"
    imgDir = r"G:\graduation_project(mars_fan)\CTX\run_code_data\clipAllMars\output\Img\test_25mpx"

    oldBoxShapefilePath = r"G:\graduation_project(mars_fan)\CTX\run_code_data\modelPredict_box_and_point\input\Fan_Merge3.shp"
    # oldPointShapefilePath=r"G:\graduation_project(mars_fan)\CTX\run_code_data\modelPredict\input\Morgan2022_FanDatabase_apices.shp"


    print("modelPath:", modelPath)
    print("imgDir:", imgDir)
    print("oldBoxShapefilePath:", oldBoxShapefilePath)
    # print("oldPointShapefilePath:",oldPointShapefilePath)
    print("imgsz", imgsz)
    print("confidence:", confidence)
    print("iou:", iou)
    # print("distanceIsOnePoint:",distanceIsOnePoint)


    geoBoxList,geoPointList = modelPredict(modelPath, imgDir, oldBoxShapefilePath,imgsz,confidence,
                              iou) # ,oldPointShapefilePath,distanceIsOnePoint
    print("去重后的geoBoxList:", geoBoxList)
    print("去重后的geoPointList:",geoPointList)

    if len(geoBoxList) != 0:
        boxShapefileName = "newBox"
        boxShapefilePath = outputDir + "/" + boxShapefileName
        createBoxShapefile(boxShapefilePath, geoBoxList)

    if len(geoPointList)!=0:
        pointShapefileName="newPoint"
        pointShapefilePath=outputDir+"/" + pointShapefileName
        createPointShapefile(pointShapefilePath, geoPointList)

    print("结束:", str(datetime.timedelta(seconds=time.time() - time_start)))
