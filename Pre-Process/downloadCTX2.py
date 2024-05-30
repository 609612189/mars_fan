'''
这个文件的代码仅供参考，因为太久远了，有一堆代码文件，不确定是不是这个文件，懒的整理了再跑一遍了。
downloadCTX()：代码下载火星CTX图片，为一个个5m/px的马赛克，地址：https://murray-lab.caltech.edu/CTX/。人工在网页中点开一个个链接下载也可以。
getDownloadCTXMapImg1()：打印个0/1图，表示地图上，对应位置的CTX马赛克是否下载。
downloadAllCTX()：知道哪些CTX没被下载。
'''
import shapefile
import wget
import ssl
import time
import os
from shutil import copy #shutil 是用来复制粘贴文件的

'''
CTX马赛克44行90列，44×90=3960。范围：[N-64，N44]外有16行，(44-16)×90=2520。范围内：28行90列。
下面2行是地图4角的CTX马赛克名。
E-180N84    E176N84
E-180N-88   E176N-88

列：
-180 0
-176 1
-172 2
...
176 89

行：
84 0
80 1
76 2
...
-88 43
'''
import numpy as np
def getDownloadCTXMapImg1(pathList):
    array = np.zeros((44, 90))
    np.set_printoptions(threshold=10000,linewidth=500) # threshold：触发汇总而不是完整重现的数组元素总数（默认 1000）。linewidth：用于插入换行符的每行字符数（默认 75）。

    for path in pathList:
        for filepath, dirnames, filenames in os.walk(path):
            for fileName in filenames:
                if os.path.splitext(fileName)[1]=='.tif':
                    if "MurrayLab_CTX_V01_" not in fileName:
                        print(os.path.join(filepath, fileName))
                        continue
                    tempList = fileName.split("_")
                    longitude = int(tempList[3].lstrip("E"))  # 经度
                    latitude = int(tempList[4].lstrip("N")) # 维度
                    row=int((latitude-84)/(-4))
                    column=int((longitude+180)/4)
                    array[row][column]=array[row][column]+1
    print(array)

'''
下载剩下没有下载的CTX的zip。争取下载火星的[N-60,N60]的CTX的zip。
'''
def downloadAllCTX(CTX_Shapefile_Path,pathList):
    start_time = time.time()

    QuadSet=getHaveDownloadedCTX(pathList)
    notDownloadCTXSet=set()

    file1 = shapefile.Reader(CTX_Shapefile_Path)
    #shapes = file.shapes()
    records1 = file1.records()
    for record1 in records1:
        Quad = record1["Quad"]
        tempList = Quad.split("_")
        longitude = int(tempList[0].lstrip("E"))
        latitude = int(tempList[1].lstrip("N"))
        if latitude<=60 and latitude>=-64:
            if Quad not in QuadSet:
                notDownloadCTXSet.add(Quad)
    end_time = time.time()
    print(end_time - start_time)
    return notDownloadCTXSet

'''
输入pathList路径，得到该路径下，所有.tif和.zip，返回已经下载的QuadSet。就是"E056N-48"这种格式的的一个Set。
'''
def getHaveDownloadedCTX(pathList):
    haveDownloadedCTXSet = set()

    for path in pathList:
        for filepath, dirnames, filenames in os.walk(path):
            for fileName in filenames:
                if os.path.splitext(fileName)[1]=='.tif' or os.path.splitext(fileName)[1]=='.zip':
                    if os.path.splitext(fileName)[1] == '.tif':
                        if "MurrayLab_CTX_V01_" not in fileName:
                            print("错误文件：", os.path.join(filepath, fileName))
                            continue
                        else:  # 名字格式：MurrayLab_CTX_V01_E-040_N-24_Mosaic.tif，MurrayLab_GlobalCTXMosaic_V01_E-080_N00.zip
                            temp = fileName.replace("MurrayLab_CTX_V01_", "")
                            temp = temp.replace("_Mosaic.tif", "")
                            haveDownloadedCTXSet.add(temp)
                    if os.path.splitext(fileName)[1]=='.zip':
                        if "MurrayLab_GlobalCTXMosaic_V01_" not in fileName:
                            print("错误文件：", os.path.join(filepath, fileName))
                            continue
                        else:  # 名字格式：MurrayLab_CTX_V01_E-040_N-24_Mosaic.tif，MurrayLab_GlobalCTXMosaic_V01_E-080_N00.zip
                            temp = fileName.replace("MurrayLab_GlobalCTXMosaic_V01_", "")
                            temp = temp.replace(".zip", "")
                            haveDownloadedCTXSet.add(temp)
    return haveDownloadedCTXSet

'''
输入有冲积扇的CTX列表Quadlist，得到有扇的CTX和扇上下左右周围的CTX，返回fanAndAroundCTXSet。就是"E056N-48"这种格式的的一个Set。
方向：
          N84
E-180               E176
          N-88
'''
def getFanAndAroundCTX(Quadlist):
    directions=[ # [E方向,N方向]
        [0,4], # 上
        [4,4], # 右上
        [4,0], # 右
        [4,-4], # 右下
        [0,-4], # 下
        [-4,-4], # 左下
        [-4,0], # 左
        [-4,4] # 左上
    ]
    fanAndAroundCTXSet=set()
    for Quad in Quadlist:
        tempList = Quad.split("_")
        longitude = int(tempList[0].lstrip("E"))
        latitude = int(tempList[1].lstrip("N"))
        for direction in directions:
            tempLongitude=longitude+direction[0]
            tempLatitude=latitude+direction[1]
            if tempLongitude<-180 or tempLongitude>176 or tempLatitude>44:
                continue

            tempLongitudeString=str(tempLongitude)
            tempLatitudeString=str(tempLatitude)
            if "-" in str(tempLongitude):
                if len(str(tempLongitude).replace("-",""))==1:
                    tempLongitudeString="-00"+str(tempLongitude).replace("-","")
                elif len(str(tempLongitude).replace("-",""))==2:
                    tempLongitudeString="-0"+str(tempLongitude).replace("-","")
            elif "-" not in str(tempLongitude):
                if len(str(tempLongitude))==1:
                    tempLongitudeString="00"+str(tempLongitude)
                elif len(str(tempLongitude))==2:
                    tempLongitudeString="0"+str(tempLongitude)

            if "-" in str(tempLatitude):
                if len(str(tempLatitude).replace("-",""))==1:
                    tempLatitudeString="-0"+str(tempLatitude).replace("-","")
            elif "-" not in str(tempLatitude):
                if len(str(tempLatitude))==1:
                    tempLatitudeString="0"+str(tempLatitude)

            fanAndAroundCTXSet.add("E"+tempLongitudeString+"_N"+tempLatitudeString)
    fanAndAroundCTXSet.update(Quadlist)
    return fanAndAroundCTXSet

'''
得到没有下载的，有扇的CTX和扇上下左右周围的CTX。就是"E056N-48"这种格式的一个Set。
最终目的：如果没有下载的话，优先下载它们。
'''
def get_Undownloaded_FanAndAroundCTX(Quadlist,pathList):
    fanAndAroundCTXSet=getFanAndAroundCTX(Quadlist)
    haveDownloadedCTXSet=getHaveDownloadedCTX(pathList)
    undownloaded_FanAndAroundCTX_Set=fanAndAroundCTXSet-haveDownloadedCTXSet
    return undownloaded_FanAndAroundCTX_Set

'''
把扇的、扇周围的ctx输出到一个map上，比如扇是黑方块，扇周围是1，其他是0；确保在下载的文件，是扇和扇周围的CTX。
CTX马赛克44行90列，44×90=3960。范围：[N-64，N44]外有16行，(44-16)×90=2520。范围内：28行90列。
下面2行是地图4角的CTX马赛克名。
E-180N84    E176N84
E-180N-88   E176N-88

列：
-180 0
-176 1
-172 2
...
176 89

行：
84 0
80 1
76 2
...
-88 43
'''
import numpy as np
def getDownloadCTXMapImg(fanCTXSet,fanAroundCTXSet):
    array = np.zeros((44, 90))
    np.set_printoptions(threshold=10000,linewidth=500) # threshold：触发汇总而不是完整重现的数组元素总数（默认 1000）。linewidth：用于插入换行符的每行字符数（默认 75）。

    for Quad in fanCTXSet:
        tempList = Quad.split("_")
        longitude = int(tempList[0].lstrip("E"))  # 经度
        latitude = int(tempList[1].lstrip("N")) # 维度
        row=int((latitude-84)/(-4))
        column=int((longitude+180)/4)
        array[row][column]=8
    for Quad in fanAroundCTXSet:
        tempList = Quad.split("_")
        longitude = int(tempList[0].lstrip("E"))  # 经度
        latitude = int(tempList[1].lstrip("N")) # 维度
        row=int((latitude-84)/(-4))
        column=int((longitude+180)/4)
        array[row][column]=1
    print(array)

'''
下载Quadlist列出的CTX马赛克，即包含火星冲积扇的CTX马赛克，为一个个.zip的文件。
数据格式：http://murray-lab.caltech.edu/CTX/V01/tiles/MurrayLab_GlobalCTXMosaic_V01_E-036_N-28.zip
https://murray-lab.caltech.edu/CTX/ 网站有写：一次仅下载一个图块。就是说，无法使用多进程、多线程来同时下载多个图块。
'''
def downloadCTX(Quadlist,outputPath):
    start_time = time.time()
    for Quad in Quadlist:
        zipURL="http://murray-lab.caltech.edu/CTX/V01/tiles/MurrayLab_GlobalCTXMosaic_V01_"+Quad+".zip"
        ssl._create_default_https_context = ssl._create_unverified_context
        path= outputPath+"/" + zipURL.split("/")[-1]
        # print(path) # D:\Desktop\MurrayLab_GlobalCTXMosaic_V01_E-180_N84.zip
        wget.download(zipURL, path)
        print(Quad)
    end_time = time.time()
    print(end_time - start_time)

if __name__ == '__main__':

    # 步骤13：得到没有下载的，有扇的CTX和扇上下左右周围的CTX。
    # Quadlist=['E-180_N24', 'E-180_N20', 'E-180_N12', 'E-180_N-12', 'E-180_N-16', 'E-180_N-20', 'E-180_N-36', 'E-180_N-40', 'E-180_N-44', 'E-180_N-52', 'E-176_N-12', 'E-176_N-36', 'E-172_N16', 'E-172_N-20', 'E-168_N00', 'E-168_N-04', 'E-168_N-12', 'E-168_N-16', 'E-164_N-16', 'E-164_N-32', 'E-160_N-12', 'E-160_N-32', 'E-160_N-36', 'E-160_N-40', 'E-156_N-16', 'E-156_N-36', 'E-156_N-44', 'E-152_N-08', 'E-152_N-44', 'E-152_N-56', 'E-148_N-08', 'E-148_N-12', 'E-148_N-16', 'E-128_N24', 'E-124_N-28', 'E-116_N-40', 'E-116_N-44', 'E-116_N-52', 'E-104_N-40', 'E-100_N24', 'E-092_N-44', 'E-088_N-36', 'E-084_N36', 'E-084_N-08', 'E-080_N-08', 'E-080_N-12', 'E-076_N36', 'E-072_N20', 'E-072_N04', 'E-072_N-08', 'E-072_N-16', 'E-072_N-28', 'E-072_N-40', 'E-072_N-44', 'E-068_N32', 'E-068_N20', 'E-068_N-16', 'E-064_N24', 'E-064_N20', 'E-064_N12', 'E-064_N08', 'E-064_N-08', 'E-064_N-12', 'E-064_N-16', 'E-060_N32', 'E-060_N08', 'E-060_N04', 'E-060_N-08', 'E-060_N-16', 'E-056_N44', 'E-056_N36', 'E-056_N32', 'E-056_N16', 'E-056_N12', 'E-056_N08', 'E-056_N04', 'E-056_N00', 'E-056_N-04', 'E-056_N-12', 'E-056_N-16', 'E-052_N12', 'E-052_N08', 'E-052_N04', 'E-052_N00', 'E-052_N-04', 'E-052_N-08', 'E-052_N-12', 'E-052_N-16', 'E-052_N-20', 'E-052_N-32', 'E-048_N08', 'E-048_N04', 'E-048_N00', 'E-048_N-04', 'E-048_N-08', 'E-048_N-12', 'E-048_N-24', 'E-048_N-28', 'E-048_N-32', 'E-044_N04', 'E-044_N00', 'E-044_N-04', 'E-044_N-08', 'E-044_N-12', 'E-044_N-16', 'E-040_N04', 'E-040_N00', 'E-040_N-04', 'E-040_N-08', 'E-040_N-12', 'E-040_N-16', 'E-040_N-20', 'E-040_N-24', 'E-040_N-28', 'E-040_N-32', 'E-036_N08', 'E-036_N04', 'E-036_N-04', 'E-036_N-08', 'E-036_N-12', 'E-036_N-16', 'E-036_N-20', 'E-036_N-24', 'E-036_N-28', 'E-032_N36', 'E-032_N08', 'E-032_N-12', 'E-032_N-16', 'E-032_N-28', 'E-032_N-32', 'E-032_N-44', 'E-028_N12', 'E-028_N04', 'E-028_N-28', 'E-028_N-32', 'E-028_N-44', 'E-024_N24', 'E-024_N20', 'E-024_N16', 'E-024_N-16', 'E-024_N-20', 'E-024_N-24', 'E-024_N-28', 'E-024_N-64', 'E-020_N-20', 'E-020_N-28', 'E-020_N-44', 'E-020_N-48', 'E-016_N32', 'E-016_N28', 'E-016_N20', 'E-016_N08', 'E-016_N-12', 'E-016_N-16', 'E-016_N-24', 'E-016_N-28', 'E-012_N32', 'E-012_N24', 'E-012_N20', 'E-012_N-08', 'E-008_N-08', 'E-008_N-12', 'E-008_N-20', 'E-008_N-28', 'E-008_N-32', 'E-004_N36', 'E-004_N32', 'E-004_N28', 'E-004_N-08', 'E-004_N-12', 'E-004_N-40', 'E000_N36', 'E000_N32', 'E000_N28', 'E000_N-16', 'E000_N-24', 'E000_N-28', 'E004_N36', 'E004_N32', 'E004_N28', 'E004_N24', 'E004_N-28', 'E004_N-32', 'E008_N36', 'E008_N28', 'E008_N24', 'E008_N16', 'E008_N12', 'E008_N-20', 'E008_N-24', 'E012_N28', 'E012_N-12', 'E012_N-20', 'E012_N-24', 'E016_N32', 'E016_N28', 'E016_N-24', 'E016_N-48', 'E020_N32', 'E020_N28', 'E020_N24', 'E020_N-40', 'E024_N32', 'E024_N28', 'E024_N-20', 'E024_N-24', 'E024_N-28', 'E028_N-24', 'E028_N-28', 'E028_N-32', 'E028_N-36', 'E032_N32', 'E032_N04', 'E032_N-20', 'E032_N-32', 'E036_N36', 'E036_N20', 'E036_N04', 'E036_N-08', 'E036_N-12', 'E036_N-16', 'E036_N-20', 'E036_N-24', 'E040_N12', 'E040_N04', 'E040_N-20', 'E044_N28', 'E052_N36', 'E056_N36', 'E056_N28', 'E056_N24', 'E056_N00', 'E056_N-04', 'E056_N-24', 'E060_N32', 'E060_N-20', 'E064_N28', 'E064_N20', 'E064_N-16', 'E064_N-24', 'E068_N28', 'E068_N-20', 'E068_N-24', 'E068_N-28', 'E072_N24', 'E072_N20', 'E072_N-24', 'E072_N-28', 'E072_N-32', 'E076_N24', 'E076_N20', 'E076_N16', 'E076_N-16', 'E076_N-24', 'E076_N-32', 'E080_N-20', 'E080_N-28', 'E080_N-32', 'E080_N-36', 'E084_N24', 'E084_N00', 'E084_N-20', 'E084_N-24', 'E084_N-32', 'E084_N-36', 'E088_N00', 'E088_N-16', 'E088_N-28', 'E092_N04', 'E092_N-28', 'E096_N-28', 'E100_N08', 'E100_N04', 'E100_N-36', 'E104_N04', 'E104_N00', 'E108_N00', 'E112_N00', 'E112_N-36', 'E112_N-40', 'E116_N08', 'E116_N04', 'E116_N00', 'E116_N-04', 'E116_N-08', 'E120_N00', 'E120_N-20', 'E120_N-24', 'E124_N00', 'E124_N-04', 'E124_N-32', 'E124_N-36', 'E128_N00', 'E128_N-04', 'E128_N-16', 'E128_N-20', 'E128_N-36', 'E128_N-40', 'E132_N00', 'E132_N-04', 'E132_N-08', 'E132_N-36', 'E136_N-08', 'E136_N-12', 'E140_N12', 'E140_N08', 'E140_N-04', 'E140_N-08', 'E140_N-24', 'E140_N-32', 'E140_N-36', 'E140_N-40', 'E140_N-44', 'E144_N00', 'E144_N-04', 'E144_N-08', 'E144_N-12', 'E148_N-08', 'E148_N-12', 'E148_N-16', 'E148_N-28', 'E152_N-08', 'E152_N-12', 'E152_N-36', 'E156_N08', 'E156_N-16', 'E156_N-40', 'E160_N28', 'E160_N24', 'E164_N-08', 'E164_N-12', 'E164_N-28', 'E164_N-40', 'E168_N-28', 'E168_N-32', 'E168_N-48', 'E172_N20', 'E172_N-16', 'E172_N-32', 'E176_N20', 'E176_N-16', 'E176_N-32', 'E176_N-44']
    # pathList=[r"G:\new\graduation_project(mars_fan)\CTX"]
    # undownloaded_FanAndAroundCTX_Set=get_Undownloaded_FanAndAroundCTX(Quadlist,pathList)
    # print(undownloaded_FanAndAroundCTX_Set)

    # 步骤14：把扇的、扇周围的ctx输出到一个map上，比如扇是黑方块，扇周围是1，其他是0；确保在下载的文件，是扇和扇周围的CTX。
    # fanCTXSet={'E-180_N24', 'E-180_N20', 'E-180_N12', 'E-180_N-12', 'E-180_N-16', 'E-180_N-20', 'E-180_N-36', 'E-180_N-40', 'E-180_N-44', 'E-180_N-52', 'E-176_N-12', 'E-176_N-36', 'E-172_N16', 'E-172_N-20', 'E-168_N00', 'E-168_N-04', 'E-168_N-12', 'E-168_N-16', 'E-164_N-16', 'E-164_N-32', 'E-160_N-12', 'E-160_N-32', 'E-160_N-36', 'E-160_N-40', 'E-156_N-16', 'E-156_N-36', 'E-156_N-44', 'E-152_N-08', 'E-152_N-44', 'E-152_N-56', 'E-148_N-08', 'E-148_N-12', 'E-148_N-16', 'E-128_N24', 'E-124_N-28', 'E-116_N-40', 'E-116_N-44', 'E-116_N-52', 'E-104_N-40', 'E-100_N24', 'E-092_N-44', 'E-088_N-36', 'E-084_N36', 'E-084_N-08', 'E-080_N-08', 'E-080_N-12', 'E-076_N36', 'E-072_N20', 'E-072_N04', 'E-072_N-08', 'E-072_N-16', 'E-072_N-28', 'E-072_N-40', 'E-072_N-44', 'E-068_N32', 'E-068_N20', 'E-068_N-16', 'E-064_N24', 'E-064_N20', 'E-064_N12', 'E-064_N08', 'E-064_N-08', 'E-064_N-12', 'E-064_N-16', 'E-060_N32', 'E-060_N08', 'E-060_N04', 'E-060_N-08', 'E-060_N-16', 'E-056_N44', 'E-056_N36', 'E-056_N32', 'E-056_N16', 'E-056_N12', 'E-056_N08', 'E-056_N04', 'E-056_N00', 'E-056_N-04', 'E-056_N-12', 'E-056_N-16', 'E-052_N12', 'E-052_N08', 'E-052_N04', 'E-052_N00', 'E-052_N-04', 'E-052_N-08', 'E-052_N-12', 'E-052_N-16', 'E-052_N-20', 'E-052_N-32', 'E-048_N08', 'E-048_N04', 'E-048_N00', 'E-048_N-04', 'E-048_N-08', 'E-048_N-12', 'E-048_N-24', 'E-048_N-28', 'E-048_N-32', 'E-044_N04', 'E-044_N00', 'E-044_N-04', 'E-044_N-08', 'E-044_N-12', 'E-044_N-16', 'E-040_N04', 'E-040_N00', 'E-040_N-04', 'E-040_N-08', 'E-040_N-12', 'E-040_N-16', 'E-040_N-20', 'E-040_N-24', 'E-040_N-28', 'E-040_N-32', 'E-036_N08', 'E-036_N04', 'E-036_N-04', 'E-036_N-08', 'E-036_N-12', 'E-036_N-16', 'E-036_N-20', 'E-036_N-24', 'E-036_N-28', 'E-032_N36', 'E-032_N08', 'E-032_N-12', 'E-032_N-16', 'E-032_N-28', 'E-032_N-32', 'E-032_N-44', 'E-028_N12', 'E-028_N04', 'E-028_N-28', 'E-028_N-32', 'E-028_N-44', 'E-024_N24', 'E-024_N20', 'E-024_N16', 'E-024_N-16', 'E-024_N-20', 'E-024_N-24', 'E-024_N-28', 'E-024_N-64', 'E-020_N-20', 'E-020_N-28', 'E-020_N-44', 'E-020_N-48', 'E-016_N32', 'E-016_N28', 'E-016_N20', 'E-016_N08', 'E-016_N-12', 'E-016_N-16', 'E-016_N-24', 'E-016_N-28', 'E-012_N32', 'E-012_N24', 'E-012_N20', 'E-012_N-08', 'E-008_N-08', 'E-008_N-12', 'E-008_N-20', 'E-008_N-28', 'E-008_N-32', 'E-004_N36', 'E-004_N32', 'E-004_N28', 'E-004_N-08', 'E-004_N-12', 'E-004_N-40', 'E000_N36', 'E000_N32', 'E000_N28', 'E000_N-16', 'E000_N-24', 'E000_N-28', 'E004_N36', 'E004_N32', 'E004_N28', 'E004_N24', 'E004_N-28', 'E004_N-32', 'E008_N36', 'E008_N28', 'E008_N24', 'E008_N16', 'E008_N12', 'E008_N-20', 'E008_N-24', 'E012_N28', 'E012_N-12', 'E012_N-20', 'E012_N-24', 'E016_N32', 'E016_N28', 'E016_N-24', 'E016_N-48', 'E020_N32', 'E020_N28', 'E020_N24', 'E020_N-40', 'E024_N32', 'E024_N28', 'E024_N-20', 'E024_N-24', 'E024_N-28', 'E028_N-24', 'E028_N-28', 'E028_N-32', 'E028_N-36', 'E032_N32', 'E032_N04', 'E032_N-20', 'E032_N-32', 'E036_N36', 'E036_N20', 'E036_N04', 'E036_N-08', 'E036_N-12', 'E036_N-16', 'E036_N-20', 'E036_N-24', 'E040_N12', 'E040_N04', 'E040_N-20', 'E044_N28', 'E052_N36', 'E056_N36', 'E056_N28', 'E056_N24', 'E056_N00', 'E056_N-04', 'E056_N-24', 'E060_N32', 'E060_N-20', 'E064_N28', 'E064_N20', 'E064_N-16', 'E064_N-24', 'E068_N28', 'E068_N-20', 'E068_N-24', 'E068_N-28', 'E072_N24', 'E072_N20', 'E072_N-24', 'E072_N-28', 'E072_N-32', 'E076_N24', 'E076_N20', 'E076_N16', 'E076_N-16', 'E076_N-24', 'E076_N-32', 'E080_N-20', 'E080_N-28', 'E080_N-32', 'E080_N-36', 'E084_N24', 'E084_N00', 'E084_N-20', 'E084_N-24', 'E084_N-32', 'E084_N-36', 'E088_N00', 'E088_N-16', 'E088_N-28', 'E092_N04', 'E092_N-28', 'E096_N-28', 'E100_N08', 'E100_N04', 'E100_N-36', 'E104_N04', 'E104_N00', 'E108_N00', 'E112_N00', 'E112_N-36', 'E112_N-40', 'E116_N08', 'E116_N04', 'E116_N00', 'E116_N-04', 'E116_N-08', 'E120_N00', 'E120_N-20', 'E120_N-24', 'E124_N00', 'E124_N-04', 'E124_N-32', 'E124_N-36', 'E128_N00', 'E128_N-04', 'E128_N-16', 'E128_N-20', 'E128_N-36', 'E128_N-40', 'E132_N00', 'E132_N-04', 'E132_N-08', 'E132_N-36', 'E136_N-08', 'E136_N-12', 'E140_N12', 'E140_N08', 'E140_N-04', 'E140_N-08', 'E140_N-24', 'E140_N-32', 'E140_N-36', 'E140_N-40', 'E140_N-44', 'E144_N00', 'E144_N-04', 'E144_N-08', 'E144_N-12', 'E148_N-08', 'E148_N-12', 'E148_N-16', 'E148_N-28', 'E152_N-08', 'E152_N-12', 'E152_N-36', 'E156_N08', 'E156_N-16', 'E156_N-40', 'E160_N28', 'E160_N24', 'E164_N-08', 'E164_N-12', 'E164_N-28', 'E164_N-40', 'E168_N-28', 'E168_N-32', 'E168_N-48', 'E172_N20', 'E172_N-16', 'E172_N-32', 'E176_N20', 'E176_N-16', 'E176_N-32', 'E176_N-44'}
    # fanAndAroundCTXSet=getFanAndAroundCTX(fanCTXSet)
    # fanAroundCTXSet=fanAndAroundCTXSet-fanCTXSet
    # getDownloadCTXMapImg(fanCTXSet,fanAroundCTXSet) #扇的CTX确实被其他周围的要下载的CTX围着。

    # 步骤15
    # CTX_Shapefile_Path = '../run_code_data/downloadCTX/MurrayLab_GlobalCTXMosaic_V01_quad-map/MurrayLab_GlobalCTXMosaic_V01_quad-map.shp'
    # pathList=[r"G:\new\graduation_project(mars_fan)\CTX"]
    # notDownloadCTXSet=downloadAllCTX(CTX_Shapefile_Path,pathList)
    # QuadList=['E-020_N40', 'E028_N40', 'E-016_N-60', 'E-048_N-44', 'E-004_N00', 'E-036_N-64', 'E-048_N40', 'E-004_N08', 'E012_N-60', 'E040_N-36', 'E012_N04', 'E-004_N-52', 'E-056_N-44', 'E-076_N44', 'E-060_N-36', 'E-004_N12', 'E-032_N24', 'E028_N12', 'E052_N-12', 'E-012_N-56', 'E-004_N-56', 'E-032_N-56', 'E040_N-48', 'E048_N-60', 'E-072_N-56', 'E020_N-08', 'E-012_N-52', 'E036_N-64', 'E052_N08', 'E040_N-44', 'E-016_N-56', 'E-068_N-56', 'E-024_N44', 'E048_N-36', 'E-044_N24', 'E048_N-04', 'E048_N-16', 'E020_N-12', 'E-044_N-48', 'E008_N-56', 'E008_N-04', 'E032_N-44', 'E-036_N28', 'E020_N04', 'E-040_N-52', 'E008_N04', 'E-076_N28', 'E-064_N-52', 'E-060_N-56', 'E-036_N44', 'E-040_N16', 'E008_N-52', 'E-040_N20', 'E008_N-40', 'E044_N40', 'E-008_N00', 'E-032_N20', 'E020_N12', 'E-048_N28', 'E-048_N-60', 'E-020_N44', 'E024_N04', 'E-028_N44', 'E024_N40', 'E028_N-60', 'E-008_N44', 'E052_N44', 'E-020_N-04', 'E-008_N-56', 'E-060_N-60', 'E032_N-56', 'E-016_N44', 'E-076_N12', 'E032_N44', 'E052_N16', 'E052_N-16', 'E-008_N04', 'E-072_N-52', 'E-004_N-60', 'E-028_N-56', 'E024_N44', 'E048_N08', 'E012_N-40', 'E044_N36', 'E004_N-64', 'E-024_N40', 'E-032_N-64', 'E020_N16', 'E044_N-36', 'E-052_N-44', 'E024_N00', 'E028_N-04', 'E-076_N-64', 'E-036_N24', 'E024_N-08', 'E-064_N-40', 'E016_N-32', 'E-068_N-60', 'E-044_N36', 'E-036_N-56', 'E008_N-44', 'E048_N-32', 'E052_N-32', 'E-048_N20', 'E028_N-08', 'E-076_N-60', 'E044_N-56', 'E016_N-56', 'E020_N-04', 'E056_N44', 'E044_N-48', 'E-052_N-48', 'E028_N-12', 'E-020_N00', 'E-076_N-56', 'E044_N-60', 'E-072_N44', 'E016_N40', 'E004_N04', 'E044_N-04', 'E048_N-48', 'E-004_N44', 'E016_N04', 'E044_N-32', 'E044_N-40', 'E056_N-12', 'E012_N-36', 'E020_N-64', 'E-044_N32', 'E048_N-64', 'E036_N-40', 'E-064_N-64', 'E-040_N24', 'E044_N20', 'E036_N-60', 'E-044_N28', 'E-052_N-60', 'E028_N-44', 'E-012_N-64', 'E048_N20', 'E020_N44', 'E040_N-60', 'E048_N-12', 'E044_N-52', 'E-056_N-40', 'E-044_N-64', 'E-044_N-56', 'E056_N08', 'E-048_N36', 'E052_N-52', 'E012_N44', 'E036_N-52', 'E048_N00', 'E-056_N-60', 'E-024_N36', 'E008_N44', 'E-068_N40', 'E024_N-52', 'E008_N-60', 'E048_N-24', 'E012_N-64', 'E052_N-36', 'E-024_N32', 'E048_N-52', 'E-060_N-48', 'E-056_N24', 'E-016_N-64', 'E028_N-48', 'E-048_N-64', 'E044_N-12', 'E-044_N-60', 'E036_N-56', 'E028_N20', 'E012_N00', 'E-040_N-48', 'E016_N-64', 'E-008_N-64', 'E048_N04', 'E-036_N-60', 'E-044_N16', 'E-004_N04', 'E-040_N44', 'E-052_N24', 'E028_N-52', 'E036_N-44', 'E-068_N-64', 'E048_N-08']
    # notDownloadCTXSet=notDownloadCTXSet-set(QuadList)
    # print(notDownloadCTXSet)

    # 步骤16。打印个0/1图，表示地图上，对应位置的CTX马赛克是否下载。
    pathList=[r"G:\graduation_project(mars_fan)\CTX\allCTX\5-meter_per_pixel"]
    getDownloadCTXMapImg1(pathList)

    # 先下载：没有下载的，有扇的CTX和扇上下左右周围的CTX。
    QuadList=['E-176_N-52', 'E-048_N12'] # 这里QuadList只是格式举例。
    outputPath=r"G:\graduation_project(mars_fan)\CTX\zip"
    downloadCTX(QuadList,outputPath)