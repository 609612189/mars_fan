'''
构建vrt图片。有了vrt图片，就可以把多个马赛克图片，当作一整张合并后的图片来使用。
'''
import time
from osgeo import gdal,gdalconst,ogr
import datetime
import os


def test_gdalbuildvrt_time(Dir,vrtPath):
    in_files=[]
    for filepath, dirnames, filenames in os.walk(Dir):
        for fileName in filenames:
            if os.path.splitext(fileName)[1]=='.tif':
                temp = fileName.replace("25-MurrayLab_CTX_V01_", "")
                temp = temp.replace("_Mosaic.tif", "")
                tempList = temp.split("_")
                longitude = int(tempList[0].lstrip("E"))
                latitude = int(tempList[1].lstrip("N"))
                if latitude <= 50 and latitude >= -50:
                    in_files.append(os.path.join(filepath, fileName))

    start_time = time.time()

    vrt_options = gdal.BuildVRTOptions() # resampleAlg='cubic', addAlpha=True
    rastername = gdal.BuildVRT(vrtPath, in_files, options=vrt_options)

    end_time = time.time()
    print(str(datetime.timedelta(seconds=end_time - start_time)))


if __name__ == '__main__':
    outputDir=r"G:\graduation_project(mars_fan)\CTX\vrt//"
    Dir=r"G:\graduation_project(mars_fan)\CTX\allCTX\5-meter_per_pixel"
    vrtName="5-meter_per_pixel.vrt"
    vrtPath=outputDir+vrtName
    test_gdalbuildvrt_time(Dir,vrtPath)