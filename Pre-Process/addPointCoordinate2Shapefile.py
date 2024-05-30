'''
给扇的语义分割的shapefile文件中，加入扇的关键点的信息。这样可以使用一个shapefile来裁剪出多种任务的数据，框、分割、关键点。
'''

import shapefile
from osgeo import gdal, gdalconst, ogr

'''
在Fan_Merge3中，给每个扇的属性表，加入扇中心点坐标。
'''
def addPointCoordinate2Shapefile(Fan_Merge3_Dir,apices_Dir):
    pointXList=[]
    pointYList=[]
    file1 = shapefile.Reader(Fan_Merge3_Dir)
    shapes1 = file1.shapes()
    records1 = file1.records()
    file2 = shapefile.Reader(apices_Dir)
    shapes2 = file2.shapes()
    records2 = file2.records()

    for record1 in records1: # Fan_Merge3
        fan_id1=record1["fan_id"]
        i=0
        for record2 in records2: # apices
            fan_id2=record2["fan_id"]
            if fan_id1==fan_id2:
                pointXList.append(shapes2[i].points[0][0])
                pointYList.append(shapes2[i].points[0][1])
                break
            i=i+1

    in_ds = ogr.Open(Fan_Merge3_Dir, True)  # False - read only, True - read/write
    in_layer = in_ds.GetLayer(0)  # 这里的in_ds就是.shp文件，相当于in_ds.GetLayerByIndex(0)。总共就1个Layer，Layer里存了5个shp面。
    in_lydefn = in_layer.GetLayerDefn()  # GetLayerDefn得到关于Layer的一些架构信息，比如有几个字段，字段名是什么。

    point_x = 'point_x'
    point_y = 'point_y'
    # print(dir(in_lydefn))
    # for i in range(in_lydefn.GetFieldCount()):
    #     if in_lydefn.GetFieldDefn(i).GetName() == name:
    #         return

    field1 = ogr.FieldDefn(point_x, ogr.OFTReal)  # ogr.OFTReal是双精度浮点型。
    field1.SetWidth(32)  # 设置长度
    field1.SetPrecision(7)  # 设置小数点位数
    in_layer.CreateField(field1)
    field2 = ogr.FieldDefn(point_y, ogr.OFTReal)
    field2.SetWidth(32)  # 设置长度
    field2.SetPrecision(7)  # 设置小数点位数
    in_layer.CreateField(field2)


    for i, feature in enumerate(in_layer):
        feature.SetField(point_x, pointXList[i])
        in_layer.SetFeature(feature)
        feature.SetField(point_y, pointYList[i])
        in_layer.SetFeature(feature)
        # print(i, ":set is ok ")
    del in_ds


if __name__ == '__main__':
    Fan_Merge3_Dir=r"G:\graduation_project(mars_fan)\CTX\run_code_data\addPointCoordinate2Shapefile\input\Fan_Merge3.shp"
    apices_Dir=r"G:\graduation_project(mars_fan)\CTX\run_code_data\addPointCoordinate2Shapefile\input\Morgan2022_FanDatabase_apices.shp"
    addPointCoordinate2Shapefile(Fan_Merge3_Dir,apices_Dir)