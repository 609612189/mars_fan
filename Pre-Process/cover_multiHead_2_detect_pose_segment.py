import os

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def list2string(list):
    result=""
    for temp in list:
        result = result + temp + " "
    result = result.rstrip(' ') + "\n"
    return result

def createTXT(inputDir,outputDir):
    for filepath, dirnames, filenames in os.walk(inputDir):
        for filename in filenames:
            if os.path.splitext(filename)[1]=='.txt':
                inputTXT=os.path.join(filepath, filename)

                result=""
                with open(inputTXT,'r') as f:  #打开一个文件只读模式
                    line = f.readlines()
                    for line_list in line:
                        line_new =line_list.replace('\n','')  #将换行符替换为空('')
                        line_new_list = line_new.split(' ')
                        # result=result+list2string(line_new_list[0:7]) # pose(no visible)
                        # result=result+list2string(line_new_list[0:8]) # pose(visible)
                        # result = result +"0 "+ list2string(line_new_list[8:]) # segment
                        result = result + list2string(line_new_list[0:5])  # detect

                outputTXT=os.path.join(outputDir, filename)
                ff = open(outputTXT, 'w')
                ff.write(result)
                ff.close()

def multi2pose(inputDir,outputDir):
    input_labelTrainDir = os.path.join(inputDir, "train")
    input_labelValDir = os.path.join(inputDir, "val")
    input_labelTestDir = os.path.join(inputDir, "test")

    output_labelTrainDir = os.path.join(outputDir, "train")
    output_labelValDir = os.path.join(outputDir, "val")
    output_labelTestDir = os.path.join(outputDir, "test")

    makedir(output_labelTrainDir)
    makedir(output_labelValDir)
    makedir(output_labelTestDir)

    createTXT(input_labelTrainDir,output_labelTrainDir)
    createTXT(input_labelValDir,output_labelValDir)
    createTXT(input_labelTestDir,output_labelTestDir)

if __name__ == '__main__':

    Dir=r"D:\Desktop\graduation_project(mars_fan)\dataset\20240425-2\testCenterClip6(keypoints)_3-25mpx-1280"
    inputDir=os.path.join(Dir,"multi\labels")
    outputDir=os.path.join(Dir,"detect\labels")
    makedir(outputDir)
    multi2pose(inputDir,outputDir)