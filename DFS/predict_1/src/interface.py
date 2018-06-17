import os
from dynamic_predict import dynamic_predict

#0表示下载，1表示上传
def getFilename(command):
    cmd = command.split()
    if cmd[2] == "-get":
        return (0,cmd[3])
    elif cmd[2] == "-put":
        return (1,cmd[3])
    else: return (-1,None)

def addToMap(name):
    length = len(name_id_map)
    name_id_map[name] = length + 1
    id_name_map[length + 1] = name

def nameToId(name):
    return name_id_map[name]

def idToName(id):
    return id_name_map(id)

def check_here(name):
    pass

def check_hdfs(name):
    pass

name_id_map = {}
id_name_map = {}
path = "/home/linan/hadoop-2.7.6/bin/"
tmp_here = "/home/linan/tmp/"
tmp_hdfs = "/tmp/"
while True:
    command = input()
    if command == "exit":
        break
    elif command[0:6] == "hadoop":              #是hadoop相关的命令
        command = path + command
        filename = getFilename(command)         #如果是上传或者下载命令，获取源文件路径

        if filename[0] == 0:                    #下载命令，检查本地临时文件夹是否已有该文件
            if check_here(filename[1]) == True:
                continue
        if filename[0] == 1:                    #上传命令，检查HDFS临时文件夹是否已有该命令
            if check_hdfs(filename[1]) == True:
                continue

        print(os.popen(command).read())         #执行该命令，并打印出结果
        if filename[0] == -1:
            continue
        else:                                   #下面是预测部分
            addToMap(filename[1])               # 将当前文件路径加到map中
            id_in = nameToId(filename[1])       #当前文件的id
            id = dynamic_predict(id_in)         #预测得到要预取或预存的文件的id
            if id == 0:                         #预测失败
                continue
            else:
                name = idToName(id)             #要预取或预存的文件名
                if filename[0] == 0:            #预取或预存到临时文件夹
                    cmd = path + "hadoop fs -get " + filename[1] + " " + tmp_here
                    os.popen(cmd)
                else:
                    cmd = path + "hadoop fs -put " + filename[1] + " " + tmp_hdfs
                    os.popen(cmd)
    else:
        continue
