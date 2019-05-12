#coding:utf-8
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
dir="G:/python_projects/Tensorflow/License plate recognition/HyperLPR/hyperlpr-train/train_labels.txt"
dir1="loc_train.log"
dir2="./result_origin/train_log_loss.txt"

def dataRead(pathName):
    index = 1
    data=[]
    f=open(pathName,"r")
    lines=f.readlines()
    for line in lines:
        print(index)
        print(line)
        index += 1

        data.append(line)

    # return(data)
dataRead(dir1)