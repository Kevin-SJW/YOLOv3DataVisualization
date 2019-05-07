#coding=utf-8
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class Yolov3LogVisualization:

    def __init__(self,log_path,result_dir):

        self.log_path = log_path
        self.result_dir = result_dir

    def extract_log(self, save_log_path, key_word):
        with open(self.log_path, 'r') as f:
            with open(save_log_path, 'w') as train_log:
                next_skip = False
                for line in f:
                    if next_skip:
                        next_skip = False
                        continue
                    # 去除多gpu的同步log
                    if 'Syncing' in line:
                        continue
                    # 去除除零错误的log
                    if 'nan' in line:
                        continue
                    if 'Saving weights to' in line:
                        next_skip = True
                        continue
                    if key_word in line:
                        train_log.write(line)
        f.close()
        train_log.close()

    def parse_loss_log(self,log_path, line_num=13457):
        result = pd.read_csv(log_path, skiprows=[x for x in range(line_num)if ((x < 1000)) ],
                             error_bad_lines=False, names=['loss', 'avg', 'rate', 'seconds', 'images'])
        result['loss'] = result['loss'].str.split(' ').str.get(1)
        result['avg'] = result['avg'].str.split(' ').str.get(1)
        result['rate'] = result['rate'].str.split(' ').str.get(1)
        result['seconds'] = result['seconds'].str.split(' ').str.get(1)
        result['images'] = result['images'].str.split(' ').str.get(1)

        result['loss'] = pd.to_numeric(result['loss'])
        result['avg'] = pd.to_numeric(result['avg'])
        result['rate'] = pd.to_numeric(result['rate'])
        result['seconds'] = pd.to_numeric(result['seconds'])
        result['images'] = pd.to_numeric(result['images'])
        return result




    def parse_iou_log(self,log_path, line_num=283698):
        result = pd.read_csv(log_path, skiprows=[x for x in range(line_num) if x<4000],
                             error_bad_lines=False,names=['Region Avg IOU', 'Class', 'Obj', 'No Obj', 'Avg Recall', 'count'])
        result['Region Avg IOU'] = result['Region Avg IOU'].str.split(': ').str.get(1)
        result['Class'] = result['Class'].str.split(': ').str.get(1)
        result['Obj'] = result['Obj'].str.split(': ').str.get(1)
        result['No Obj'] = result['No Obj'].str.split(': ').str.get(1)
        result['Avg Recall'] = result['Avg Recall'].str.split(': ').str.get(1)
        result['count'] = result['count'].str.split(': ').str.get(1)

        result['Region Avg IOU'] = pd.to_numeric(result['Region Avg IOU'])
        result['Class'] = pd.to_numeric(result['Class'])
        result['Obj'] = pd.to_numeric(result['Obj'])
        result['No Obj'] = pd.to_numeric(result['No Obj'])
        result['Avg Recall'] = pd.to_numeric(result['Avg Recall'])
        result['count'] = pd.to_numeric(result['count'])
        return result

    def gene_loss_pic(self, pd_loss):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # ax.plot(pd_loss['avg'].values, label='avg_loss')
        length = len(pd_loss['avg'].values)
        x_ticks = np.arange(0, length, 1)
        ax.scatter(x_ticks, pd_loss['avg'].values[:length], s=0.05, label='avg_loss')

        print('loss length',length)
        # plt.grid()
        # x_ticks = [0, 50000, 100000, 150000, 200000, 250000]
        # plt.xticks(x_ticks)
        # y_ticks = [0, 2, 4, 6, 8, 10]
        # plt.yticks(y_ticks)

        ax.legend(loc='best')
        ax.set_title('The Loss Scatter Diagram')
        ax.set_xlabel('batches')
        fig.savefig(self.result_dir + '/avg_loss')
        logger.info('save loss pic done')


    def gene_iou_pic(self, pd_loss):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        print(pd_loss['Region Avg IOU'].values)
        l = len(pd_loss['Region Avg IOU'].values)
        print('length',l)
        x_ticks = np.arange(0,13000,1)
        ax.scatter(x_ticks,pd_loss['Region Avg IOU'].values[:13000],s=0.05, label='Region Avg IOU')
        # ax.plot(result['Class'].values,label='Class')
        # ax.plot(result['Obj'].values,label='Obj')
        # ax.plot(result['No Obj'].values,label='No Obj')
        # ax.plot(result['Avg Recall'].values,label='Avg Recall')
        # ax.plot(result['count'].values,label='count')
        # plt.grid()
        ax.legend(loc='best')
        ax.set_title('The Region Avg IOU Scatter Diagram')
        ax.set_xlabel('batches')
        fig.savefig(self.result_dir + '/region_avg_iou')
        logger.info('save iou pic done')
        # plt.gca()
        plt.xlim(0,13000)
        plt.ylim(0,1)

        # plt.show()

    def gene_class_pic(self, pd_loss):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # ax.plot(pd_loss['Class'].values, label='Class')
        x_ticks = np.arange(0, 13000, 1)
        ax.scatter(x_ticks, pd_loss['Class'].values[:13000], s=0.05, label='classification Acc')
        # plt.yticks(y_ticks)  # 如果不想自己设置纵坐标，可以注释掉。
        # x_ticks=[0,50000,100000,150000,200000,250000]
        # plt.xticks(x_ticks)
        # y_ticks=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        # plt.yticks(y_ticks)
        # plt.grid()
        ax.legend(loc='best')
        ax.set_title('The Confidence Scatter Diagram')
        ax.set_xlabel('batches')
        fig.savefig(self.result_dir + '/classification acc')
        logger.info('save class pic done')
        plt.xlim(0, 13000)
        plt.ylim(0, 1)

    def gene_recall_pic(self, pd_loss):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        print('Avg Recall',pd_loss['Avg Recall'].values)
        # l = len(pd_loss['Avg Recall'].values)
        x_ticks = np.arange(0, 13000, 1)
        ax.scatter(x_ticks, pd_loss['Avg Recall'].values[:13000], s=0.05, label='Avg Recall')
        # t = np.arange(0, l, 1)
        # ax.scatter(t,pd_loss['Avg Recall'].values, label='Avg Recall')
        #ax.plot(pd_loss['Avg Recall'].values, label='Avg Recall')
        # plt.yticks(y_ticks)  # 如果不想自己设置纵坐标，可以注释掉。
        # x_ticks=[0,50000,100000,150000,200000,250000]
        # plt.xticks(x_ticks)
        # y_ticks=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        # plt.yticks(y_ticks)
        # plt.grid()
        ax.legend(loc='best')
        ax.set_title('The Avg Recall Scatter Diagram')
        ax.set_xlabel('batches')
        fig.savefig(self.result_dir + '/Avg Recall')
        logger.info('save Avg Recall pic done')
        plt.xlim(0, 13000)
        plt.ylim(0, 1)

        plt.show()

    def loss_pic(self):
        train_log_loss_path = os.path.join(self.result_dir, 'train_log_loss.txt')
        # self.extract_log(train_log_loss_path, 'images')
        pd_loss = self.parse_loss_log(train_log_loss_path)
        self.gene_loss_pic(pd_loss)


    def iou_pic(self):
        train_log_iou_path = os.path.join(self.result_dir, 'train_log_iou.txt')
        # self.extract_log(train_log_loss_path, 'IOU')
        pd_loss = self.parse_iou_log(train_log_iou_path)
        self.gene_iou_pic(pd_loss)

    def class_pic(self):
        train_log_class_path = os.path.join(self.result_dir, 'train_log_iou.txt')
        # self.extract_log(train_log_loss_path, 'IOU')
        pd_loss = self.parse_iou_log(train_log_class_path)
        self.gene_class_pic(pd_loss)

    def recall_pic(self):
        train_log_recall_path = os.path.join(self.result_dir, 'train_log_iou.txt')
        # self.extract_log(train_log_loss_path, 'IOU')
        pd_loss = self.parse_iou_log(train_log_recall_path)
        self.gene_recall_pic(pd_loss)


if __name__ == '__main__':
    log_path = 'loc_train.log'
    result_dir = './result1'
    logVis = Yolov3LogVisualization(log_path,result_dir)
    # logVis.loss_pic()
    logVis.iou_pic()
    logVis.class_pic()
    logVis.recall_pic()
