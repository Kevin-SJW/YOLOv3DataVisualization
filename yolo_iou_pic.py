#coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


log_path='iou_data_separated.csv'
result_dir = './result'
line_num=283623

def gen_iou_pic(log_path,line_num,result_dir):
    result = pd.read_csv(log_path,skiprows=[x for x in range(line_num) if ((x % 10 != 9) | (x < 1500))],
                         error_bad_lines=False,
                         names=['Region Avg IOU', 'Class', 'Obj', 'No Obj', 'Avg Recall_1', 'Avg Recall_2', 'count'])
    print(result)
    # print("result['Region Avg IOU'] ", result['Region Avg IOU'])
    result['Region Avg IOU'] = pd.to_numeric(result['Region Avg IOU'])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    print('avg iou', result['Region Avg IOU'].values)
    length = len(result['Region Avg IOU'].values)
    print('avg iou length:', length)
    x_ticks = np.arange(0, 13000, 1)

    ax.scatter(x_ticks, result['Region Avg IOU'].values[:13000], s=1, label='Region Avg IOU')
    # ax.plot(result['Class'].values,label='Class')
    # ax.plot(result['Obj'].values,label='Obj')
    # ax.plot(result['No Obj'].values,label='No Obj')
    # ax.plot(result['Avg Recall'].values,label='Avg Recall')
    # ax.plot(result['count'].values,label='count')
    # plt.grid()
    ax.legend(loc='best')
    ax.set_title('The Region Avg IOU Scatter Diagram')
    ax.set_xlabel('batches')
    fig.savefig(result_dir + '/region_avg_iou')
    logger.info('save iou pic done')
    # plt.gca()
    plt.xlim(0, 13000)
    plt.ylim(0, 1)
    plt.show()

gen_iou_pic(log_path,line_num,result_dir)