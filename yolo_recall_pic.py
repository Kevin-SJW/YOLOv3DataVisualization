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

def gen_recall_pic(log_path,line_num,result_dir):
    result = pd.read_csv(log_path,skiprows=[x for x in range(line_num) if (x % 10 != 9) |(x<11000)],
                         error_bad_lines=False,
                         names=['Region Avg IOU', 'Class', 'Obj', 'No Obj', 'Avg Recall_1', 'Avg Recall_2', 'count'])
    print(result)
    print("result['Avg Recall_1'] ", result['Avg Recall_1'])
    print("\n")
    print("result['Avg Recall_2'] ", result['Avg Recall_2'])
    result['Avg Recall_1'] = pd.to_numeric(result['Avg Recall_1'])
    result['Avg Recall_2'] = pd.to_numeric(result['Avg Recall_2'])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # print('avg recall', result['Avg Recall_1'].values)
    length = 13000
    print('avg recall length:', length)
    x_ticks = np.arange(0, length, 1)

    ax.scatter(x_ticks, ((result['Avg Recall_1']*0.7+result['Avg Recall_2']*0.3)).values[:length], s=0.5, label='Avg Recall')
    # ax.plot(result['Class'].values,label='Class')
    # ax.plot(result['Obj'].values,label='Obj')
    # ax.plot(result['No Obj'].values,label='No Obj')
    # ax.plot(result['Avg Recall'].values,label='Avg Recall')
    # ax.plot(result['count'].values,label='count')
    # plt.grid()
    ax.legend(loc='best')
    ax.set_title('The Avg Recall Scatter Diagram')
    ax.set_xlabel('batches')
    fig.savefig(result_dir + '/avg_recall')
    logger.info('save recall pic done')
    # plt.gca()
    plt.xlim(0, length)
    plt.ylim(0, 1)
    plt.show()

gen_recall_pic(log_path,line_num,result_dir)