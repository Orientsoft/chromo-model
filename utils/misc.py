# coding:utf-8
import os
import numpy as np

from utils import confusion_matrix, create_logger

def gen_list(root, train_save_path, val_save_path):
    """record root path data to file
    Args:
        root: str, path
    """
    if not os.path.exists(root):
        raise Exception(root)
    train_0 = os.path.join(root, 'train', '0')
    train_1 = os.path.join(root, 'train', '1')
    val_0 = os.path.join(root, 'val', '0')
    val_1 = os.path.join(root, 'val', '1')

    val_0_data = [os.path.join(val_0, p) for p in os.listdir(val_0)]
    val_1_data = [os.path.join(val_1, p) for p in os.listdir(val_1)]
    train_0_data = [os.path.join(train_0, p) for p in os.listdir(train_0)]
    train_1_data = [os.path.join(train_1, p) for p in os.listdir(train_1)]
    val_content = []
    train_content = []
    for data in val_0_data:
        val_content.append('{}, {}\n'.format(data, '0'))
    for data in val_1_data:
        val_content.append('{}, {}\n'.format(data, '1'))
    for data in train_0_data:
        train_content.append('{}, {}\n'.format(data, '0'))
    for data in train_1_data:
        train_content.append('{}, {}\n'.format(data, '1'))
    with open(train_save_path, 'w') as f:
        f.writelines(train_content)
    with open(val_save_path, 'w') as f:
        f.writelines(val_content)

def strategy1_focal(pred0, pred1, pred2, th):
    """strategy 1 for focal model finetune threshold
        softmax manner
    Args:
        gt: float, ground truth
        pred0: float, class 0' output
        pred1: float, class 1's output
        pred2: float, class 2's output
        logger: logger for output
    Return:
        ret: float, prediction result, index of class
    """
    pred = np.argmax([pred0, pred1, pred2])
    return pred

def strategy2_focal(pred0, pred1, pred2, th):
    """strategy 2 for focal model finetune threshold
        only pred0 as stander
    Args:
        gt: float, ground truth
        pred0: float, class 0' output
        pred1: float, class 1's output
        pred2: float, class 2's output
        logger: logger for output
    Return:
        ret: float, prediction result, index of class
    """
    if pred0 < th:
        pred = 0.0
    else:
        pred = 1.0
    return pred

def strategy3_focal(pred0, pred1, pred2, th):
    """strategy 3 for focal model finetune threshold
        filter by max then filter the 0 class
    Args:
        gt: float, ground truth
        pred0: float, class 0' output
        pred1: float, class 1's output
        pred2: float, class 2's output
        logger: logger for output
    Return:
        ret: float, prediction result, index of class
    """
    pred = np.argmax([pred0, pred1, pred2])
    if pred == 0:
        if pred > th:
            pred = 1
    return pred

def strategy4_focal(pred0, pred1, pred2, th):
    """strategy 4 for focal model finetune
        positive class's recall is very low
        use not bad class's output to aid this recall
    Args:
        gt: float, ground truth
        pred0: float, class 0' output
        pred1: float, class 1's output
        pred2: float, class 2's output
        logger: logger for output
    Return:
        ret: float, prediction result, index of class
    """
    pred = np.argmax([pred0, pred1, pred2])
    if pred == 0: return 0
    if pred0 + pred1 > th: return 0
    else: return 1

def strategy5_focal(pred0, pred1, pred2, th):
    """strategy 5 for focal model fintune
        positive class's recall is low
        when pred0+pred1>pred2 output 0
    Args:
        gt: float, ground truth
        pred0: float, class 0' output
        pred1: float, class 1's output
        pred2: float, class 2's output
        logger: logger for output
    Return:
        ret: float, prediction result, index of class
    """
    pred = np.argmax([pred0, pred1, pred2])
    if pred == 0: return 0
    if pred0 + pred1 > pred2: return 0
    else: return 1

def calc_acc_for_th(result, th, logger, strategy):
    logger.info('Finetuneing th {:.3f}'.format(th))
    for_conf = []
    for r in result:
        gt, pred0, pred1, pred2 = r
        pred = globals()[strategy](pred0, pred1, pred2, th)
        for_conf.append([gt, pred])
    confusion_matrix(for_conf, logger, th)

def focal_finetune_threshold(focal_result, strategy):
    """finetune the threshold for focal model
        when training or evaluating, the auto threshold computing
        is for classification, the max value of cls output as prediction
        so this func finetune the threshold for focal model
    Args:
        focal_result: str, path, eval focal model get result file
            format: 'gt, pred0, pred1, pred2\n'
        strategy: str, strategy for get prediction
    """
    if not os.path.exists(focal_result):
        raise FileNotFoundException(focal_result)
    with open(focal_result) as f:
        lines = f.readlines()
    logger = create_logger('finetune_logger', './focal_finetune.log')
    logger.info('* {}'.format(strategy))
    content = []
    for line in lines[1:]:
        l = line.strip('\n').split(', ')
        if len(l) == 4:
            gt, pred0, pred1, pred2 = l
        elif len(l) == 5:
            gt, pred0, pred1, pred2, path = l
        gt, pred0, pred1, pred2 = float(gt), float(pred0), float(pred1), float(pred2)
        content.append([gt, pred0, pred1, pred2])
    for th in np.arange(-0.9, 1.0, 0.05):
        calc_acc_for_th(content, th, logger, strategy)

def read_the_pred0_from_result(focal_result):
    """print data's output which prediction is good
        to see the output value
    Args:
        focal_result: str, path
    """
    with open(focal_result) as f:
        lines = f.readlines()
    content = []
    for line in lines[1:]:
        gt, pred0, pred1, pred2 = line.strip('\n').split(', ')
        gt, pred0, pred1, pred2 = float(gt), float(pred0), float(pred1), float(pred2)
        content.append([gt, pred0, pred1, pred2])
    print('after read there are {} data'.format(len(content)))
    pred0 = [p for p in content if np.argmax(np.array(p[1:]))==0]
    print('there are {} data pred as 0'.format(len(pred0)))
    print('gt, pred0, pred1, pred2')
    for p in pred0:
        print('{:.0f}, {:.3f}, {:.3f}, {:.3f}'.format(p[0], p[1], p[2], p[3]))


def get_three_class_acc(focal_result):
    """calc and print focal loss, three class result
    Args:
        focal_result: str, path format "gt, pred0, pred1, pred2, path\n"
    """
    with open(focal_result) as f:
        lines = f.readlines()
    content = []
    for line in lines[1:]:
        gt, pred0, pred1, pred2, path = line.strip('\n').split(', ')
        gt, pred0, pred1, pred2 = float(gt), float(pred0), float(pred1), float(pred2)
        content.append([gt, pred0, pred1, pred2])

    for i in (0,1,2):
        pred = [p for p in content if np.argmax(np.array(p[1:])) == i]
        perc = len([p for p in pred if p[0] == float(i)])
        recall = perc / len([p for p in content if p[0] == float(i)])
        perc = perc / len(pred) if len(pred)!=0 else 0
        print("class {}, perc {:.3f}, recall {:.3f}".format(i, perc, recall))
    
"""
strategy4是为了提升
"""

if __name__ == '__main__':
    #root = '/home/voyager/data/chromosome'
    #train_save_path = '/home/voyager/data/chromosome/list/train_74944.txt'
    #val_save_path = '/home/voyager/data/chromosome/list/val_1242.txt'
    #gen_list(root, train_save_path, val_save_path)
    # 现在的情况是只看最大的时候准确率低，召回率高，那么就在最大的基础上，如果筛选出来的是0类就限制一下th
    # 不如先可视化一下效果

    #strategy = 'strategy3_focal'
    #focal_result = '/home/voyager/jpz/chromosome/exp/third_dataset/res34_bs64_automask_epoch50_lr0.0001/focal_val_result_0046_epoch.txt'
    #focal_finetune_threshold(focal_result, strategy)
    #read_the_pred0_from_result(focal_result)

    #focal_result = '/home/voyager/jpz/chromosome/exp/third_dataset/res34_test/focal_val_result_0046_epoch.txt'
    #get_three_class_acc(focal_result)

    strategy = 'strategy5_focal'
    focal_result = '/home/voyager/jpz/chromosome/exp/third_dataset/res34_test/focal_val_result_0046_epoch.txt'
    focal_finetune_threshold(focal_result, strategy)