# coding:utf-8
import os
import torch
import logging
import numpy as np

def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l

def save_val_result(result, epoch, file, logger):
    """save val dataset result to file.
    Args:
        result: List[[gt, pred], ...]
        file: str, file path
        logger: logger instance
    """
    file = file.format(epoch)
    if not os.path.exists(os.path.dirname(file)):
        logger.error('save_val_output: {} not exists'.format(os.path.dirname(file)))
    with open(file, 'w') as f:
        if len(result[0]) == 2:
            lines = ['gt, pred\n']
        elif len(result[0]) == 5:
            lines = ['gt, pred0, pred1, pred2, path\n']
        for item in result:
            if len(item) == 2:
                gt, pred = item
                lines.append('{:.3f}, {:.3f}\n'.format(gt, pred))
            elif len(item) == 5:
                gt, pred0, pred1, pred2, path = item
                lines.append('{:.3f}, {:.3f}, {:.3f}, {:.3f}, {}\n'.format(gt, pred0, pred1, pred2, path))
        f.writelines(lines)

def ext_save_output(output_list, save_file):
    """save model's original output to file
    Args:
        output_list: [[path, gt, pred0, pred1, pred2], ...]
        save_file: str, 
    """
    content = ['{}, {}, {:.4f}, {:.4f}, {:.4f}\n'.format(p[0], p[1], p[2], p[3], p[4]) for p in output_list]
    with open(save_file, 'w') as f:
        f.writelines(content)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    Args:
        output: Tensor, size: (batchsize, classes)
        target: Tensor, size: (batchsize, 1)
        topk: tuple
    Return:
        res: list[topk_res]
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True) # 沿着dim=1给出maxk, size:(batchsize, 5)
    pred = pred.t() # (5, batchsize)
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # (5, batchsize) bool

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True) # scalar
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def regress_accuracy_val(result, threshold=0.5):
    """Compute the precision for regression output
    Args:
        result: list[[gt, pred], ...]
        threshold: float, for classify
    Return:
        acc: float, accuracy
    """
    all, right = len(result), 0
    for gt, pred in result:
        if (gt == 0 and pred < threshold) or (gt > 0 and pred >threshold):
            right += 1
    return (right/all) * 100.0

def regress_accuracy_train(output, target, threshold=0.5, ispul=False):
    """Computes the precision for regression output
    Args:
        output: Tensor, size(batchsize, 1)
        target: Tensor, size(batchsize, 1)
        threshold: float, for 0/1 classify
        ispul: bool, if true the positive label is 1, negative label is -1.
    Return:
        res: list[top1_res]
    """
    if not ispul:
        higher, lower = 1, 0
    else:
        higher, lower = 1, -1
    batch_size = target.size(0)
    pred = output[:]
    pred[pred > threshold] = higher
    pred[pred <= threshold] = lower
    target = target.to(torch.long)
    pred = pred.to(torch.long)
    correct = pred.eq(target) # (5, batchsize) bool

    res = []
    correct_k = correct.view(-1).float().sum(0, keepdim=True) # scalar
    res.append(correct_k.mul_(100.0 / batch_size))
    return res

def load_state(path, model):
    def map_func(storage, location):
        return storage.cuda()
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=map_func)
        model.load_state_dict(checkpoint['net'], strict=False)
    else:
        raise FileNotFoundError('State Dict File {} not found'.format(path))

def focal_get_result(paths, gts, output):
    """save batch result to result array, for focal loss model
    Args:
        gts: tensor, (batchsize, 1), ground truth
        output: tensor, (batchsize, num_class)
        paths: list[str], data's path
    Return:
        ret: list[[gt, pred], [gt, pred], ...]
    """
    ret = []
    pred = output[:]
    for pth, gt, p in zip(paths, gts, pred):
        #print(pth, gt, p)
        gt = gt.tolist()
        p = p.tolist()
        ret.append([gt, p[0], p[1], p[2], pth])
    return ret

def regress_get_result(gts, output):
    """save batch result to result array, for regression model
    Args:
        gts: tensor, (batchsize, 1), ground truth
        output: tensor, (batchsize, 1)
    Return:
        ret: list[[gt, pred], [gt, pred], ...]
    """
    ret = []
    pred = output[:]
    #pred[pred > threshold] = 1
    #pred[pred <= threshold] = 0
    #target = gts.to(torch.long)
    #pred = pred.to(torch.long)

    for gt, p in zip(gts, pred):
        gt = gt.tolist()[0]
        p = p.tolist()[0]
        ret.append([gt, p])
    return ret

def get_result(gts, output):
    """save batch result to result array
    Args:
        gts: tensor, (batchsize ,1), ground truth
        output: tensor, (batchsize, num_class)
    Return:
        ret: list[[gt, pred], [gt, pred], ...]
    """
    ret = []
    _, pred = output.topk(1, 1, True, True)
    for gt, p in zip(gts, pred):
        #gt, p = gt.tolist(), p.tolist()
        #print('gt type {} pred type {}'.format(type(gt), type(p)))
        gt = int(gt.tolist())
        p = int(p.tolist()[0])
        ret.append([gt, p])
    return ret

def ext_get_output(paths, gts, outputs):
    """get model's original output
    Args：
        paths: tensor
        gts: tensor
        outputs: tensor
    Returns:
        ret: [[path, gt, output], ...] 
    """
    ret = []
    for path, gt, pred in zip(paths, gts, outputs):
        gt = int(gt.tolist())
        pred = pred.tolist()
        pred0 = float(pred[0])
        pred1 = float(pred[1])
        pred2 = float(pred[2])
        ret.append([path, gt, pred0, pred1, pred2])
        #print([path, gt, pred0, pred1, pred2])
        #exit()
    return ret

def confusion_matrix(result, logger, threshold=0.5):
    """calc confusion matrix and precision recall of class 0/1
    Args:
        result: list[[gt, pred], ...]
        logger: logger object
        threshold: float
    Returns:
        tp: [tp_number, tp_ratio], true positive
        fp: [fp_number, pf_ratio], false positive
        fn: [fn_number, fn_ratio], false negative
        tn: [fn_number, tn_ratio], true negative
        cls0_prec: float, precision of class 0
        cls0_recall: float, recall of class 0
        cls1_perc: float, precision of calss 1
        cls1_recall: float, precision of class1
    """
    tp, fp, fn, tn = [0,0], [0,0], [0,0], [0,0]
    for res in result:
        gt, pred = res
        if gt == 0  and pred < threshold: # true positive
            tp[0] += 1
        elif gt > 0 and pred < threshold: # false positive
            fp[0] += 1
        elif gt == 0 and pred > threshold: # false negative
            fn[0] += 1
        else: # true negative
            tn[0] += 1

    total = tp[0] + fp[0] + fn[0] + tn[0]
    tp[1] = (tp[0]/total) * 100.0
    fp[1] = (fp[0]/total) * 100.0
    fn[1] = (fn[0]/total) * 100.0
    tn[1] = (tn[0]/total) * 100.0

    cls0_prec = (tp[0] / (tp[0] + fp[0]) if (tp[0]+fp[0]) != 0 else 0) * 100.0
    cls0_recall = (tp[0] / (tp[0] + fn[0]) if (tp[0] + fn[0]) != 0 else 0) * 100.0
    cls1_prec = (tn[0] / (tn[0] + fn[0]) if (tn[0]+fn[0])!=0 else 0) * 100.0
    cls1_recall = (tn[0] / (tn[0] + fp[0]) if (tn[0]+fp[0])!=0 else 0) * 100.0

    logger.info('confusion matrix:'
                '\n\tgt\pred\t0\t\t1'
                '\n\t      0\t{:.3f}({})\t{:.3f}({})'
                '\n\t      1\t{:.3f}({})\t{:.3f}({})'.format(
                    tp[1], tp[0], fn[1], fn[0],
                    fp[1], fp[0], tn[1], tn[0]
                )) # output confusion matrix
    logger.info(
                '\n\tClass 0 Precision: {:.3f}, Recall: {:.3f}'
                '\n\tClass 1 Precision: {:.3f}, Recall: {:.3f}'.format(cls0_prec, cls0_recall,cls1_prec, cls1_recall))

def confusion_matrix_regress(result, logger, pul=False):
    """auto select threshold to compute best prec, threshold step is 0.1
    Args:
        result: list[[float(gt), float(pred)], ...]
        logger: logger object
        pul: bool, positive unlabeled learning
    Return:
        val_perc: float
    """
    best_acc, best_th = -1, 0
    logger.info("Computing threshold...")
    arange = np.arange(-0.9, 2, 0.1)
    for th in arange:
        acc = regress_accuracy_val(result, th)
        if acc > best_acc:
            best_acc = acc
            best_th = th
    logger.info("best threshold:{:.2f} best accuracy:{:.3f}".format(best_th, best_acc))
    confusion_matrix(result, logger, threshold=th)
    return best_acc
    
def gen_list(path=[], label='1'):
    """generate list for train and val
    Args:
        path: List[str(path), ...], path is dir contains data(JPG)
        label: str, label of this list
    Returns:
        lines: List['data_pth label\n', ...]
    """
    lines = []
    for p in path:
        if not os.path.exists(p):
            raise Exception('Path {} not exists'.format(p))
        data = [dp for dp in os.listdir(p) if 'JPG' in dp]
        data = [os.path.join(p, dp) for dp in data]
        lines.extend(['{} {}\n'.format(dp, label) for dp in data])
    return lines


def ext_calc_bad_cls_result(val_file, cls_idx=2):
    """get the recall and perc from val file of ext dataset
       Here we just consider the most bad class.
    Args:
        val_file: str, val dataset output file
        cls_idx: target class index
    """
    cls_idx = int(cls_idx)
    if not os.path.exists(val_file):
        raise FileNotFoundError(val_file)
    cls_prec, cls_recall, cls_gt, cls_pred = 0, 0, 0, 0
    with open(val_file) as f:
        lines = f.readlines()
    for line in lines[1:]:
        gt, pred = line.strip('\n').split(', ')
        pred, gt = int(float(pred)), int(float(gt))
        if gt == cls_idx:
            cls_gt += 1
        if gt==cls_idx and pred==cls_idx:
            cls_prec += 1
        if pred == cls_idx:
            cls_pred += 1

    cls_recall = cls_prec / cls_gt
    cls_prec /= cls_pred
    print('cls: {} gt: {}, pred: {}, prec: {}, recall: {}'.format(cls_idx, cls_gt, cls_pred, cls_prec, cls_recall))

def calc_bad_cls_result(val_file):
    """use val result file to calc the most bad data ratio
       the val_file may named as trainset_{:04d}.txt
    Args:
        val_file: str, val dataset output file
    """
    if not os.path.join(val_file):
        raise FileNotFoundError(val_file)
    most_bad, bad, pred_bad = 0, 0, 0
    with open(val_file) as f:
        lines = f.readlines()
    for line in lines[1:]:
        gt, pred = line.strip('\n').split(', ')
        pred, gt = int(float(pred)), int(float(gt))
        if gt == 1:
            bad += 1
        if gt == 1 and pred == 2: # pred as most bad
            most_bad += 1
        if gt == 1 and pred == 1: # pred as bad
            pred_bad += 1

    print('Total bad data {}, pred most bad {}, normal bad {}'.format(bad, most_bad, pred_bad))


if __name__ == '__main__':
    VAL_V1_0 = '/home/voyager/data/chromosome/val-v1/0'
    VAL_V1_1 = '/home/voyager/data/chromosome/val-v1/1'
    TRAIN_0 = '/home/voyager/data/chromosome/train/0'
    TRAIN_1 = '/home/voyager/data/chromosome/train/1'
    VAL_0 = '/home/voyager/data/chromosome/val/0'
    VAL_1 = '/home/voyager/data/chromosome/val/0'
    #val_file = '/home/voyager/jpz/chromosome/exp/ext/res34_bs64_automask_lr0.0001/val_result_0046_epoch.txt'
    #val_file = '/home/voyager/jpz/chromosome/exp/ext/res34_bs64_automask_lr0.0001/trainset_0047.txt'
    #ext_calc_bad_cls_result(val_file)
    #calc_bad_cls_result(val_file)
    path = '/home/voyager/jpz/chromosome/exp/ext/res101_bs16_epoch80_automask_lr0.0001'
    val_file_list = [p for p in os.listdir(path) if 'val_result' in p]
    for val_file in val_file_list:
        print('Evaluating good result', val_file)
        ext_calc_bad_cls_result(os.path.join(path, val_file), 0)