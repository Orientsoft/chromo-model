# coding:utf-8
import os

def embed_ext_result_(p_cls_res, n_cls_res, balance='down'):
    """embeding two ext model's val res file
    Args:
        p_cls_res: str, path, for positive classifier's output
        n_cls_res: str, path, for negative classifier's output
        balance: str, balance stragey
    """
    assert os.path.exists(p_cls_res) and os.path.exists(n_cls_res)
    res = {
        '0':{
            'pred': 0,
            'gt': 0,
            'correct':0
        },
        '1':{
            'pred': 0,
            'gt': 0,
            'correct':0
        }
    }
    
    with open(p_cls_res) as f:
        p_res = f.readlines()
    with open(n_cls_res) as f:
        n_res = f.readlines()

    balance_n_flag = 0
    balance_p_flag = 0
    if balance == 'down':
        balance_flag = 50
    else:
        balance_flag = len(p_res)
    
    p_res = [(p.strip('\n').split(', ')) for p in p_res[1:]]
    p_res = [(p[0].split('.')[0], p[1].split('.')[0]) for p in p_res]
    n_res = [(p.strip('\n').split(', ')) for p in n_res[1:]]
    n_res = [(p[0].split('.')[0], p[1].split('.')[0]) for p in n_res]
    for p_r, n_r in zip(p_res, n_res):
        p_gt, p_pred = p_r
        n_gt, n_pred = n_r
        # 当坏分类器预测为坏的时候那么肯定它是坏的
        # 当好分类器预测为好的时候那么相信它是好的
        # 当坏分类器预测为好，好分类器预测为坏的时候相信它是好的
        if balance_p_flag > balance_flag or balance_n_flag > balance_flag:
            break
        if p_gt == '1': balance_p_flag += 1
        if p_gt == '0': balance_n_flag += 1
        res[p_gt]['gt'] += 1
        if n_pred == '1' or n_pred == '2':
            pred = '1'
        elif p_pred == '0' and n_pred=='0':
            pred = p_pred
        else:
            pred = '1'
        res[pred]['pred'] += 1
        if pred == p_gt:
            res[p_gt]['correct'] += 1
    cls0_perc = res['0']['correct'] / res['0']['pred'] if res['0']['pred'] !=0 else 0 
    cls0_recall = res['0']['correct'] / res['0']['gt'] if res['0']['gt'] !=0 else 0
    cls1_perc = res['1']['correct'] / res['1']['pred'] if res['1']['pred'] != 0 else 0
    cls1_recall =  res['1']['correct'] / res['1']['gt'] if res['1']['gt'] != 0 else 0
    acc = (res['0']['correct'] + res['1']['correct']) / (res['0']['gt'] + res['1']['gt'])
    print(res)
    print('total acc: {:.4f}'.format(acc))
    print('class 0 perc: {:.4f}, recall: {:.4f}'.format(cls0_perc, cls0_recall))
    print('class 1 perc: {:.4f}, recall: {:.4f}'.format(cls1_perc, cls1_recall))

if __name__ == '__main__':
    p_cls_res = '/home/voyager/jpz/chromosome/exp/ext/res101_bs16_epoch80_automask_lr0.0001/trainset_0038.txt'
    n_cls_res = '/home/voyager/jpz/chromosome/exp/ext/res101_bs16_epoch80_automask_lr0.0001/trainset_0024.txt'
    embed_ext_result_(p_cls_res, n_cls_res, balance='down')