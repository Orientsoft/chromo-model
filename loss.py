# coding:utf-8
import torch
import torch.nn.functional as F
import logging

import numpy as np

class PULoss():
    def __init__(self, prior, logger, gamma=1, beta=0, loss_func=None):
        """Positive Unlabeled Learning Loss.
            reference from 
                NIPS 2017 "Positive-Unlabeled Learning with Non-Negative Risk Estimator"
                GITHUB https://github.com/kiryor/nnPUlearning
            $$ 
            L= -prior*\sum_{gt=positive}[sigmoid(-f(X))]
                + (prior*\sum_{gt=positive}[sigmoid(f(X)-1)]
                    - \sum_{gt=unlabeled}[sigmoid(f(X)-1)])_{+}
            $$
            the positive label value must be 1, unlabeled label value must be -1
        Args:
            prior: float or tensor, positive data ratio in unlabeled data.
            gamma: param for compute loss_back.
            beta: smallest negative part loss value.
            loss_func: function, compute original loss value.
                loss_func in paper is `loss_func=lambda x:torch.sigmoid(-x)`, but this will cause gradient vanish.
        """
        self.logger = logger
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        #self.loss_func = torch.nn.L1Loss()
        self.loss_func = torch.nn.SmoothL1Loss()

    def __call__(self, output, gt):
        """
        Args:
            output: tensor, model output.
            gt: tensor, ground truth.
        Returns:
            loss_val: loss value for evaluation.
            loss_back: loss for backward.
        """
        if not isinstance(self.prior, torch.Tensor):
            prior = torch.Tensor([self.prior]).to(output.device).to(output.dtype)
        positive, unlabeled = gt==1, gt==-1
        positive, unlabeled = positive.to(torch.float), unlabeled.to(torch.float)

        n_positive = torch.max(torch.Tensor([1]).to(output.device).to(output.dtype), torch.sum(positive)) 
        n_unlabeled = torch.max(torch.Tensor([1]).to(output.device).to(output.dtype), torch.sum(unlabeled))

        pred_positive = self.loss_func(output, torch.ones_like(output)) # loss_func = sigmoid(-x) pred_positive改成每个数据预测的结果和1之间的距离
        pred_unlabeled = self.loss_func(output, -torch.ones_like(output)) # pred_unlabeled改成每个结果和-1之间的距离

        positive_part_loss = torch.sum(prior * positive / n_positive * pred_positive)
        # 负样本中的梯度减去本该属于正样本的梯度
        negative_part_loss = torch.sum((unlabeled / n_unlabeled - prior * positive / n_positive) * pred_unlabeled)
        #log_str = 'Info\n\tpositive_part_loss: {}\n\tnegative_part_loss: {}'.format(positive_part_loss, negative_part_loss)
        #log_str += '\n\tPOS pred_positive: {}\n\tn_positive: {}\n\tpositive*pred_positive:{}'.format(torch.sum(pred_positive), n_positive, torch.sum(positive*pred_positive))
        #log_str += '\n\tNEG pred_negative: {}\n\tn_negative: {}\n\tnegative_before:{}\n\tnegative_agter:{}'.format(
        #        torch.sum(pred_unlabeled), n_unlabeled, torch.sum(unlabeled / n_unlabeled * pred_unlabeled), torch.sum(prior * positive / n_positive * pred_unlabeled)
        #    )
        #log_str += '\n\toutput mean: {}'.format(torch.mean(output))
        if negative_part_loss < -self.beta: # restrict negative_part_loss >= -beta
            beta = torch.Tensor([self.beta]).to(output.device).to(output.dtype)
            gamma = torch.Tensor([self.gamma]).to(output.device).to(output.dtype)
            loss_back = -gamma * negative_part_loss
            loss_val = positive_part_loss - beta
            #log_str += '\n\t*loss_val: {}\n\tloss_back: {}'.format(loss_val, loss_back)
            #self.logger.info(log_str)
            return loss_val, loss_back
        loss_val = positive_part_loss + negative_part_loss
        loss_back = loss_val
        #log_str += '\n\tloss_val: {}\n\tloss_back: {}'.format(loss_val, loss_back)
        #self.logger.info(log_str)
        return loss_val, loss_back

    def __repr__(self):
        return 'PULoss()'

class FocalLoss():
    def __call__(self, pred,
                    target,
                    weight=None,
                    gamma=0.8,
                    alpha=0.5,
                    reduction='mean'):
        #logger = logging.getLogger('global_logger')
        # add by jpz
        one_hot = torch.zeros_like(pred).cuda()
        #logger.info('sigmoid focal loss, one hot shape {} {}'.format(one_hot.shape, one_hot))
        y = target.to(torch.long).view(-1, 1)
        #logger.info('sigmoid focal loss, y shape {} {}'.format(y.shape,y))
        one_hot.scatter_(1, y, 1)
        #logger.info('sigmoid focal loss, after scatter one hot {}'.format(one_hot))
        target = one_hot
        
        #logger.info('sigmoid focal loss, target {}'.format(target))
        
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        if weight==None: # weight是每个标签的权重，这里暂改为全1
            weight = torch.ones((pred.shape[1])).cuda()
            weight.fill_(20) # 10倍正常的lr即可
        
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
        weight = weight * pt.pow(gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * weight
        
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            raise Error('unknow reduction manner: {}'.format(reduction))

    def __repr__(self):
        return 'FocalLoss()'

if __name__ == '__main__':
    loss = PULoss(0.4)
    print(loss)
    #output = torch.zeros((10,))
    #gt = torch.ones((10,))
    output = torch.Tensor([1, 1, 1])
    gt = torch.Tensor([1,1,1])
    print(loss(output, gt))