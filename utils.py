from sklearn import metrics
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, classification_report)
import numpy as np
import pickle


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_dict_to_txt(dict, save_path):
    with open(save_path, 'w') as f:
        print(dict, file=f)


def update_false_list(output, target, fname, false_pred_dict):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.cpu().numpy()
    for i in range(pred.shape[0]):
        if pred[i] != target[i]:
            gt = target[i].cpu().numpy()
            false_pred_dict[fname[i]] = [gt, pred[i]]


def update_score_dict(output, fname, score_dict):
    _, pred = output.topk(1, 1, True, True)
    for i in range(pred.shape[0]):
        score_dict[fname[i]] = output[i].cpu().numpy()


def print_metrices_out(y_predicted, y_test, y_prob, avg='micro'):
    y_prob = y_prob[:, 1]
    print("Accuracy is %f (in percentage)" %
          (accuracy_score(y_test, y_predicted) * 100))
    matrix = confusion_matrix(y_test, y_predicted)
    print("Confusion Matrix: \n" + str(confusion_matrix(y_test, y_predicted)))
    print("Recall score is %f." % recall_score(y_test, y_predicted, average=avg))
    print("Precision score is %f." %
          precision_score(y_test, y_predicted, average=avg))
    print("F1 score is %f." % f1_score(y_test, y_predicted, average=avg))
    test_auc = metrics.roc_auc_score(y_test, y_prob, average=avg)
    print("AUC score is %f." % test_auc)
    test_ap = metrics.average_precision_score(y_test, y_predicted, average=avg)
    print("Average precision score is %f." % test_ap)
    print("classification Report: \n" +
          str(classification_report(y_test, y_predicted)))
    print("-----------------------------------\n")
    return matrix


def print_metrices_out_multiclass(y_predicted, y_test, y_prob, avg='micro'):
    print("Accuracy is %f (in percentage)" %
          (accuracy_score(y_test, y_predicted) * 100))
    matrix = confusion_matrix(y_test, y_predicted)
    print("Confusion Matrix: \n" + str(confusion_matrix(y_test, y_predicted)))
    print("Recall score is %f." % recall_score(y_test, y_predicted, average=avg))
    print("Precision score is %f." %
          precision_score(y_test, y_predicted, average=avg))
    print("F1 score is %f." % f1_score(y_test, y_predicted, average=avg))
    test_auc = metrics.roc_auc_score(y_test, y_prob, multi_class='ovr')
    print("AUC score is %f." % test_auc)
    # test_ap = metrics.average_precision_score(y_test, y_prob)
    # print("Average precision score is %f." % test_ap)
    print("classification Report: \n" +
          str(classification_report(y_test, y_predicted)))
    print("-----------------------------------\n")
    return matrix



