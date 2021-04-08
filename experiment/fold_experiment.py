import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import AverageMeter
from utils import accuracy, update_false_list
import time
from datasets.dataloader import EcgDataset
import datetime
import os
import pickle, json
import matplotlib.pyplot as plt

from networks.AnomalyClassifier import AnomalyClassifier
from networks.ECG12Classifier import ECG12Classifier
from networks.RNNClassifier import RNNClassifier
from networks.segment.ECGUNet import Classifier
from networks.ECGGroupClassifier import ECGGroupClassifier

class EcgFoldExperiment(object):
    def __init__(self, config, splits):
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = torch.Tensor([7, 1]).to(self.device)
        self.nll_loss = nn.NLLLoss(self.weight)

        tr_keys = splits[self.config.fold]['train']
        val_keys = splits[self.config.fold]['val']

        ecg_train_dataset = EcgDataset(data_dir=self.config.data_dir, keys=tr_keys, datasets=self.config.datasets)
        self.train_data_loader = DataLoader(dataset=ecg_train_dataset, batch_size=self.config.batch_size,
                                            shuffle=True)
        ecg_val_dataset = EcgDataset(data_dir=self.config.data_dir, keys=val_keys, datasets=self.config.datasets)
        self.val_data_loader = DataLoader(dataset=ecg_val_dataset, batch_size=self.config.batch_size,
                                          shuffle=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = ECG12Classifier(input_channel=12, initial_channel=32, num_classes=2)
        # self.model = AnomalyClassifier(in_channels=12, num_classes=2)
        # self.model = RNNClassifier()
        self.model = Classifier(in_channels=2)
        # self.model = ECGGroupClassifier()

        """
        if torch.cuda.device_count() > 1:
            print("Let's ues", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        """

        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate)

        self.train_losses = []
        self.train_counter = []
        self.val_losses = []
        self.val_counter = []
        self.val_accuracy = []
        self.test_losses = []
        self.test_accuracy = []
        self.false_dict = dict()
        for k in self.config.datasets:
            self.false_dict[k] = []
        """
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.model.apply(init_weights)
        """

        self.now = str(datetime.datetime.now())[:19]
        self.save_dir = os.path.join(self.config.result_dir, self.now)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
            print('Creating result dir:', self.save_dir)

    def train(self, epoch):
        print("======= Train =======")

        # switch to train mode
        self.model.train()

        length = 0

        length += len(self.train_data_loader.dataset)

        for i, data_batch in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            data = data_batch["ecg"].float().to(self.device)
            label = data_batch["label"].long().to(self.device)

            # compute output
            output = self.model(data)
            loss = self.nll_loss(F.log_softmax(output, dim=1), label)

            l2_loss = torch.tensor([0]).to(self.device)
            for name, param in self.model.named_parameters():
                if "fc" in name and "weight" in name:
                    weight = param.data
                    l2_loss = self.config.beta * torch.sum(weight * weight).to(self.device)

            loss = loss

            loss.backward()
            self.optimizer.step()

            prec = accuracy(output, label)

            if i % self.config.print_freq == 0:
                print('Epoch: [{0}][{1}][{2}/{3}]\t'
                      'Loss: {loss:.3f}\t'
                      'Prec: {prec:.3f}'.format(self.config.fold,
                       epoch, i, len(self.train_data_loader), loss=loss.item(), prec=prec[0]))
                self.train_losses.append(loss.item())
                self.train_counter.append(
                    (i * self.config.batch_size) + ((epoch + 1) * len(self.train_data_loader.dataset)))
                # save result
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model.pth'))
                torch.save(self.optimizer.state_dict(), os.path.join(self.save_dir, 'optimizer.pth'))
                self.save_loss()
        self.val_counter.append((epoch+1)*length)

    def val(self, epoch):
        print("======= Val =======")
        losses = AverageMeter()
        total_accuracy = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            for i, data_batch in enumerate(self.val_data_loader):
                self.optimizer.zero_grad()
                data = data_batch["ecg"].float().to(self.device)
                label = data_batch["label"].long().to(self.device)

                # compute output
                output = self.model(data)
                loss = self.nll_loss(F.log_softmax(output, dim=1), label)

                l2_loss = torch.tensor([0]).to(self.device)
                for name, param in self.model.named_parameters():
                    if "fc.weight" in name:
                        weight = param.data
                        l2_loss = self.config.beta * torch.sum(weight * weight).to(self.device)

                loss = loss + l2_loss

                prec = accuracy(output, label)
                losses.update(loss.item(), data.size(0))
                total_accuracy.update(prec[0].item(), data.size(0))

            print('Epoch: [{0}][{1}]]\t'
                  'Loss: {loss.avg:.3f}\t'
                  'Prec@1 {total_accuracy.val:.3f} '
                  '({total_accuracy.avg:.3f})\t'.format(self.config.fold, epoch,
                                                        loss=losses, total_accuracy=total_accuracy))
            self.val_losses.append(losses.avg)
            self.val_accuracy.append(total_accuracy.avg)
            self.save_loss()

    def test(self):
        print("======= Test =======")
        losses = AverageMeter()
        total_accuracy = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        num_of_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("number of parameters:", num_of_parameters)

        y_predict = []
        y_test = []
        y_prob = []

        with torch.no_grad():
            for i, data_batch in enumerate(self.val_data_loader):
                self.optimizer.zero_grad()
                data = data_batch["ecg"].float().to(self.device)
                label = data_batch["label"].long().to(self.device)
                fname = data_batch['fname']

                # compute output
                output = self.model(data)
                soft_max = F.softmax(output, dim=1)
                pred = torch.argmax(soft_max, dim=1)
                loss = self.nll_loss(F.log_softmax(output, dim=1), label)

                prec = accuracy(output, label)
                losses.update(loss.item(), data.size(0))
                total_accuracy.update(prec[0].item(), data.size(0))

                y_predict.append(pred.cpu().numpy())
                y_test.append(label.cpu().numpy())
                y_prob.append(soft_max[:, 1].cpu().numpy())

                update_false_list(output, label, fname, self.config.datasets, self.false_dict)

            print('Test result:'
                  'Loss: {losses.avg:.3f}\t'
                  'Prec@1 {total_accuracy.val:.3f} ({total_accuracy.avg:.3f})\t'.format(
                  losses=losses, total_accuracy=total_accuracy))

            y_predict = np.concatenate(y_predict)
            y_prob = np.concatenate(y_prob)
            y_test = np.concatenate(y_test)
            metrics = dict()
            metrics['predict'] = y_predict
            metrics['prob'] = y_prob
            metrics['test'] = y_test

            self.test_losses.append(losses.avg)
            self.test_accuracy.append(total_accuracy.avg)

            self.save_loss()

        self.plot_result()

        return total_accuracy, self.false_dict, metrics

    def save_loss(self):
        result = dict()
        result["train_losses"] = np.asarray(self.train_losses)
        result["train_counter"] = np.asarray(self.train_counter)
        result["val_losses"] = np.asarray(self.val_losses)
        result["val_counter"] = np.asarray(self.val_counter)
        result["val_accuracy"] = np.asarray(self.val_accuracy)
        result['test_accuracy'] = np.asarray(self.test_accuracy)
        result['test_losses'] = np.asarray(self.test_losses)
        result['false_dict'] = self.false_dict

        with open(os.path.join(self.save_dir, "result_log.p"), 'wb') as fp:
            pickle.dump(result, fp)

    def plot_result(self):
        result_log_path = os.path.join(self.save_dir, "result_log.p")
        with open(result_log_path, 'rb') as f:
            result_dict = pickle.load(f)

        train_losses = result_dict['train_losses']
        train_counter = result_dict['train_counter']
        val_losses = result_dict['val_losses']
        val_counter = result_dict['val_counter']
        val_accuracy = result_dict['val_accuracy']

        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(111)
        ax1.plot(train_counter, train_losses, 'b')
        ax1.legend('Train_losses')
        plt.savefig(os.path.join(self.save_dir, "train_loss.png"))

        fig2 = plt.figure(figsize=(12, 8))
        ax2 = fig2.add_subplot(111)
        ax2.plot(val_counter, val_losses, 'b', label='Val_losses')
        ax2.legend()
        ax3 = ax2.twinx()
        ax3.plot(val_counter, val_accuracy, 'r', label='Val_accuracy')
        ax3.legend()
        plt.savefig(os.path.join(self.save_dir, "validation.png"))
        plt.close()


