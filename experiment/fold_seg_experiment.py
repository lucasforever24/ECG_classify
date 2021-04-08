import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import AverageMeter
from utils import accuracy, update_false_list, update_score_dict
import time
from datasets.dataloader import EcgDataset
import datetime
import os
import pickle, json
import matplotlib.pyplot as plt

from networks.AnomalyClassifier import AnomalyClassifier
from networks.ECG12Classifier import ECG12Classifier
from networks.RNNClassifier import RNNClassifier
from networks.segment.ECGUNet import ECGUNet
from networks.segment.ECGFCN import ECGFCN


class EcgFoldSegExperiment(object):
    def __init__(self, config, splits):
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = torch.Tensor([2, 1, 10, 10]).to(self.device)
        self.nll_loss = nn.NLLLoss(self.weight)

        tr_keys = splits[self.config.fold]['train']
        val_keys = splits[self.config.fold]['val']

        ecg_train_dataset = EcgDataset(data_dir=self.config.data_dir, keys=tr_keys, datasets=self.config.datasets)
        self.train_data_loader = DataLoader(dataset=ecg_train_dataset, batch_size=self.config.batch_size,
                                            shuffle=True)
        ecg_val_dataset = EcgDataset(data_dir=self.config.data_dir, keys=val_keys, datasets=self.config.datasets)
        self.val_data_loader = DataLoader(dataset=ecg_val_dataset, batch_size=self.config.batch_size,
                                          shuffle=False)

        self.inf_data_loader = DataLoader(dataset=ecg_val_dataset, batch_size=1,
                                          shuffle=False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_channels = config.input_channels
        if self.config.model == 'u':
            self.model = ECGUNet(in_channels=self.input_channels, num_classes=config.num_classes, unet_classes=1)
        elif self.config.model == 'f':
            self.model = ECGFCN()
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

        self.now = str(datetime.datetime.now())[:19]
        self.save_dir = os.path.join(self.config.result_dir, self.now)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
            print('Creating result dir:', self.save_dir)

        if self.config.do_load_checkpoint:
            self.model.load_state_dict(torch.load(os.path.join(self.config.checkpoint_dir, 'model.pth')))
            self.optimizer.load_state_dict(torch.load(os.path.join(self.config.checkpoint_dir, 'optimizer.pth')))

    def train(self, epoch):
        print("======= Train =======")

        # switch to train mode
        self.model.train()

        length = 0

        length += len(self.train_data_loader.dataset)

        if epoch > 100:
            for name, param in self.model.unet.state_dict().items():
                param.requires_grad = False


        for i, data_batch in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            data = data_batch["ecg"].float().to(self.device)
            label = data_batch["label"].long().to(self.device)
            # fake_label = torch.ones(label.shape).long().to(self.device)
            # fake_label = 2*fake_label

            # compute output
            output = self.model(data) # output = (atten, y, x, y1)
            soft_max = F.softmax(output[1], dim=1)
            loss = self.nll_loss(F.log_softmax(output[1], dim=1), label)
            # over_loss = self.nll_loss(F.log_softmax(output[3], dim=1), fake_label)
            # m_loss = torch.mul(F.softmax(output[3], dim=1), F.softmax(output[3], dim=1)).sum()

            l2_loss = torch.tensor([0]).to(self.device).float()
            for name, param in self.model.named_parameters():
                if "fc" in name and 'weight' in name:
                    weight = param.data
                    l2_loss += self.config.beta * torch.sum(weight * weight).to(self.device)

            loss = loss

            loss.backward()
            self.optimizer.step()

            prec = accuracy(soft_max, label)

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
                output = self.model(data)  # output = (atten, y, y1)
                soft_max = F.softmax(output[1], dim=1)
                nll_loss = self.nll_loss(F.log_softmax(output[1], dim=1), label)
                # m_loss = torch.mul(F.softmax(output[3], dim=1), F.softmax(output[3], dim=1)).sum()
                loss = nll_loss

                prec = accuracy(soft_max, label)
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
        false_dict = dict()
        score_dict = dict()
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
                output = self.model(data)  # output = (atten, y, y1)
                soft_max = F.softmax(output[1], dim=1)
                pred = torch.argmax(soft_max, dim=1)
                loss = self.nll_loss(F.log_softmax(output[1], dim=1), label)
                # m_loss = torch.mul(F.softmax(output[2], dim=1), F.softmax(output[2], dim=1)).sum()

                prec = accuracy(soft_max, label)
                losses.update(loss.item(), data.size(0))
                total_accuracy.update(prec[0].item(), data.size(0))

                y_predict.append(pred.cpu().numpy())
                y_test.append(label.cpu().numpy())
                y_prob.append(soft_max.cpu().numpy())

                # plot the attention map
                atten = output[0]
                after_atten = output[2]
                atten_map = atten[0].cpu().numpy()
                after_atten = after_atten[0].cpu().numpy()
                ecg_data = data[0].cpu().numpy()
                # self.plot_seg(i, atten_map, ecg_data, after_atten)
                # self.plot_seg(i, atten_map, ecg_data, after_atten)

                update_false_list(soft_max, label, fname, false_dict)
                update_score_dict(soft_max, fname, score_dict)



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

        return total_accuracy, false_dict, score_dict, metrics

    def inference(self):
        print("====== Inferece ======")
        self.model.eval()
        losses = AverageMeter()
        total_accuracy = AverageMeter()

        y_predict = []
        y_test = []
        y_prob = []

        with torch.no_grad():
            for i, data_batch in enumerate(self.inf_data_loader):
                self.optimizer.zero_grad()
                data = data_batch["ecg"].float().to(self.device)
                label = data_batch["label"].long().to(self.device)
                fname = data_batch['fname']

                # compute output
                output = self.model(data)
                soft_max = F.softmax(output[1], dim=1)
                pred = torch.argmax(soft_max, dim=1)
                loss = self.nll_loss(F.log_softmax(output[1], dim=1), label)

                prec = accuracy(soft_max, label)
                losses.update(loss.item(), data.size(0))
                total_accuracy.update(prec[0].item(), data.size(0))

                y_predict.append(pred.cpu().numpy())
                y_test.append(label.cpu().numpy())
                y_prob.append(soft_max[:, 0].cpu().numpy())

                # plot the attention map
                atten = output[0]
                after_atten = output[2]
                atten_map = atten[0].cpu().numpy()
                after_atten = after_atten[0].cpu().numpy()
                ecg_data = data[0].cpu().numpy()
                self.plot_seg(i, atten_map, ecg_data, after_atten)

                update_false_list(soft_max, label, fname, self.config.datasets, self.false_dict)

                print(fname)

            print('Test result:'
                  'Loss: {losses.avg:.3f}\t'
                  'Prec@1 {total_accuracy.val:.3f} ({total_accuracy.avg:.3f})\t'.format(
                  losses=losses, total_accuracy=total_accuracy))


    def save_loss(self):
        result = dict()
        result["train_losses"] = np.asarray(self.train_losses)
        result["train_counter"] = np.asarray(self.train_counter)
        result["val_losses"] = np.asarray(self.val_losses)
        result["val_counter"] = np.asarray(self.val_counter)
        result["val_accuracy"] = np.asarray(self.val_accuracy)
        result['test_accuracy'] = np.asarray(self.test_accuracy)
        result['test_losses'] = np.asarray(self.test_losses)

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

    """
    def plot_seg(self, i, atten_map, ecg_data, after_atten):
        k = torch.randint(0, 11, (1,))
        save_dir = os.path.join(self.save_dir, 'seg')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(311)
        ax1.plot(atten_map[k], 'r', label='attention map of lead' + str(k))
        ax1.legend()
        ax2 = fig.add_subplot(312)
        ax2.plot(ecg_data[k], 'b', label='ECG signal of lead' + str(k))
        ax2.legend()
        ax3 = fig.add_subplot(313)
        ax3.plot(after_atten[k], 'b', label='ECG after atten of lead' + str(k))
        ax3.legend()
        plt.savefig(os.path.join(save_dir, str(i) + '.png'))
        plt.close()
    """

    def plot_seg(self, i, atten_map, ecg_data, after_atten, mode='val'):
        if mode == 'val':
            save_dir = os.path.join(self.save_dir, 'seg_12')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        elif mode == 'inf':
            save_dir = os.path.join(self.save_dir, 'inference')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

        self.save_heatmap(atten_map, os.path.join(save_dir, str(i) + '_atten_map.png'))
        self.save_ecg(ecg_data, os.path.join(save_dir, str(i) + '_ecg.png'))
        self.save_ecg(after_atten, os.path.join(save_dir, str(i) + '_after_atten.png'))

    def save_ecg(self, ecg, save_path):
        if self.input_channels == 12:
            fig, ax = plt.subplots(4, 3)
            fig.set_size_inches(12, 8)
            for (m, n), subplot in np.ndenumerate(ax):
                index = m * 3 + n
                subplot.plot(ecg[index])
                subplot.set_ylim(ecg.min() * 1.1, ecg.max() * 1.1)
                subplot.set_yticks([])

            plt.savefig(save_path)
            plt.close()
        elif self.input_channels == 2:
            fig, ax = plt.subplots(2, 1)
            fig.set_size_inches(12, 8)
            for m, subplot in np.ndenumerate(ax):
                index = m
                subplot.plot(ecg[index])
                subplot.set_ylim(ecg.min() * 1.1, ecg.max() * 1.1)
                subplot.set_yticks([])

            plt.savefig(save_path)
            plt.close()

    def save_heatmap(self, atten, save_path):
        if self.input_channels == 12:
            fig, ax = plt.subplots(4, 3)
            fig.set_size_inches(12, 8)
            for (m, n), subplot in np.ndenumerate(ax):
                index = m * 3 + n
                data = atten[index]
                subplot.imshow(data[np.newaxis, :], cmap='plasma', aspect='auto', extent=[0, 2000, 0, 1])
                subplot.set_yticks([])

            plt.savefig(save_path)
            plt.close()
        elif self.input_channels == 2:
            fig, ax = plt.subplots(2, 1)
            fig.set_size_inches(12, 8)
            for m, subplot in np.ndenumerate(ax):
                data = atten[m]
                subplot.imshow(data[np.newaxis, :], cmap='plasma', aspect='auto', extent=[0, 2000, 0, 1])
                subplot.set_yticks([])

            plt.savefig(save_path)
            plt.close()


    def loss_check(self):
        loss_list = self.val_losses
        l = len(loss_list)
        if l > 5:
            stop = (loss_list[l - 1] > loss_list[l - 2])
            stop = stop*(loss_list[l - 2] > loss_list[l - 3])
            stop = stop*(loss_list[l - 3] > loss_list[l - 4])
            stop = stop*(loss_list[l - 4] > loss_list[l - 5])
        else:
            stop = False

        return stop



