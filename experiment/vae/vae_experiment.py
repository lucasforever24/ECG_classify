import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import AverageMeter
from datasets.dataloader import VaeDataset
import datetime
import os
import pickle
import matplotlib.pyplot as plt

from networks.vae.VAE_1d import VAE


class VaeExperiment(object):
    def __init__(self, config):
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.loss_type = self.config.loss_type
        self.beta = self.config.beta

        pkl_dir = self.config.split_dir
        with open(os.path.join(pkl_dir, "splits.pkl"), 'rb') as f:
            splits = pickle.load(f)

        tr_keys = splits['train']
        val_keys = splits['val']
        test_keys = splits['test']

        vae_train_dataset = VaeDataset(data_dir=self.config.data_dir, keys=tr_keys)
        self.train_data_loader = DataLoader(dataset=vae_train_dataset, batch_size=self.config.batch_size,
                                            shuffle=True)
        vae_val_dataset = VaeDataset(data_dir=self.config.data_dir, keys=val_keys)
        self.val_data_loader = DataLoader(dataset=vae_val_dataset, batch_size=self.config.batch_size,
                                          shuffle=True)
        vae_test_dataset = VaeDataset(data_dir=self.config.data_dir, keys=test_keys)
        self.test_data_loader = DataLoader(dataset=vae_test_dataset, batch_size=self.config.batch_size,
                                           shuffle=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = ECG12Classifier(input_channel=12, initial_channel=32, num_classes=2)
        self.model = VAE(in_channels=1, latent_dim=32, batch_size=self.config.batch_size)

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
        self.test_losses = []

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

        length += len(self.train_data_loader)

        for i, data_batch in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            data = data_batch["ecg"].float().to(self.device)

            # compute output
            results = self.model(data)
            loss = self.loss_function(*results,
                                      M_N=self.config.batch_size / len(self.train_data_loader.dataset))
            loss.backward()
            self.optimizer.step()

            if i % self.config.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss: {loss:.3f}\t'.format(
                       epoch, i, len(self.train_data_loader), loss=loss.item()/self.config.batch_size))
                self.train_losses.append(loss.item())
                self.train_counter.append(
                    (i * self.config.batch_size) + ((epoch + 1) * len(self.train_data_loader)))
                # save result
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model.pth'))
                torch.save(self.optimizer.state_dict(), os.path.join(self.save_dir, 'optimizer.pth'))
                self.save_loss()
        self.val_counter.append((epoch+1)*length)

    def val(self, epoch):
        print("======= Val =======")
        losses = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            for i, data_batch in enumerate(self.val_data_loader):
                self.optimizer.zero_grad()
                data = data_batch["ecg"].float().to(self.device)

                # compute output
                results = self.model(data)
                loss = self.loss_function(*results,
                                          M_N=self.config.batch_size/len(self.train_data_loader.dataset))

                losses.update(loss.item()/self.config.batch_size, data.size(0))

                if i % self.config.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss: {loss.avg:.3f}\t'.format(
                           epoch, i, len(self.val_data_loader), loss=losses))

            self.val_losses.append(losses.avg)
            self.save_loss()

    def test(self):
        print("======= Val =======")
        losses = AverageMeter()
        # switch to evaluate mode
        self.model.eval()

        num_of_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("number of parameters:", num_of_parameters)

        with torch.no_grad():
            for i, data_batch in enumerate(self.test_data_loader):
                self.optimizer.zero_grad()
                data = data_batch["ecg"].float().to(self.device)

                # compute output
                results = self.model(data)
                loss = self.loss_function(*results,
                                          M_N=self.config.batch_size / len(self.train_data_loader.dataset))

                losses.update(loss.item()/self.config.batch_size, data.size(0))

                if i % self.config.print_freq == 0:
                    recon = results[0].cpu().numpy()
                    input = results[1].cpu().numpy()

                    self.plot_ecg(i, input[0, 0], recon[0, 0])

            print('Test result:'
                  'Loss: {losses.avg:.3f}\t'.format(
                  losses=losses))

            self.test_losses.append(losses.avg)

            self.save_loss()

        self.plot_result()

    def save_loss(self):
        result = dict()
        result["train_losses"] = np.asarray(self.train_losses)
        result["train_counter"] = np.asarray(self.train_counter)
        result["val_losses"] = np.asarray(self.val_losses)
        result["val_counter"] = np.asarray(self.val_counter)
        result['test_losses'] = np.asarray(self.test_losses)

        with open(os.path.join(self.save_dir, "result_log.p"), 'wb') as fp:
            pickle.dump(result, fp)

        # plot the accu

    def plot_result(self):
        result_log_path = os.path.join(self.save_dir, "result_log.p")
        with open(result_log_path, 'rb') as f:
            result_dict = pickle.load(f)

        train_losses = result_dict['train_losses']
        train_counter = result_dict['train_counter']
        val_losses = result_dict['val_losses']
        val_counter = result_dict['val_counter']

        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(111)
        ax1.plot(train_counter, train_losses, 'b')
        ax1.legend('Train_losses')
        plt.savefig(os.path.join(self.save_dir, "train_loss.png"))

        fig2 = plt.figure(figsize=(12, 8))
        ax2 = fig2.add_subplot(111)
        ax2.plot(val_counter, val_losses, 'b')
        ax2.legend("Val_losses")
        plt.savefig(os.path.join(self.save_dir, "validation.png"))

    def plot_ecg(self, i, input, recon):
        ecg_save_dir = os.path.join(self.save_dir, 'ecg_signal')
        if not os.path.exists(ecg_save_dir):
            os.mkdir(ecg_save_dir)

        fig3 = plt.figure(figsize=(12, 8))
        ax2 = fig3.add_subplot(111)
        ax2.plot(input, 'b')
        ax3 = ax2.twinx()
        ax3.plot(recon, 'r')
        ax2.legend("original signal")
        ax3.legend("reconstructed signal")
        plt.savefig(os.path.join(ecg_save_dir, str(i)+'.png'))

    def loss_function(self, *args, M_N):
        # self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = M_N # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input, reduction='sum')

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + self.beta * kld_weight * kld_loss

        """
        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')
        """

        # return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}
        return loss


