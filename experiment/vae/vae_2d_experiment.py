import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset
from utils import AverageMeter
import datetime
import os
import pickle
import matplotlib.pyplot as plt

from networks.vae.VAE_2d import VAE


class VaeExperiment(object):
    def __init__(self, config):
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.loss_type = self.config.loss_type
        self.beta = self.config.beta

        # MNIST Dataset
        train_dataset = datasets.MNIST(root=self.config.data_dir, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.MNIST(root=self.config.data_dir, train=False, transform=transforms.ToTensor(),
                                      download=False)

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.config.batch_size, shuffle=False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = ECG12Classifier(input_channel=12, initial_channel=32, num_classes=2)
        self.model = VAE(in_channels=1, latent_dim=64, batch_size=self.config.batch_size)
        # self.model = VAE_mlp(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2)


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

        now = str(datetime.datetime.now())[:19]
        self.save_dir = os.path.join(self.config.result_dir, 'vae_' + now)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
            print('Creating result dir:', self.save_dir)

    def train(self, epoch):
        print("======= Train =======")

        # switch to train mode
        self.model.train()

        length = 0

        length += len(self.train_loader)

        for i, (data, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            # data = data_batch["ecg"].float().to(self.device)
            data = data.to(self.device)

            # compute output
            results = self.model(data)
            loss = self.loss_function(*results,
                                      M_N=self.config.batch_size / len(self.train_loader.dataset))
            loss.backward()
            self.optimizer.step()

            if i % self.config.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss: {loss:.3f}\t'.format(
                       epoch, i, len(self.train_loader), loss=loss.item()))
                self.train_losses.append(loss.item())
                self.train_counter.append(
                    i + ((epoch + 1) * len(self.train_loader)))
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
            for i, (data, _) in enumerate(self.test_loader):
                self.optimizer.zero_grad()
                # data = data_batch["ecg"].float().to(self.device)
                data = data.to(self.device)

                # compute output
                results = self.model(data)
                loss = self.loss_function(*results,
                                          M_N=self.config.batch_size/len(self.train_loader.dataset))

                losses.update(loss.item(), data.size(0))

                if i % self.config.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss: {loss.avg:.3f}\t'.format(
                           epoch, i, len(self.test_loader), loss=losses))

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
            for i, (data, _) in enumerate(self.test_loader):
                self.optimizer.zero_grad()
                # data = data_batch["ecg"].float().to(self.device)
                data = data.to(self.device)

                # compute output
                results = self.model(data)
                loss = self.loss_function(*results,
                                          M_N=self.config.batch_size / len(self.train_loader.dataset))

                losses.update(loss.item(), data.size(0))

                if i % self.config.print_freq == 0:
                    recon = results[0].cpu()
                    input = results[1].cpu()

                    self.save_image(i, input, recon)

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

    def save_image(self, i, input, recon):
        save_dir = os.path.join(self.save_dir, 'mnist')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        images = torch.cat((input[0:16], recon[0:16]))
        save_image(images, os.path.join(save_dir, str(i)+'.png'))


    def loss_function(self, *args, M_N):
        # self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = M_N # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input, reduction='sum')
        # recons_loss = F.binary_cross_entropy(F.sigmoid(recons), input, reduction='sum')

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

