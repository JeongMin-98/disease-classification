import os
import time

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import utils.tools
from utils.dataLoader import download_mnist_data
from utils.dataLoader import ImageDataset
from utils.tools import check_folder, count_parameters, find_latest_ckpt, check_device, cross_entroy_loss
from utils.tools import requires_grad, apply_gradients
from utils.tools import parse_model_config, accuracy
from utils.tools import visualize_feature_map
from network.exampleNet import Net

from torchinfo import summary
from tqdm import tqdm


def check_model_build(args):
    device = check_device()
    model = DeepNetwork(args)
    model.build_model(device)

    model_stats = summary(model.network,
                          input_size=(1, 3, 224, 224),
                          device="cuda",
                          verbose=1,
                          col_width=16,
                          col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
                          row_settings=["var_names"])


def run_fn(args):
    device = check_device()
    model = DeepNetwork(args)
    model.build_model(device)

    model_stats = summary(model.network,
                          input_size=(1, 3, 224, 224),
                          device="cuda",
                          verbose=1,
                          col_width=16,
                          col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
                          row_settings=["var_names"])

    if args['phase'] == "train":
        model.train_model(device)
    if args['phase'] == "test":
        model.test_model(device)


def run_visualize_feature_map_func(args):
    device = check_device()
    model = DeepNetwork(args)
    model.build_model(device)

    for idx, (img, label) in enumerate(model.train_loader):
        visualize_feature_map(model, img, label)
        if idx == 10:
            break


class DeepNetwork():
    def __init__(self, args):
        super(DeepNetwork, self).__init__()
        self.model_name = args['model_name']
        self.checkpoint_dir = args['checkpoint_dir']
        self.result_dir = args['result_dir']
        self.log_dir = args['log_dir']
        self.sample_dir = args['sample_dir']
        self.config_dir = args['config_dir']
        self.dataset_name = args['dataset']

        """ Network parameters """
        self.feature_size = args['feature_size']

        """ Network architecture from config """
        self.config_dir = args['config_dir']

        """ Training parameters """
        self.train_size = args['train_size']
        self.lr = args['lr']
        self.iteration = args['iteration']
        self.img_size = args['img_size']
        self.batch_size = args['batch_size']
        # FIX global_batch_size
        self.global_batch_size = self.batch_size

        """ Misc """
        # self.save_freq = args['save_freq']
        self.log_template = 'step [{}/{}]: loss: {:.3f}'
        self.acc_log_template = 'step [{}/{}]: accuracy: {:.3f}'

        """ Directory """
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        check_folder(self.checkpoint_dir)
        self.log_dir = os.path.join(self.log_dir, self.model_dir)
        check_folder(self.log_dir)
        self.config_dir = os.path.join(self.config_dir, self.model_name)
        check_folder(self.config_dir)

        # load config file
        config_path = os.path.join(self.config_dir, self.model_name + ".cfg")
        self.cfg = parse_model_config(config_path)
        check_folder(config_path)

        """ Dataset """
        dataset_path = './dataset'
        self.dataset_path = os.path.join(dataset_path, self.dataset_name)

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self, device):
        # train_data = download_mnist_data(train=True)
        train_data = utils.dataLoader.load_imagenet(root=self.dataset_path, train=True)
        self.dataset_num = train_data.__len__()
        # split the dataset into training and validation sets.
        train_size = int(self.train_size * self.dataset_num)
        val_size = self.dataset_num - train_size

        train_data, val_data = random_split(train_data, [train_size, val_size])

        """ Load dataset"""
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.trainning_set_iter = iter(self.train_loader)

        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
        self.validation_set_iter = iter(self.val_loader)

        """ Network """
        self.network = Net(config=self.cfg).to(device)

        """ Optimizer """
        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        """ Checkpoint """
        latest_ckpt_name, start_iter = find_latest_ckpt(self.checkpoint_dir)

        if latest_ckpt_name is not None:
            # "rank == 0" means a first gpu.
            print('Latest checkpoint restored!! ', latest_ckpt_name)
            print('start iteration : ', start_iter)
            self.start_iteration = start_iter

            latest_ckpt = os.path.join(self.checkpoint_dir, latest_ckpt_name)
            ckpt = torch.load(latest_ckpt, map_location=device)

            self.network.load_state_dict(ckpt["network"])
            self.optim.load_state_dict(ckpt["optim"])
        else:
            self.start_iteration = 0

    def train_step(self, real_images, label, device=torch.device('cuda')):
        # gradient check
        requires_grad(self.network, True)

        # forward pass
        logit = self.network(real_images)
        # loss
        loss = cross_entroy_loss(logit, label)

        # backword
        apply_gradients(loss, self.optim)

        return logit, loss

    def test_model(self, device, val_loader=None, valid=False):

        phase = "Test"
        if valid:
            phase = "Validation"
        print()
        print("=======================================")
        print("phase : {} ".format(phase))

        if valid is False:
            test_data = download_mnist_data(train=False)
            eval_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        else:
            eval_loader = val_loader
        self.network.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(eval_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.network(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += accuracy(outputs, labels)

                images = images.cpu().numpy()
                predicted = predicted.cpu().numpy()

                if valid is False:
                    utils.tools.visualize_inference(images, predicted, batch_size=self.batch_size)

        acc = 100 * correct / total
        print() if valid else print(self.acc_log_template.format(1, 1, acc))
        return acc

    def train_model(self, device):
        # setup tensorboard
        train_summary_writer = SummaryWriter(self.log_dir)

        # start training
        print()
        print(self.dataset_path)
        print("Dataset number : ", self.dataset_num)
        print("Training set ratio : ", self.train_size)
        print("Each batch size : ", self.batch_size)
        print("Target image size : ", self.img_size)
        # print("Save frequency : ", self.save_freq)
        print("PyTorch Version :", torch.__version__)
        print('max_steps: {}'.format(self.iteration))
        print()

        print("=======================================")
        print("Phase : training")
        train_loss_list = []
        best_loss = float("inf")
        # number of self.dataset_iter
        iter_per_epoch = max(int(self.dataset_num * self.train_size // self.batch_size), 1)
        print(f"iter_per_epoch: {iter_per_epoch}")
        epoch = 0
        current_progress = 0
        progress = tqdm(total=iter_per_epoch, desc="Iteration per epoch")
        for idx in range(self.start_iteration, self.iteration):

            if idx == 0:
                print("=======================================")
                print("count params")
                n_params = count_parameters(self.network)
                print("network parameters : ", format(n_params, ','))
                print("=======================================")

            if idx % iter_per_epoch == 0:
                if idx > 0:
                    epoch += 1
                    print("=======================================")
                    train_loss = sum(train_loss_list) / iter_per_epoch
                    print("train Loss " + self.log_template.format(epoch, self.iteration // iter_per_epoch, train_loss))
                    train_summary_writer.add_scalar("train_loss", train_loss, idx)
                    print()

                    # validation
                    val_acc = self.test_model(device, self.val_loader, True)
                    print("val Acc " + self.acc_log_template.format(epoch, self.iteration // iter_per_epoch, val_acc))
                    train_summary_writer.add_scalar("val_acc", val_acc, idx)

                    """ Each epoch, Save model """
                    self.torch_save(idx)

                    loss = 0
                    train_loss_list = []
                    progress.reset(iter_per_epoch)

                self.trainning_set_iter = iter(self.train_loader)

            real_img, label = next(self.trainning_set_iter)
            real_img = real_img.to(device)
            label = label.to(device)

            logit, loss = self.train_step(real_img, label, device=device)
            # acc = accuracy(logit, label)
            train_loss_list.append(loss)
            train_summary_writer.add_scalar('loss', loss, global_step=idx)

            time.sleep(0.1)

            # update progress
            current_progress += 1
            progress.update(1)

        print("=======================================")

    def torch_save(self, idx):
        print()
        print("Saving model")
        print("path : " + os.path.join(self.checkpoint_dir, "iter_{}.pt".format(idx)))
        torch.save(
            {
                'network': self.network.state_dict(),
                'optim': self.optim.state_dict()
            },
            os.path.join(self.checkpoint_dir, 'iter_{}.pt'.format(idx))
        )

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.img_size)
