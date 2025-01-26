import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu')
        # self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.save_dir = opt.checkpoints_dir
        self.reconModel = opt.model_name
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.parallel_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0         # used for learning rate policy 'plateau'

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        # load_suffix = opt.epoch
        # self.load_networks(load_suffix)
        self.load_networks()
        # self.print_networks(opt.verbose)      # params: 24M

    def parallelize(self, convert_sync_batchnorm=True):
        if not self.opt.use_ddp:
            for name in self.parallel_names:
                if isinstance(name, str):
                    module = getattr(self, name)
                    setattr(self, name, module.to(self.device))
        else:
            for name in self.model_names:
                if isinstance(name, str):
                    module = getattr(self, name)
                    if convert_sync_batchnorm:
                        module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
                    setattr(self, name, torch.nn.parallel.DistributedDataParallel(module.to(self.device),
                                                                                  device_ids=[self.device.index],
                                                                                  find_unused_parameters=True,
                                                                                  broadcast_buffers=True))

            # DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.
            for name in self.parallel_names:
                if isinstance(name, str) and name not in self.model_names:
                    module = getattr(self, name)
                    setattr(self, name, module.to(self.device))

    def eval(self):
        """Make models eval mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def test(self):
        with torch.no_grad():
            self.forward()
            # self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)[:, :3, ...]
        return visual_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        save_filename = 'epoch_%s.pth' % (epoch)
        save_path = os.path.join(self.save_dir, save_filename)

        save_dict = {}
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel) or isinstance(net,
                                                                        torch.nn.parallel.DistributedDataParallel):
                    net = net.module
                save_dict[name] = net.state_dict()

        for i, optim in enumerate(self.optimizers):
            save_dict['opt_%02d' % i] = optim.state_dict()

        for i, sched in enumerate(self.schedulers):
            save_dict['sched_%02d' % i] = sched.state_dict()

        torch.save(save_dict, save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # def load_networks(self, epoch):
    def load_networks(self):
        """
        加载模型：模型文件在./checkpoints/内
        :return: None
        """
        # load_dir = self.save_dir
        # load_filename = 'epoch_%s.pth' % epoch
        # load_path = os.path.join(load_dir, load_filename)
        load_path = os.path.join(self.save_dir, self.reconModel)
        state_dict = torch.load(load_path, map_location=self.device)
        # print('loading the model from %s' % load_path)

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                net.load_state_dict(state_dict[name])

    def print_networks(self, verbose):
        """
        打印网络结构和参数
        :param verbose: True or False
        :return: None
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
