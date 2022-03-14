import os.path
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
import glob
from data.base_dataset import BaseDataset
from models.sne_model import SNE


class R2Ddataset(BaseDataset):
    """dataloader for kitti dataset"""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.root = opt.dataroot  # path for the dataset
        self.use_sne = opt.use_sne
        self.num_labels = 2
        self.use_size = (opt.useWidth, opt.useHeight)
        if self.use_sne:
            self.sne_model = SNE()

        if opt.phase == "train":
            self.image_list = sorted(glob.glob(os.path.join(
                self.root, 'Town02/day_clear/0', 'depth/left', '*.png')))
        elif opt.phase == "val":
            self.image_list = sorted(glob.glob(os.path.join(
                self.root, 'Town02/day_clear/0', 'depth/left', '*.png')))
        else:
            self.image_list = sorted(glob.glob(os.path.join(
                self.root, 'Town02/day_clear/0', 'depth/left', '*.png')))

    def __getitem__(self, index):
        useDir = "/".join(self.image_list[index].split('/')[:-3])
        name = self.image_list[index].split('/')[-1]

        rgb_image = cv2.cvtColor(cv2.imread(os.path.join(
            useDir, 'rgb/left', name[:-4]+'.jpg')), cv2.COLOR_BGR2RGB)
        depth_image = cv2.cvtColor(cv2.imread(os.path.join(
            useDir, 'depth/left', name)), cv2.COLOR_BGR2RGB).astype(np.float32)
        distance = (depth_image[:, :, 0] + depth_image[:, :, 1] * 256 +
                    depth_image[:, :, 2] * 256 * 256) / (256 * 256 * 256 - 1) * 1000    # meter
        distance[distance > 999] = 0
        oriHeight, oriWidth, _ = rgb_image.shape
        if self.opt.phase == 'test' and self.opt.no_label:
            # Since we have no gt label (e.g., kitti submission), we generate pseudo gt labels to
            # avoid destroying the code architecture
            label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
        else:
            label_image = cv2.cvtColor(cv2.imread(os.path.join(
                useDir, 'semseg/left', name)), cv2.COLOR_BGR2RGB)
            label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
            label[np.logical_or(label_image[:, :, 0] == 6,
                                label_image[:, :, 0] == 7)] = 1

        # resize image to enable sizes divide 32
        rgb_image = cv2.resize(rgb_image, self.use_size)
        label = cv2.resize(label, self.use_size,
                           interpolation=cv2.INTER_NEAREST)

        # another_image will be normal when using SNE, otherwise will be depth
        if self.use_sne:
            camParam = torch.tensor([[320, 0, 320],
                                     [0, 320, 240],
                                     [0, 0, 1]], dtype=torch.float32)
            normal = self.sne_model(torch.tensor(
                distance.astype(np.float32)), camParam)
            another_image = normal.cpu().numpy()
            another_image = np.transpose(another_image, [1, 2, 0])
            another_image = cv2.resize(another_image, self.use_size)
        else:
            another_image = distance.astype(np.float32) * 1000 / 65535
            another_image = cv2.resize(another_image, self.use_size)
            another_image = another_image[:, :, np.newaxis]

        label[label > 0] = 1
        rgb_image = rgb_image.astype(np.float32) / 255

        rgb_image = transforms.ToTensor()(rgb_image)
        another_image = transforms.ToTensor()(another_image)

        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)

        # return a dictionary containing useful information
        # input rgb images, another images and labels for training;
        # 'path': image name for saving predictions
        # 'oriSize': original image size for evaluating and saving predictions
        return {'rgb_image': rgb_image, 'another_image': another_image, 'label': label,
                'path': name, 'oriSize': (oriWidth, oriHeight)}

    def __len__(self):
        return len(self.image_list)

    def name(self):
        return 'R2D'
