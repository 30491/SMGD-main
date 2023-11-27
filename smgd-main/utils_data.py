import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import pickle

class SubsetImageNet(Dataset):  #
    def __init__(self, root, class_to_idx='./imagenet_class_to_idx.npy', transform=None):  # root 图像路径地址
    # def __init__(self, root, class_to_idx='./imagenet_class_to_idx.npy', transform=None):
        super(SubsetImageNet, self).__init__()
        self.root = root
        self.transform = transform
        img_path = os.listdir(root)
        img_path = sorted(img_path)
        self.img_path = [item for item in img_path if 'png' in item]
        self.class_to_idx = np.load(class_to_idx, allow_pickle=True)[()]

    def __getitem__(self, item):
        filepath = os.path.join(self.root, self.img_path[item])
        sample = Image.open(filepath, mode='r')

        if self.transform:
            sample = self.transform(sample)

        class_name = self.img_path[item].split('_')[0]
        label = self.class_to_idx[class_name]

        return sample, label, item

    def __len__(self):
        return len(self.img_path)


def save_images(images, img_list, idx, output_dir):
    """Saves images to the output directory.
        Args:
          images: tensor with minibatch of images
          img_list: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
          output_dir: directory where to save images
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i, sample_idx in enumerate(idx.numpy()):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        filename = img_list[sample_idx]
        cur_images = (images[i, :, :, :].transpose(1, 2, 0) * 255).astype(np.uint8)

        im = Image.fromarray(cur_images)
        im.save('{}.png'.format(os.path.join(output_dir, filename)))

def save_images_49000(images, img_list, file_list,  output_dir, batch_idx, batch_size):
    """Saves images to the output directory.
        Args:
          images: tensor with minibatch of images
          img_list: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
          output_dir: directory where to save images
    """
    # output_dir = output_dir + '/' +
    x = np.arange(batch_size)
    idx = x
    # idx = np.array([0,1,2,3,4,5,6,7,8,9,10,11])

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    sample_batch_idx = 0
    for i in idx:
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        # print('##################')
        # print(i)
        # print(sample_idx)
        # exit()
        sample_idx = sample_batch_idx + batch_idx * images.shape[0]
        filename = file_list[sample_idx]
        file_up_name = img_list[sample_idx]

        output_dir1 = output_dir + '/' + file_up_name

        if not os.path.exists(output_dir1):
            os.mkdir(output_dir1)
        cur_images = (images[i, :, :, :].transpose(1, 2, 0) * 255).astype(np.uint8)
        sample_batch_idx = sample_batch_idx + 1
        im = Image.fromarray(cur_images)

        final_path = os.path.join(output_dir1, filename)
        im.save(final_path)

class SubsetImageNet1(Dataset):  #
    def __init__(self, root, class_to_idx='./imagenet_class_to_idx.npy', transform=None):  # root 图像路径地址
        super(SubsetImageNet1, self).__init__()
        self.root = root
        self.transform = transform
        img_path = os.listdir(root)
        img_path = sorted(img_path)
        self.img_path = [item for item in img_path if 'png' in item]
        self.class_to_idx = np.load(class_to_idx, allow_pickle=True)[()]

    def __getitem__(self, item):
        filepath = os.path.join(self.root, self.img_path[item])
        sample = Image.open(filepath, mode='r')

        if self.transform:
            sample = self.transform(sample)

        class_name = self.img_path[item].split('_')[0]
        label = self.class_to_idx[class_name]

        return sample, label, item

    def __len__(self):
        return len(self.img_path)

def generate_data_pickle():
    all_image = dict()
    label_dict = dict()

    with open('imagenet_class_index.json') as f:
        data = json.load(f)
        for i in data.keys():
            label_dict[data[i][0]] = int(i)

    for root, dirs, files in os.walk('../val5000'):
        for name in dirs:
            dir_name = os.path.join(root, name)
            image_list = []
            for root_, dirs_, files_ in os.walk(dir_name):
                for name_ in files_:
                    obj = Image.open(os.path.join(root_, name_))
                    obj = obj.convert('RGB')
                    image_list.append(obj)
            all_image[label_dict[name]] = image_list

    with open('data.pickle', 'wb') as f:
        pickle.dump(all_image, f, pickle.HIGHEST_PROTOCOL)