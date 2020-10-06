import pandas as pd
import numpy as np
import warnings
import os

from torch.utils.data.dataset import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class CustomDatasetFromImages(Dataset):
    def __init__(self, fine_data, coarse_data, transform):
        """
        Args:
            fine_data (list): list of fine evaluation results [source img name, patch path, precision, recall,
            average precision, mean of loss, counts of object]
            coarse_data (list): list of coarse evaluation results [source img name, patch path, precision, recall,
            average precision, mean of loss, counts of object]
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform

        # [source img name, patch path, precision, recall, average precision, mean of loss, counts of object]
        self.fine_data = fine_data
        self.fine_data.sort(key=lambda element: element[1])
        self.coarse_data = coarse_data
        self.coarse_data.sort(key=lambda element: element[1])

        # Calculate len
        if len(self.fine_data) == len(self.coarse_data):
            self.data_len = len(self.fine_data) / 4

        # Second column is the image paths
        # self.image_arr = np.asarray(data_info.iloc[:, 1])
        # First column is the image IDs
        # self.label_arr = np.asarray(data_info.iloc[:, 0])

    def __getitem__(self, index):
        index = index * 4
        source_path = os.sep.join(self.fine_data[index][1].split(os.sep)[:-4])
        source_path = os.path.join(source_path, self.fine_data[index][1].split(os.sep)[-3], 'images', self.fine_data[index][0]) + '.jpg'
        # print('\ncomplete_source_path', source_path)

        img_as_img = Image.open(source_path)

        # Transform the image
        img_as_tensor = self.transforms(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        # single_image_label = self.label_arr[index]

        f_p, c_p = [], []
        f_r, c_r = [], []
        f_ap, c_ap = [], []
        f_loss, c_loss = [], []
        f_ob, c_ob = [], []
        f_stats, c_stats = [], []

        target_dict = dict()

        for i in range(4):

            f_p.append(self.fine_data[index+i][2])
            c_p.append(self.coarse_data[index + i][2])

            f_r.append(self.fine_data[index + i][3])
            c_r.append(self.coarse_data[index + i][3])

            f_ap.append(self.fine_data[index + i][4])
            c_ap.append(self.coarse_data[index + i][4])

            f_loss.append(self.fine_data[index + i][5])
            c_loss.append(self.coarse_data[index + i][5])

            f_ob.append(self.fine_data[index + i][6])
            c_ob.append(self.coarse_data[index + i][6])

            f_stats.append(self.fine_data[index+i][7])
            c_stats.append(self.coarse_data[index+i][7])

        target_dict['f_p'] = f_p
        target_dict['c_p'] = c_p
        target_dict['f_r'] = f_r
        target_dict['c_r'] = c_r
        target_dict['f_ap'] = f_ap
        target_dict['c_ap'] = c_ap
        target_dict['f_loss'] = f_loss
        target_dict['c_loss'] = c_loss
        target_dict['f_ob'] = f_ob
        target_dict['c_ob'] = c_ob
        target_dict['f_stats'] = f_stats
        target_dict['c_stats'] = c_stats

        return img_as_tensor, target_dict

    def __len__(self):
        return int(self.data_len)


class CustomDatasetFromImages_timetest(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        data_info = pd.read_csv(csv_path, header=None)
        # Second column is the image paths
        self.image_arr = np.asarray(data_info.iloc[:, 1])
        # First column is the image IDs
        self.label_arr = np.asarray(data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(data_info)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index] + '.jpg'
        # Open image
        img_as_img = Image.open(single_image_name.replace('/media/data2/dataset', '/home'))
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_name)

    def __len__(self):
        return self.data_len