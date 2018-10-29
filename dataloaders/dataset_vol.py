import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from network.model.graph_front.graphFront import _graphFront


class VolleyballDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset='volleyball', split='train', clip_len=16, preprocess=False):
        # self.root_dir, self.output_dir, self.bbox_output_dir = root_dir, output_dir, bbox_output_dir
        self.root_dir, self.bbox_output_dir = Path.db_dir(dataset)
        # dic ={'train': '1 3 6 7 10 13 15 16 18 22 23 31 32 36 38 39 40 41 42 48 50 52 53 54', \
        dic ={'train': '38 39 40 41 42 48 50 52 53 54', \
                'val': '0 2 8 12 17 19 24 26 27 28 30 33 46 49 51', \
                'test': '4 5 9 11 14 20 21 25 29 34 35 37 43 44 45 47'}

        label_index = {'r_set': 0, 'r_spike': 1, 'r-pass': 2, 'r_winpoint': 3, 'l_winpoint': 4, \
        'l-pass': 5, 'l-spike': 6, 'l_set': 7}
        video_index = dic[split].split(' ')
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128#256#112#780
        self.resize_width = 192#384#112#1280

        self.fnames, self.labels, self.bboxes = self.make_dataset_sth(video_index, label_index)
        self.graph = _graphFront()


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        labels = np.array(self.labels[index])
        buffer, buffer_bbox = self.load_frames(self.fnames[index], self.bboxes[index])
        # buffer, buffer_bbox = self.crop(buffer, buffer_bbox, self.clip_len, self.crop_size)
        adjacent_matrix = self.graph.build_graph(buffer_bbox[::2,:,:])

        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)

        # return torch.from_numpy(buffer), torch.from_numpy(labels), torch.from_numpy(buffer_bbox)
        return torch.from_numpy(buffer), torch.from_numpy(buffer_bbox), \
        torch.from_numpy(labels), adjacent_matrix

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def make_dataset_sth(self, video_index, label_index):
        frame_name = []
        frame_label = []
        frame_bbox = []
        for video in video_index:
            with open(os.path.join(self.root_dir, video, 'annotations.txt'),'r') as f:
                info = f.readlines()
                for item in info:
                    item_index = item.split(' ')
                    frame_name.append(os.path.join(self.root_dir, video, \
                    item.split(' ')[0][:-4]))
                    frame_label.append(label_index[item_index[1]])
                    frame_bbox.append(os.path.join(self.bbox_output_dir, video, \
                    item.split(' ')[0][:-4], 'person_detections.txt'))

        return frame_name, frame_label, frame_bbox

    def load_frames(self, file_dir, bbox_dir):
        with open(bbox_dir, 'r') as f:
            det_lines = f.readlines()

        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)

        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), \
        np.dtype('float32'))
        buffer_bbox = np.empty((frame_count, 12, 5), np.dtype('float32'))

        person_index = 0
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            im_info = frame.shape
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            buffer[i] = frame
            det_index = det_lines[person_index].strip().split('\t')
            if frame_name.strip().split('/') != det_index[0]:
                continue
            person_index += 1
            for j in range(min(12,int(det_index[1]))):
                buffer_bbox[i][j][1:] = [float(x) for x in det_index[(2+j*6):(2+j*6)+4]]
                buffer_bbox[i][j][1] = buffer_bbox[i][j][1]/im_info[2] * self.resize_width
                buffer_bbox[i][j][2] = buffer_bbox[i][j][2]/im_info[1] * self.resize_height
                buffer_bbox[i][j][3] = buffer_bbox[i][j][3]/im_info[2] * self.resize_width
                buffer_bbox[i][j][4] = buffer_bbox[i][j][4]/im_info[1] * self.resize_height

        return buffer, buffer_bbox

    def crop(self, buffer, buffer_bbox, clip_len, crop_size):
        # randomly select time index for temporal jittering
        # time_index = np.random.randint(buffer.shape[0] - clip_len)
        #
        # # Randomly select start indices in order to crop the video
        # height_index = np.random.randint(buffer.shape[1] - crop_size)
        # width_index = np.random.randint(buffer.shape[2] - crop_size)

        time_index = 0
        # Randomly select start indices in order to crop the video
        height_index = 0
        width_index = 0
        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames

        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        buffer_bbox = buffer_bbox[time_index:time_index + clip_len, :]

        return buffer, buffer_bbox





if __name__ == "__main__":
    from torch.utils.data import DataLoader
    root_dir = '/data/dataset/volleyball/videos/'
    train_data = VideoDataset(dataset='volleyball', split='test', clip_len=8, preprocess=True)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break
