import os
# from sklearn.model_selection import train_test_split
import torch
import cv2
import numpy as np
import json
from torch.utils.data import Dataset
from mypath import Path
from network.model.graph_front.graphFront import _graphFront

class VideoDataset(Dataset):
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
    # def __init__(self, root_dir, output_dir, bbox_output_dir, dataset='ucf101', split='train', \
    #      clip_len=16, preprocess=False):

    def __init__(self, dataset='ucf101', split='train', clip_len=16, transforms=None):
        self.root_dir,split_dir = Path.db_dir(dataset)
        # self.root_dir, self.output_dir, self.bbox_output_dir = root_dir, output_dir, \
        #     bbox_output_dir
        self.fnames, self.labels  = self.make_dataset_sth(split_dir, split)
        self.transforms = transforms
        #folder = os.path.join(self.output_dir, split)
        #bbox_folder = self.bbox_output_dir
        self.clip_len = clip_len
        self.split = split
        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 256
        self.resize_width = 256
        self.crop_size = 256
        # self.graph = _graphFront()

    def __getitem__(self, index):
        # print(self.fnames[index])
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        # adjacent_matrix = self.graph.build_graph(buffer_bbox[::2,:,:])
        #labels = np.array(self.label_array[index])
        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        label = np.array(self.labels[index])
        return torch.from_numpy(buffer), torch.from_numpy(label)
        # return torch.from_numpy(buffer), torch.from_numpy(buffer_bbox), torch.from_numpy(labels), adjacent_matrix

        # vid, label= self.data[index]
        # imgs = load_rgb_frames_sth(self.root, vid)
        # imgs = self.transforms(imgs)  #from input to 224
        # return video_to_tensor(imgs), torch.from_numpy(label)

    # def __len__(self):
    #     return len(self.data)

    def __len__(self):
        return len(self.fnames)

    # def __getitem__(self, index):
    #     # Loading and preprocessing.
    #     buffer, buffer_bbox = self.load_frames(self.fnames[index], self.bbox_output_dir)
    #     buffer, buffer_bbox = self.crop(buffer, buffer_bbox, self.clip_len, self.crop_size)
    #     labels = self.load_labels()
    #     # adjacent_matrix = self.graph.build_graph(buffer_bbox[:8,:,:])
    #     # labels = np.array(self.label_array[index])
    #     if self.split == 'test':
    #         # Perform data augmentation
    #         buffer = self.randomflip(buffer)
    #     buffer = self.normalize(buffer)
    #     buffer = self.to_tensor(buffer)
    #     return torch.from_numpy(buffer), torch.from_numpy(buffer_bbox), torch.from_numpy(labels), \
    #     adjacent_matrix

    def make_dataset_sth(self, split_file, split, num_classes=157):
        with open(split_file+split+'.json', 'r') as f:
            data = json.load(f)
        frame_name = []
        frame_label = []
        if split == 'test':
            for vid in data:
                frame_name.append(os.path.join(self.root_dir, vid['id']))
            return frame_name, frame_label
        with open(split_file+'labels.json', 'r') as f:
            labels = json.load(f)
        num_labels = len(labels)
        #frame_name = []
        #frame_label = []
        for vid in data:
            #print(vid['template'])
            vid_index = int(labels[vid['template'].replace('[','').replace(']','')])
            #num_frames = len(os.listdir(os.path.join(root, vid['id'])))
            #labels_index = np.zeros((num_labels, num_frames), np.float32)
            frame_name.append(os.path.join(self.root_dir, vid['id']))
            frame_label.append(vid_index)
            # labels_index = np.zeros(num_labels, np.float32)
            # labels_index[vid_index] = 1
            # dataset.append((vid['id'],labels_index))
        return frame_name, frame_label

    # def check_integrity(self):
    #     if not os.path.exists(self.root_dir):
    #         return False
    #     else:
    #         return True
    #
    # def check_preprocess(self):
    #     # TODO: Check image size in output_dir
    #     if not os.path.exists(self.output_dir):
    #         return False
    #     if not os.path.exists(os.path.join(self.output_dir, 'train')):
    #         return False
    #
    #     for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
    #         for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
    #             video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
    #                                 sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
    #             image = cv2.imread(video_name)
    #             #if np.shape(image)[0] != 112 or np.shape(image)[1] != 112:
    #             if np.shape(image)[0] != 112 or np.shape(image)[1] != 112:
    #                 return False
    #             else:
    #                 break
    #
    #         if ii == 10:
    #             break
    #
    #     return True
    #
    # def read_json(self):
    #     with open(split_file+split+'.json', 'r') as f:
    #         data = json.load(f)
    #
    #         for video_file in os.listdir(self.root_dir)):
    #             for fname in os.listdir(os.path.join(folder, label)):
    #                 self.fnames.append(os.path.join(folder, label, fname))
    #                 labels.append(label)
    #
    # def preprocess(self):
    #     if not os.path.exists(self.output_dir):
    #         os.mkdir(self.output_dir)
    #         os.mkdir(os.path.join(self.output_dir, 'train'))
    #         os.mkdir(os.path.join(self.output_dir, 'val'))
    #         os.mkdir(os.path.join(self.output_dir, 'test'))
    #
    #     # file_path = os.path.join(self.root_dir, file)
    #     video_files = [name for name in os.listdir(self.root_dir)]
    #     train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
    #     train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)
    #
    #     # for file in os.listdir(self.root_dir):
    #     train_dir = os.path.join(self.output_dir, 'train')
    #     val_dir = os.path.join(self.output_dir, 'val')
    #     test_dir = os.path.join(self.output_dir, 'test')
    #     for video in train:
    #         self.process_video(video, train_dir)
    #     for video in val:
    #         self.process_video(video, val_dir)
    #     for video in test:
    #         self.process_video(video, test_dir)
    #     print('Preprocessing finished.')
    #
    #
    # def process_video(self, video_filename, save_dir):
    #     # Initialize a VideoCapture object to read video data into a numpy array
    #     if not os.path.exists(os.path.join(save_dir, video_filename)):
    #         os.mkdir(os.path.join(save_dir, video_filename))
    #
    #     img_list = os.listdir(os.path.join(self.root_dir, video_filename))
    #     for item in img_list:
    #         img = cv2.imread(os.path.join(self.root_dir, video_filename, item))
    #         img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    #         cv2.imwrite(filename=os.path.join(save_dir, video_filename, item), img=img)
    #
    #
    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer
    #
    #
    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            # img_index = frame_name.split('/')[-1][:-4]
            # frame_index = frame_name.split('/')[-3:-1]
            frame = np.array(cv2.resize(cv2.imread(frame_name),(self.resize_height, self.resize_width)).astype(np.float64))
            # with open(os.path.join(bbox_file_path, frame_index[0],frame_index[1], img_index + '.jpg_det.txt'), 'r') as f:
            #     bboxes = f.readlines()
            #     for j in range(len(bboxes)):
            #         buffer_bbox[i][j][:] = bboxes[j].strip().split(' ')
            buffer[i] = frame
        # return buffer, buffer_bbox
        return buffer

    def crop(self, buffer, clip_len, crop_size):
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

        # buffer_bbox = buffer_bbox[time_index:time_index + clip_len, :]

        return buffer





if __name__ == "__main__":

    root_dir = '/data/dataset/something-somthing-v2/20bn-something-something-v2-frames/'
    output_dir = '/data/dataset/something-somthing-v2/20bn-something-something-v2-frames-224/'
    bbox_output_dir = '/data/dataset/something-somthing-v2/20bn-something-something-v2-frames-224-20/'
    split_dir = '/data/dataset/something-somthing-v2/label/something-something-v2-'

    from torch.utils.data import DataLoader
    train_data = VideoDataset(root_dir, output_dir, bbox_output_dir, split_dir, dataset='somthing', \
    split='train', clip_len=8)
    print(train_data.fnames)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    # for i, sample in enumerate(train_loader):
    #     inputs = sample[0]
    #     labels = sample[1]
    #     print(inputs.size())
    #     print(labels)
    #
    #     if i == 1:
    #         break
