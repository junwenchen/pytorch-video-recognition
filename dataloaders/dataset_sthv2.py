def make_dataset_sth(split_file, split, root, num_classes=157):
    dataset = []
    #print os.path.join(root,split_file)
    #with open(os.path.join(root,split_file), 'r') as f:
    print(split_file+split+'.json')
    with open(split_file+split+'.json', 'r') as f:
        data = json.load(f)
    with open(split_file+'labels.json', 'r') as f:
        labels = json.load(f)
    num_labels = len(labels)
    #print(num_labels)
    for vid in data:
        vid_index = int(labels[vid['template'].replace('[','').replace(']','')])
        #print('vid_index',vid_index)
        #num_frames = len(os.listdir(os.path.join(root, vid['id'])))
        #labels_index = np.zeros((num_labels, num_frames), np.float32)
        labels_index = np.zeros(num_labels, np.float32)
        #labels_index[vid_index,:] = 1
        labels_index[vid_index] = 1
        #print(labels_index)
        dataset.append((vid['id'],labels_index))
        #dataset.append((vid['id'], np.array(int(labels[vid['template'].replace('[','').replace(']','')]))))
    return dataset

def load_rgb_frames_sth(image_dir, vid):
    frames = []
    for item in os.listdir(os.path.join(image_dir, vid)):
        img = cv2.imread(os.path.join(image_dir, vid, item))[:, :, [2, 1, 0]]
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


class Somethingv2():

    def __init__(self, split_file, split, root, mode, transforms=None):

        self.data = make_dataset_sth(split_file,split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root

    def __getitem__(self, index):

        vid, label= self.data[index]
        # print("vid is",vid)
        # print("label is", label)
        # print(type(label))
        if self.mode == 'rgb':
            imgs = load_rgb_frames_sth(self.root, vid)
        else:
            imgs = load_flow_frames(self.root, vid)

        #print((len(imgs),len(imgs[0]),len(imgs[0][0]),len(imgs[0][0][0])))

        imgs = self.transforms(imgs)  #from input to 224
        #print((len(imgs),len(imgs[0])))
        #print(label)

        return video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)
