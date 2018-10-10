class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            # root_dir = '/data/dataset/ucf101/UCF-101/'
            root_dir = '/data/dataset/ucf101/UCF-5/'
            # Save preprocess data into output_dir
            output_dir = '/data/dataset/VAR/UCF-5/'

            bbox_output_dir = '/data/dataset/UCF-101-result/UCF-5-20/'

            return root_dir, output_dir, bbox_output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/Path/to/hmdb-51'

            output_dir = '/path/to/VAR/hmdb51'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return '/path/to/Models/c3d-pretrained.pth'
