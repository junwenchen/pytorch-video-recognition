import cv2
import os

#image_dir = "/data/dataset/VAR/UCF-101/train"
#result_dir = "/data/dataset/ucf101/UCF-101_det_vis/"
#bbox_dir = "/data/dataset/UCF-101-result/UCF-101-20/"

#image_dir = "/data/dataset/something-somthing-v2/20bn-something-something-v2-frames-224/"
#result_dir = "/data/dataset/something-somthing-v2/20bn-something-something-224-20_det_vis/"
#bbox_dir = "/data/dataset/something-somthing-v2/20bn-something-something-224-20"

image_dir = '/data/dataset/something-something-v2-lite/20bn-something-something-v2-frames/'
result_dir = '/data/dataset/something-something-v2-lite/20bn-something-something-det-vis'
bbox_dir = '/data/dataset/something-something-v2-lite/20bn-something-something-det'

# image_dir = '/data/dataset/volleyball/videos/'
# result_dir = '/data/dataset/volleyball/volleyball_det_vis'
# bbox_dir = '/data/dataset/volleyball/volleyball-20'
# for label in os.listdir(image_dir):
#     if not os.path.exists(os.path.join(result_dir, label)):
#         os.makedirs(os.path.join(result_dir, label))
for frames in os.listdir(os.path.join(image_dir)):
    if not os.path.exists(os.path.join(result_dir, frames)):
        os.makedirs(os.path.join(result_dir, frames))
   # print(frames)
    for img_name in os.listdir(os.path.join(image_dir, frames)):
        #print(img_name)
        result_path = os.path.join(result_dir, frames, img_name)
        im_file = os.path.join(image_dir, frames,img_name)
        print('im_file', im_file)
        img = cv2.imread(im_file)
        print(img.shape)
        height = img.shape[0]
        width = img.shape[1]
        with open(os.path.join(bbox_dir, frames,img_name[:-4]+'_det.txt'),"r") as f:
            lines = f.readlines()
            for line in lines:
                bbox = [float(x) for x in line.strip().split(" ")]
                print(bbox)
                img = cv2.rectangle(img, (int(bbox[1]), int(bbox[2])), (int(bbox[3]), \
                int(bbox[4])), (0, 204, 0), 2)
        print(result_path)
        # cv2.imwrite(result_path, img)
        cv2.imshow(frames+img_name,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
