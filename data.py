import os, sys
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import torch


#several data augumentation strategies
def cv_random_flip(imgs, labels, depths):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        for i, img in enumerate(imgs):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
            depths[i] = depths[i].transpose(Image.FLIP_LEFT_RIGHT)
            if labels[i] is not None:
                labels[i] = labels[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs, labels, depths

def randomCrop(imgs, labels, depths):
    border=30
    image_width = imgs[-1].size[0]
    image_height = imgs[-1].size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    for i, img in enumerate(imgs):
        imgs[i] = imgs[i].crop(random_region)
        depths[i] = depths[i].crop(random_region)
        if labels[i] is not None:
            labels[i] = labels[i].crop(random_region)
    return imgs, labels, depths

def randomRotation(imgs, labels, depths):
    rand_count = random.random()
    random_angle = np.random.randint(-15, 15)
    mode=Image.BICUBIC
    if rand_count>0.8:
        for i, img in enumerate(imgs):
            imgs[i] = imgs[i].rotate(random_angle, mode)
            depths[i] = depths[i].rotate(random_angle, mode)
            if labels[i] is not None:
                labels[i] = labels[i].rotate(random_angle, mode)
    return imgs, labels, depths

def colorEnhance(imgs):
    bright_intensity=random.randint(5, 15) / 10.0
    contrast_intensity = random.randint(5, 15) / 10.0
    color_intensity = random.randint(0, 20) / 10.0
    sharp_intensity = random.randint(0, 30) / 10.0
    for i, img in enumerate(imgs):
        imgs[i]=ImageEnhance.Brightness(imgs[i]).enhance(bright_intensity)
        imgs[i]=ImageEnhance.Contrast(imgs[i]).enhance(contrast_intensity)
        imgs[i]=ImageEnhance.Color(imgs[i]).enhance(color_intensity)
        imgs[i]=ImageEnhance.Sharpness(imgs[i]).enhance(sharp_intensity)
    return imgs



class SalObjDataset(data.Dataset):
    def __init__(self, data_root, subset, augmentation, interval, sample_rate, trainsize, baseMode):

        with open(os.path.join(data_root, subset + '.txt')) as f:
            lines = f.readlines()
            videolists = sorted([line.strip() for line in lines])

        self.filenames_gt = []
        self.filenames = []
        self.filenames_dep = []

        for video in videolists:
            # Create List for only labeled GT
            label_path = os.path.join(data_root, 'data', video, 'GT')
            filenames_gt_i = [os.path.join(label_path, f) for f in os.listdir(label_path)
                              if any(f.endswith(ext) for ext in ['.jpg', '.png'])]
            self.filenames_gt += sorted(filenames_gt_i)

            # Create List for only labeled Image seqs
            image_path = os.path.join(data_root, 'data', video, 'RGB')
            file_postfix = os.listdir(image_path)[-1][-4:]
            filenames_i = [os.path.join(image_path, f[:-4] + file_postfix) for f in os.listdir(label_path)
                           if any(f.endswith(ext) for ext in ['.jpg', '.png'])]
            self.filenames += sorted(filenames_i)

            # Create List for only labeled depth seqs
            depth_path = os.path.join(data_root, 'data', video, 'Depth')
            file_dep_postfix = os.listdir(depth_path)[-1][-4:]
            filenames_dep_i = [os.path.join(depth_path, f[:-4] + file_dep_postfix) for f in os.listdir(label_path)
                              if any(f.endswith(ext) for ext in ['.jpg', '.png'])]
            self.filenames_dep += sorted(filenames_dep_i)

        self.filenames_gt.sort()
        self.filenames.sort()
        self.filenames_dep.sort()
        self.size = len(self.filenames)


        self.trainsize = trainsize
        self.baseline_mode = baseMode
        self.augment = augmentation
        self.interval = interval
        self.sample_rate = sample_rate


        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])



    def __getitem__(self, index):
        file_path = self.filenames[index]
        file_path_gt = self.filenames_gt[index]
        file_path_dep = self.filenames_dep[index]

        labels = []
        images = []
        depths = []

        # Extracting images
        All_mem_size = self.interval[0] + 1     # i.e., win_size OR stm_queue_size + 1
        for i in reversed(range(-self.interval[1], All_mem_size * self.sample_rate, self.sample_rate)): # i = 3,2,1,0
            # Labels
            if not self.baseline_mode:
                abs_file_path_gt, new_file_path_gt = self.filename_from_index(file_path_gt, i)
            else:
                abs_file_path_gt, new_file_path_gt = self.filename_from_base(file_path_gt, i)

            if not os.path.exists(abs_file_path_gt):
                label = None
                new_file_path_gt = ""
                if i == self.interval[1]:
                    print(abs_file_path_gt, "does not exist !")
                    exit(1)
            else:
                label = self.binary_loader(abs_file_path_gt)

            labels.append(label)  # [None, None, None, GT]


            # Images
            if not self.baseline_mode:
                abs_file_path, new_file_path = self.filename_from_index(file_path, i)
            else:
                abs_file_path, new_file_path = self.filename_from_base(file_path, i)
            if not os.path.exists(abs_file_path):
                print(abs_file_path, "does not exist !")
                exit(1)
            image = self.rgb_loader(abs_file_path)

            images.append(image)  # [img3, img2, img1, ori_img]


            # Depth Images
            if not self.baseline_mode:
                abs_file_path_dep, new_file_path_dep = self.filename_from_index(file_path_dep, i)
            else:
                abs_file_path_dep, new_file_path_dep = self.filename_from_base(file_path_dep, i)
            if not os.path.exists(abs_file_path_dep):
                print(abs_file_path_dep, "does not exist !")
                exit(1)
            depth = self.binary_loader(abs_file_path_dep)

            depths.append(depth)  # [depth3, depth2, depth1, ori_depth]

        file_name = file_path_gt.split('/')[-1]
        video_name = file_path_gt.split('/')[-3]

        if self.augment:
            images, labels, depths = cv_random_flip(images, labels, depths)
            images, labels, depths = randomCrop(images, labels, depths)
            images, labels, depths = randomRotation(images, labels, depths)
            images = colorEnhance(images)

        for i, img in enumerate(images):
            if labels[i] is not None:
                labels[i] = self.gt_transform(labels[i])
            images[i] = self.img_transform(images[i])
            depths[i] = self.depths_transform(depths[i])

        images = torch.stack(images)
        depths = torch.stack(depths)
        labels = labels[self.interval[0]]


        return images, labels, depths, [file_name, video_name]


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w,h),Image.BILINEAR),gt.resize((w,h),Image.NEAREST),depth.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size

    def filename_from_index(self, base_file_path, index):
        # e.g., ../DViSal/data/CDTB_bottle_room_occ_1/GT/00000026.png'
        file_name = os.path.basename(base_file_path)    # '00000026.png'
        old_num = int(file_name[-9:-4])                 # '00026' -> int = 26
        new_num = old_num - index                       # 26 - index, e.g., index=2, obtaining 24
        name_elts = "{:05d}".format(new_num)            # '00024'
        new_file_name = file_name[:-9]+name_elts+ file_name[-4:]  # new file_name

        new_path = os.path.join(base_file_path[:-len(new_file_name)], new_file_name)

        if os.path.exists(new_path):
            return new_path, new_file_name
        else:
            return base_file_path, file_name

    def filename_from_base(self, base_file_path, index):
        file_name = os.path.basename(base_file_path)
        return base_file_path, file_name


def get_loader(data_root, batchsize, trainsize,
               subset, augmentation, interval, sample_rate,
               shuffle=True, num_workers=12, pin_memory=True, baseMode=False):

    dataset = SalObjDataset(data_root, subset, augmentation, interval, sample_rate, trainsize, baseMode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

