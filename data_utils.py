from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    Compose,
    RandomCrop,
    ToTensor,
    ToPILImage,
    CenterCrop,
    Resize,
    GaussianBlur,
)
import numpy as np
import cv2
from EMAN2 import *
from EMAN2 import EMData


def is_image_file(filename):
    return any(
        filename.endswith(extension)
        for extension in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    )


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose(
        [
            RandomCrop(crop_size),
            ToTensor(),
        ]
    )


def train_lr_transform(crop_size, upscale_factor):
    return Compose(
        [
            ToPILImage(),
            Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
            GaussianBlur(5, 5),
            ToTensor(),
        ]
    )


def display_transform():
    return Compose([ToPILImage(), Resize(400), CenterCrop(400), ToTensor()])


import glob


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [
            join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)
        ]
        self.upscale_factor = upscale_factor
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

        matches = glob.glob(
            "./data/RealHospData/**/roi1_seq2_Low NA *.mrc", recursive=True
        )
        matches = sorted(matches)
        self.mine_l, self.mine_h = matches[0::2], matches[1::2]

        # self.mine_l, self.mine_h = self.mine_l[10::10], self.mine_h[10::10]

        self.mine_l_train = [x for i, x in enumerate(self.mine_l) if i % 10 != 0]
        self.mine_h_train = [x for i, x in enumerate(self.mine_h) if i % 10 != 0]
        self.mine_l, self.mine_h = self.mine_l_train, self.mine_h_train
        self.ratio = len(self.image_filenames) / (
            len(self.mine_l) + len(self.image_filenames)
        )

    def get_voc_item(self, index):
        index = index % len(self.image_filenames)
        source_img = Image.open(self.image_filenames[index])

        source_img = RandomCrop(self.crop_size)(source_img)
        hr_image = ToTensor()(source_img)

        lr_image = ToPILImage()(hr_image)
        lr_image = Resize(
            self.crop_size // self.upscale_factor, interpolation=Image.BICUBIC
        )(lr_image)
        p1, p2 = np.random.randint(low=1, high=6, size=2, dtype=np.int32) * 2 + 1
        lr_image = GaussianBlur(p1, p1)(lr_image)
        lr_image = ToTensor()(lr_image)
        return lr_image, hr_image

    def get_mrc_item(self, index):
        index = index % len(self.mine_l)

        l, h = self.mine_l[index], self.mine_h[index]

        img_s = EMData(l, 0)
        s = img_s.numpy()[0]
        s_max, s_min = np.max(s), np.min(s)
        s_norm = (s - s_min) / (s_max - s_min)
        s_norm_255 = s_norm * 255
        s_norm_255_uint = s_norm_255.astype(np.uint8)
        counts = np.bincount(s_norm_255_uint.reshape([-1]))
        count_max = np.argmax(counts)
        s_norm_bias = (s_norm_255 - count_max) / (255 - count_max)
        s_norm_bias_fix = np.where(s_norm_bias < 0, 0, s_norm_bias)
        s_norm_255 = (s_norm_bias_fix * 255).astype(np.uint8)

        img_b = EMData(h, 0)
        b = img_b.numpy()
        b_max, b_min = np.max(b), np.min(b)
        b_norm = (b - b_min) / (b_max - b_min)
        b_norm_255 = b_norm * 255
        b_norm_255_uint = b_norm_255.astype(np.uint8)
        counts = np.bincount(b_norm_255_uint.reshape([-1]))
        count_max = np.argmax(counts)
        b_norm_bias = (b_norm_255 - count_max) / (255 - count_max)
        b_norm_bias_fix = np.where(b_norm_bias < 0, 0, b_norm_bias)
        b_norm_255 = (b_norm_bias_fix * 255).astype(np.uint8)

        b_resized = cv2.resize(
            b_norm_255,
            (int(b_norm_255.shape[1] / 2), int(b_norm_255.shape[0] / 2)),
            interpolation=cv2.INTER_AREA,
        )
        # # down
        # s1 = s_norm_255[..., None].repeat(3, axis=2)
        # b1 = b_resized[..., None].repeat(3, axis=2)
        # p1 = (
        #     "./did/"
        #     + os.path.basename(os.path.dirname(l))
        #     + "_"
        #     + os.path.basename(l).split(".")[0]
        #     + ".jpg"
        # )
        # p2 = (
        #     "./did/"
        #     + os.path.basename(os.path.dirname(h))
        #     + "_"
        #     + os.path.basename(h).split(".")[0]
        #     + ".jpg"
        # )
        # cv2.imwrite(p1, s1)
        # cv2.imwrite(p2, b1)

        center_x = np.random.randint(44 + 1, 1024 - 44 - 1)
        center_y = np.random.randint(44 + 1, 1024 - 44 - 1)
        s_crop = s_norm_255[
            center_x - 44 : center_x + 44,
            center_y - 44 : center_y + 44,
        ]
        b_crop = b_resized[
            center_x - 44 : center_x + 44,
            center_y - 44 : center_y + 44,
        ]
        s = s_crop[..., None].repeat(3, axis=2)
        b = b_crop[..., None].repeat(3, axis=2)
        # lr_image = ToPILImage()(s)
        # hr_image = ToPILImage()(b)

        lr_image = ToTensor()(s)
        hr_image = ToTensor()(b)
        return lr_image, hr_image

    def __getitem__(self, index):
        x = np.random.rand()

        if x < 0.8:
            lr_image, hr_image = self.get_voc_item(index)
        else:
            lr_image, hr_image = self.get_mrc_item(index)

        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames) * 10


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [
            join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)
        ]

        matches = glob.glob("./data/RealHospDataV/*.jpg", recursive=True)
        matches = sorted(matches)
        self.mine_l, self.mine_h = matches[0::2], matches[1::2]
        self.ratio = len(self.image_filenames) / (
            len(self.mine_l) + len(self.image_filenames)
        )
        # self.mine_l, self.mine_h = self.mine_l[10::10], self.mine_h[10::10]

    def get_voc_item(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        p1, p2 = np.random.randint(low=1, high=6, size=2, dtype=np.int32) * 2 + 1
        lr_blur = GaussianBlur(p1, p1)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        lr_image = lr_blur(lr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def get_mrc_item(self, index):
        index = index % len(self.mine_l)

        l, h = self.mine_l[index], self.mine_h[index]
        s_norm_255 = cv2.imread(l)
        b_resized = cv2.imread(h)

        center_x = np.random.randint(375 // 2 + 1, 1024 - 375 // 2 - 1)
        center_y = np.random.randint(375 // 2 + 1, 1024 - 375 // 2 - 1)
        s_crop = s_norm_255[
            center_x - 375 // 2 : center_x + 375 // 2,
            center_y - 375 // 2 : center_y + 375 // 2,
            :,
        ]
        b_crop = b_resized[
            center_x - 375 // 2 : center_x + 375 // 2,
            center_y - 375 // 2 : center_y + 375 // 2,
            :,
        ]

        lr_image = ToTensor()(s_crop)
        hr_image = ToTensor()(b_crop)
        return lr_image, lr_image, hr_image

    def __getitem__(self, index):
        x = np.random.rand()

        if x < 0.5:
            lr_image, hr_restore_img, hr_image = self.get_voc_item(index)
        else:
            lr_image, hr_restore_img, hr_image = self.get_mrc_item(index)

        return lr_image, hr_restore_img, hr_image

    def __len__(self):
        return len(self.image_filenames) // 5


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + "/SRF_" + str(upscale_factor) + "/data/"
        self.hr_path = dataset_dir + "/SRF_" + str(upscale_factor) + "/target/"
        self.upscale_factor = upscale_factor
        self.lr_filenames = [
            join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)
        ]
        self.hr_filenames = [
            join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)
        ]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split("/")[-1]
        lr_image = Image.open(self.lr_filenames[index])
        p1, p2 = np.random.randint(low=1, high=6, size=2, dtype=np.int32) * 2 + 1
        lr_blur = GaussianBlur(p1, p1)
        lr_image = lr_blur(lr_image)
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize(
            (self.upscale_factor * h, self.upscale_factor * w),
            interpolation=Image.BICUBIC,
        )
        hr_restore_img = hr_scale(lr_image)
        return (
            image_name,
            ToTensor()(lr_image),
            ToTensor()(hr_restore_img),
            ToTensor()(hr_image),
        )

    def __len__(self):
        return len(self.lr_filenames)
