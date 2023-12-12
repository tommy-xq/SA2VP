# --------------------------------------------------------
# SA2VP: Spatially Aligned-and-Adapted Visual Prompt code
# reference:
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Based on timm
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------'
import os
import torch

from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from transforms import RandomResizedCropAndInterpolationWithTwoPic, RandomResizedCropAndInterpolationWithTwoPicVal
from timm.data import create_transform

from dataset_folder import ImageFolder
from timm.data.transforms import str_to_interp_mode
import json

# for food 101
from pathlib import Path
from typing import Any, Tuple, Callable, Optional
import PIL.Image
from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive,check_integrity
from torchvision.datasets.vision import VisionDataset

# for add new datesets =========================
from torchvision.datasets.folder import make_dataset
import random
import pickle
import numpy as np
from PIL import Image
import csv


class DTD(VisionDataset):
    """`Describable Textures Dataset (DTD) <https://www.robots.ox.ac.uk/~vgg/data/dtd/>`_.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        partition (int, optional): The dataset partition. Should be ``1 <= partition <= 10``. Defaults to ``1``.

            .. note::

                The partition only changes which split each image belongs to. Thus, regardless of the selected
                partition, combining all splits will result in all images.

        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    _MD5 = "fff73e5086ae6bdbea199a49dfb8a4c1"

    def __init__(
        self,
        root: str,
        split: str = "train",
        partition: int = 1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        if not isinstance(partition, int) and not (1 <= partition <= 10):
            raise ValueError(
                f"Parameter 'partition' should be an integer with `1 <= partition <= 10`, "
                f"but got {partition} instead"
            )
        self._partition = partition
        
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._base_folder = Path(root)
        self._data_folder = self._base_folder / "dtd"
        self._meta_folder = self._data_folder / "labels"
        self._images_folder = self._data_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._image_files = []
        classes = []
        with open(self._meta_folder / f"{self._split}{self._partition}.txt") as file:
            for line in file:
                cls, name = line.strip().split("/")
                self._image_files.append(self._images_folder.joinpath(cls, name))
                classes.append(cls)

        self.classes = sorted(set(classes))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._labels = [self.class_to_idx[cls] for cls in classes]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx):
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        # sample = {
            # "image": image,
            # "label": label
        # }
        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}, partition={self._partition}"

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder) and os.path.isdir(self._data_folder)

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=str(self._base_folder), md5=self._MD5)


classes = [
    'red and white circle 20 kph speed limit',
    'red and white circle 30 kph speed limit',
    'red and white circle 50 kph speed limit',
    'red and white circle 60 kph speed limit',
    'red and white circle 70 kph speed limit',
    'red and white circle 80 kph speed limit',
    'end / de-restriction of 80 kph speed limit',
    'red and white circle 100 kph speed limit',
    'red and white circle 120 kph speed limit',
    'red and white circle red car and black car no passing',
    'red and white circle red truck and black car no passing',
    'red and white triangle road intersection warning',
    'white and yellow diamond priority road',
    'red and white upside down triangle yield right-of-way',
    'stop',
    'empty red and white circle',
    'red and white circle no truck entry',
    'red circle with white horizonal stripe no entry',
    'red and white triangle with exclamation mark warning',
    'red and white triangle with black left curve approaching warning',
    'red and white triangle with black right curve approaching warning',
    'red and white triangle with black double curve approaching warning',
    'red and white triangle rough / bumpy road warning',
    'red and white triangle car skidding / slipping warning',
    'red and white triangle with merging / narrow lanes warning',
    'red and white triangle with person digging / construction / road work warning',
    'red and white triangle with traffic light approaching warning',
    'red and white triangle with person walking warning',
    'red and white triangle with child and person walking warning',
    'red and white triangle with bicyle warning',
    'red and white triangle with snowflake / ice warning',
    'red and white triangle with deer warning',
    'white circle with gray strike bar no speed limit',
    'blue circle with white right turn arrow mandatory',
    'blue circle with white left turn arrow mandatory',
    'blue circle with white forward arrow mandatory',
    'blue circle with white forward or right turn arrow mandatory',
    'blue circle with white forward or left turn arrow mandatory',
    'blue circle with white keep right arrow mandatory',
    'blue circle with white keep left arrow mandatory',
    'blue circle with white arrows indicating a traffic circle',
    'white circle with gray strike bar indicating no passing for cars has ended',
    'white circle with gray strike bar indicating no passing for trucks has ended',
]


class GTSRB(VisionDataset):
    """`German Traffic Sign Recognition Benchmark (GTSRB) <https://benchmark.ini.rub.de/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        # args,
        split: str = "train",
        percentage: float = 0.8,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(root) / "gtsrb"
        self._target_folder = (
            self._base_folder / "GTSRB" / ("Training" if self._split in ["train", "val"] else "Final_Test/Images")
        )

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if self._split in ["train", "val"]:
            samples = make_dataset(str(self._target_folder), extensions=(".ppm",))
        else:
            with open(self._base_folder / "GT-final_test.csv") as csv_file:
                samples = [
                    (str(self._target_folder / row["Filename"]), int(row["ClassId"]))
                    for row in csv.DictReader(csv_file, delimiter=";", skipinitialspace=True)
                ]

        # self._samples = samples
        # self.transform = transform
        # self.target_transform = target_transform

        if split in ["train", "val"]:
            random.shuffle(samples)
        else:
            self._samples = samples

        if split == "train":
            self._samples = samples[:int(percentage*len(samples))]
        if split == "val":
            self._samples = samples[int(percentage*len(samples)):]

        self.classes = ['a zoomed in photo of a {} traffic sign.'.format(class_name) \
            for class_name in classes]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path, target = self._samples[index]
        sample = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        #data = {
            #"image": sample,
            #"label": target
        #}
        return sample,target

    def _check_exists(self) -> bool:
        return self._target_folder.is_dir()

    def download(self) -> None:
        if self._check_exists():
            return

        base_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"

        if self._split in ["train", "val"]:
            download_and_extract_archive(
                f"{base_url}GTSRB-Training_fixed.zip",
                download_root=str(self._base_folder),
                md5="513f3c79a4c5141765e10e952eaa2478",
            )
        else:
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_Images.zip",
                download_root=str(self._base_folder),
                md5="c7e4e6327067d32654124b0fe9e82185",
            )
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_GT.zip",
                download_root=str(self._base_folder),
                md5="fe31e9c9270bbcd7b84b7f21a9d9d9e5",
            )


class Food101(VisionDataset):
    """`The Food-101 Data Set <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_.

    The Food-101 is a challenging data set of 101 food categories, with 101'000 images.
    For each class, 250 manually reviewed test images are provided as well as 750 training images.
    On purpose, the training images were not cleaned, and thus still contain some amount of noise.
    This comes mostly in the form of intense colors and sometimes wrong labels. All images were
    rescaled to have a maximum side length of 512 pixels.


    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"

    def __init__(
        self,
        root: str,
        # args,
        split: str = "train",
        percentage: float = 0.8,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "food-101"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._labels = []
        self._image_files = []
        split_name = "test" if split == "test" else "train"
        with open(self._meta_folder / f"{split_name}.json") as f:
            metadata = json.loads(f.read())

        self.classes = sorted(metadata.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for class_label, im_rel_paths in metadata.items():
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            self._image_files += [
                self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
            ]

        if split in ["train", "val"]:
            data_zip = list(zip(self._labels, self._image_files))
            random.shuffle(data_zip)
            self._labels[:], self._image_files[:] = zip(*data_zip)
            del data_zip

        if split == "train":
            self._labels = self._labels[:int(percentage*len(self._labels))]
            self._image_files = self._image_files[:int(percentage*len(self._image_files))]
        if split == "val":
            self._labels = self._labels[int(percentage*len(self._labels)):]
            self._image_files = self._image_files[int(percentage*len(self._image_files)):]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        sample = {
            "image": image,
            "label": label
        }
        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        # args,
        split: str = "train",
        percentage: float = 0.8,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.split = split  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.split == "train" or self.split == "val":
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.split == "train":
            self.data = self.data[:int(percentage*len(self.data))]
            self.targets = self.targets[:int(percentage*len(self.targets))]
        if self.split == "val":
            self.data = self.data[int(percentage*len(self.data)):]
            self.targets = self.targets[int(percentage*len(self.targets)):]

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        #sample = {
            #"image": img,
            #"label": target
        #}
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            logger.info("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = self.split
        return f"Split: {split}"


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


class SVHN(VisionDataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_list = {
        "train": [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373",
        ],
        "val": [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373",
        ],
        "test": [
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            "test_32x32.mat",
            "eb5a983be6a315427106f1b164d9cef3",
        ],
        "extra": [
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
            "extra_32x32.mat",
            "a93ce644f1a588dc4d68dda5feec44a7",
        ],
    }

    def __init__(
        self,
        root: str,
        # args,
        split: str = "train",
        percentage: float = 0.8,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        if split == "train":
            self.labels = self.labels[:int(percentage*len(self.labels))]
            self.data = self.data[:int(percentage*len(self.data))]
        if split == "val":
            self.labels = self.labels[int(percentage*len(self.labels)):]
            self.data = self.data[int(percentage*len(self.data)):]
        
        self.classes = [str(class_name) for class_name in sorted(list(set(self.labels)))]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        #sample = {
            #"image": img,
            #"label": target
        #}
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)
# end =========================

"""
class Food101(VisionDataset):
    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = Path(self.root) / "food-101"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._labels = []
        self._image_files = []
        with open(self._meta_folder / f"{split}.json") as f:
            metadata = json.loads(f.read())

        self.classes = sorted(metadata.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for class_label, im_rel_paths in metadata.items():
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            self._image_files += [
                self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
            ]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)
"""

class VTAB(datasets.folder.ImageFolder):
    def __init__(self, root, my_mode=None, train=True, transform=None, target_transform=None, mode=None,is_individual_prompt=False,**kwargs):
        self.dataset_root = root
        self.loader = datasets.folder.default_loader
        self.target_transform = None
        self.transform = transform

        if my_mode == 'train_val':
            train_list_path = os.path.join(self.dataset_root, 'train800.txt')
            test_list_path = os.path.join(self.dataset_root, 'val200.txt')
        elif my_mode == 'trainval_test':
            train_list_path = os.path.join(self.dataset_root, 'train800val200.txt')
            test_list_path = os.path.join(self.dataset_root, 'test.txt') # test
        else:
            train_list_path = os.path.join(self.dataset_root, 'train800val200.txt')
            test_list_path = os.path.join(self.dataset_root, 'val200.txt')

        self.samples = []
        
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))


class FGVC_cub(datasets.folder.ImageFolder):
    def __init__(self, root, my_mode=None, train=True, transform=None, target_transform=None, mode=None,is_individual_prompt=False,**kwargs):
        self.dataset_root = root
        self.loader = datasets.folder.default_loader
        self.target_transform = None
        self.transform = transform

        if my_mode == 'train_val':
            train_list_path = os.path.join(self.dataset_root, 'train.json')
            test_list_path = os.path.join(self.dataset_root, 'val.json')
        elif my_mode == 'trainval_test':
            train_list_path = os.path.join(self.dataset_root, 'train.json')
            test_list_path = os.path.join(self.dataset_root, 'test.json') # test
        else:
            train_list_path = None
            test_list_path = None

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                content = json.load(f)
                for name in content:
                    img_name = 'images/'+name
                    label = int(content[name])-1
                    self.samples.append((os.path.join(root,img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                content = json.load(f)
                for name in content:
                    img_name = 'images/'+name
                    label = int(content[name])-1
                    self.samples.append((os.path.join(root,img_name), label))


class FGVC_bird(datasets.folder.ImageFolder):
    def __init__(self, root, my_mode=None, train=True, transform=None, target_transform=None, mode=None,is_individual_prompt=False,**kwargs):
        self.dataset_root = root
        self.loader = datasets.folder.default_loader
        self.target_transform = None
        self.transform = transform

        if my_mode == 'train_val':
            train_list_path = os.path.join(self.dataset_root, 'train.json')
            test_list_path = os.path.join(self.dataset_root, 'val.json')
        elif my_mode == 'trainval_test':
            train_list_path = os.path.join(self.dataset_root, 'train.json')
            test_list_path = os.path.join(self.dataset_root, 'test.json') # test
        else:
            train_list_path = None
            test_list_path = None

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                content = json.load(f)
                for name in content:
                    img_name = 'images/'+name
                    label = int(content[name])
                    self.samples.append((os.path.join(root,img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                content = json.load(f)
                for name in content:
                    img_name = 'images/'+name
                    label = int(content[name])
                    self.samples.append((os.path.join(root,img_name), label))


class FGVC_flower(datasets.folder.ImageFolder):
    def __init__(self, root, my_mode=None, train=True, transform=None, target_transform=None, mode=None,is_individual_prompt=False,**kwargs):
        self.dataset_root = root
        self.loader = datasets.folder.default_loader
        self.target_transform = None
        self.transform = transform

        if my_mode == 'train_val':
            train_list_path = os.path.join(self.dataset_root, 'train.json')
            test_list_path = os.path.join(self.dataset_root, 'val.json')
        elif my_mode == 'trainval_test':
            train_list_path = os.path.join(self.dataset_root, 'train.json')
            test_list_path = os.path.join(self.dataset_root, 'test.json') # test
        else:
            train_list_path = None
            test_list_path = None

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                content = json.load(f)
                for name in content:
                    img_name = name
                    label = int(content[name])-1
                    self.samples.append((os.path.join(root,img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                content = json.load(f)
                for name in content:
                    img_name = name
                    label = int(content[name])-1
                    self.samples.append((os.path.join(root,img_name), label))


class FGVC_dog(datasets.folder.ImageFolder):
    def __init__(self, root, my_mode=None, train=True, transform=None, target_transform=None, mode=None,is_individual_prompt=False,**kwargs):
        self.dataset_root = root
        self.loader = datasets.folder.default_loader
        self.target_transform = None
        self.transform = transform

        if my_mode == 'train_val':
            train_list_path = os.path.join(self.dataset_root, 'train.json')
            test_list_path = os.path.join(self.dataset_root, 'val.json')
        elif my_mode == 'trainval_test':
            train_list_path = os.path.join(self.dataset_root, 'train.json')
            test_list_path = os.path.join(self.dataset_root, 'test.json') # test
        else:
            train_list_path = None
            test_list_path = None

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                content = json.load(f)
                for name in content:
                    img_name = 'Images/'+name
                    label = int(content[name])-1
                    self.samples.append((os.path.join(root,img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                content = json.load(f)
                for name in content:
                    img_name = 'Images/'+name
                    label = int(content[name])-1
                    self.samples.append((os.path.join(root,img_name), label))


class FGVC_car(datasets.folder.ImageFolder):
    def __init__(self, root, my_mode=None, train=True, transform=None, target_transform=None, mode=None,is_individual_prompt=False,**kwargs):
        self.dataset_root = root
        self.loader = datasets.folder.default_loader
        self.target_transform = None
        self.transform = transform

        if my_mode == 'train_val':
            train_list_path = os.path.join(self.dataset_root, 'train.json')
            test_list_path = os.path.join(self.dataset_root, 'val.json')
        elif my_mode == 'trainval_test':
            train_list_path = os.path.join(self.dataset_root, 'train.json')
            test_list_path = os.path.join(self.dataset_root, 'test.json') # test
        else:
            train_list_path = None
            test_list_path = None

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                content = json.load(f)
                for name in content:
                    img_name = name
                    label = int(content[name])-1
                    self.samples.append((os.path.join(root,img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                content = json.load(f)
                for name in content:
                    img_name = name
                    label = int(content[name])-1
                    self.samples.append((os.path.join(root,img_name), label))


class DataAugmentationForBEiT(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, second_size=args.second_input_size,
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        if args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                map_pixels,
            ])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN,
                    std=IMAGENET_INCEPTION_STD,
                ),
            ])
        else:
            raise NotImplementedError()


    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)
        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens)

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += ")"
        return repr


class DataAugmentationForBEiT_val(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        self.common_transform = transforms.Compose([
            RandomResizedCropAndInterpolationWithTwoPicVal(
                size=args.input_size, second_size=args.second_input_size,
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        if args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([
                transforms.CenterCrop((112,112)),
                transforms.ToTensor(),
                map_pixels,
            ])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN,
                    std=IMAGENET_INCEPTION_STD,
                ),
            ])
        else:
            raise NotImplementedError()


    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)
        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens)

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += ")"
        return repr


class DataAugmentationForBEiT_vtab(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        self.common_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, second_size=args.second_input_size,
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        if args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                map_pixels,
            ])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN,
                    std=IMAGENET_INCEPTION_STD,
                ),
            ])
        else:
            raise NotImplementedError()


    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)
        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens)

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += ")"
        return repr


def build_beit_pretraining_dataset(args):
    transform = DataAugmentationForBEiT(args)
    print("Data Aug = %s" % str(transform))
    return ImageFolder(args.data_path, transform=transform)


def build_beit_pretraining_dataset_val(args):
    transform = DataAugmentationForBEiT_val(args)
    return ImageFolder('/data/fgvc_deal/cub/test', transform=transform)


def build_dataset(is_train, args):
    # must choose one
    transform = build_transform_vtab(is_train, args)
    # transform = build_transform_fgvc(is_train, args)
    
    prefix_fgvc = './data/fgvc' # replace yours, sample:'./data/fgvc'
    prefix_vtab = './data/vtab-1k' # replace yours, sample:'./data/vtab-1k'
    
    if args.data_set == 'CIFAR_ori':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    elif args.data_set == 'CUB':
        if is_train:
            dataset = FGVC_cub(root=prefix_fgvc+'/CUB_200_2011', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = FGVC_cub(root=prefix_fgvc+'/CUB_200_2011', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 200
    elif args.data_set == 'DOG':
        if is_train:
            dataset = FGVC_dog(root=prefix_fgvc+'/dogs', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = FGVC_dog(root=prefix_fgvc+'/dogs', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 120
    elif args.data_set == 'FLOWER':
        if is_train:
            dataset = FGVC_flower(root=prefix_fgvc+'/OxfordFlower', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = FGVC_flower(root=prefix_fgvc+'/OxfordFlower', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 102
    elif args.data_set == 'CAR':
        if is_train:
            dataset = FGVC_car(root=prefix_fgvc+'/cars', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = FGVC_car(root=prefix_fgvc+'/cars', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 196
    elif args.data_set == 'BIRD':
        if is_train:
            dataset = FGVC_bird(root=prefix_fgvc+'/nabirds', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = FGVC_bird(root=prefix_fgvc+'/nabirds', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 555
    elif args.data_set == 'CAL101':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/caltech101', my_mode=args.my_mode, train=True, transform=transform)  # VTAB_attnmap
        else:
            dataset = VTAB(root=prefix_vtab+'/caltech101', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 102
    elif args.data_set == 'CIFAR':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/cifar', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/cifar', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 100
    elif args.data_set == 'PATCH_CAMELYON':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/patch_camelyon', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/patch_camelyon', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 2
    elif args.data_set == 'EUROSAT':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/eurosat', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/eurosat', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 10
    elif args.data_set == 'DMLAB':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/dmlab', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/dmlab', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 6
    elif args.data_set == 'CLEVR_COUNT':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/clevr_count', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/clevr_count', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 8
    elif args.data_set == 'DTD':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/dtd', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/dtd', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 47
    elif args.data_set == 'FLOWER_S':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/oxford_flowers102', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/oxford_flowers102', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 102
    elif args.data_set == 'PET':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/oxford_iiit_pet', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/oxford_iiit_pet', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 37
    elif args.data_set == 'SVHN_S':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/svhn', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/svhn', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 10
    elif args.data_set == 'SUN':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/sun397', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/sun397', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 397
    elif args.data_set == 'Resisc45':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/resisc45', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/resisc45', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 45
    elif args.data_set == 'Retinopathy':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/diabetic_retinopathy', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/diabetic_retinopathy', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 5
    elif args.data_set == 'CLEVR_DISTANCE':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/clevr_dist', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/clevr_dist', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 6
    elif args.data_set == 'KITTI_DISTANCE':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/kitti', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/kitti', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 4
    elif args.data_set == 'DS_LOC':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/dsprites_loc', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/dsprites_loc', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 16
    elif args.data_set == 'DS_ORI':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/dsprites_ori', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/dsprites_ori', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 16
    elif args.data_set == 'SN_AZI':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/smallnorb_azi', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/smallnorb_azi', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 18
    elif args.data_set == 'SN_ELE':
        if is_train:
            dataset = VTAB(root=prefix_vtab+'/smallnorb_ele', my_mode=args.my_mode, train=True, transform=transform)
        else:
            dataset = VTAB(root=prefix_vtab+'/smallnorb_ele', my_mode=args.my_mode, train=False, transform=transform)
        nb_classes = 9
    elif args.data_set == 'DTD_DAM':
        if is_train:
            dataset = DTD(root='/data/damvp_data/cal_all/dtd', split="train", transform=transform) # note: remember to change data path.
        else:
            dataset = DTD(root='/data/damvp_data/cal_all/dtd', split="test", transform=transform) # note: use 'val' to find best and then 'test'. when training, use 'val'.
        nb_classes = 47
    elif args.data_set == 'GTSRB_DAM':
        if is_train:
            dataset = GTSRB(root='/data/damvp_data/cal_all', split="train", transform=transform)
        else:
            dataset = GTSRB(root='/data/damvp_data/cal_all', split="test", transform=transform)
        nb_classes = 43
    elif args.data_set == 'FOOD_DAM':
        if is_train:
            dataset = Food101(root='/data/data', split="train", transform=transform)
        else:
            dataset = Food101(root='/data/data', split="test", transform=transform)
        nb_classes = 101
    elif args.data_set == 'CIFAR10_DAM':
        if is_train:
            dataset = CIFAR10(root='/data/damvp_data/cal_all', split="train", transform=transform)
        else:
            dataset = CIFAR10(root='/data/damvp_data/cal_all', split="val", transform=transform)
        nb_classes = 10
    elif args.data_set == 'CIFAR100_DAM':
        if is_train:
            dataset = CIFAR100(root='/data/damvp_data/cal_all', split="train", transform=transform)
        else:
            dataset = CIFAR100(root='/data/damvp_data/cal_all', split="test", transform=transform)
        nb_classes = 100
    elif args.data_set == 'SVHN_DAM':
        if is_train:
            dataset = SVHN(root='/data/damvp_data/cal_all/svhn', split="train", transform=transform)
        else:
            dataset = SVHN(root='/data/damvp_data/cal_all/svhn', split="test", transform=transform)
        nb_classes = 10
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform
    
    """
    train_t = []
    if is_train:
        train_t.append(transforms.Resize((256,256), interpolation=3))
        train_t.append(transforms.RandomCrop((224,224)))
        train_t.append(transforms.RandomHorizontalFlip(0.5))
        train_t.append(transforms.ToTensor())
        train_t.append(transforms.Normalize(mean, std))
        return transforms.Compose(train_t)
    """
    
    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images InterpolationMode.BICUBIC 3
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_transform_vtab(is_train, args):
    resize_im = args.input_size > 32  # default 224 size
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    
    # use Resize((224,224))
    if is_train:
        if True:
            transform = transforms.Compose(
                [
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]
            )
    else:
        if True:
            transform = transforms.Compose(
                [
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]
            )
    return transform

def build_transform_fgvc(is_train, args):
    resize_im = args.input_size > 32  # default 224 size
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )
    return transform

