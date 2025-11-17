import os
from tqdm import tqdm
from PIL import Image
from torch.utils import data
from torchvision import transforms
from utils import get_image, preproc

Image.MAX_IMAGE_PIXELS = None  # remove DecompressionBombWarning



class MyData(data.Dataset):
    def __init__(self, config, dataset_dir, image_size, is_train=True):
        self.size_train = image_size
        self.size_test = image_size
        self.preproc_methods = config.preproc_methods
        self.keep_size = not config.img_size
        self.data_size = [config.img_size, config.img_size]
        self.is_train = is_train
        self.load_all = config.load_all

        self.transform_image = transforms.Compose(
            [
                transforms.Resize(self.data_size[::-1]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.transform_label = transforms.Compose(
            [
                transforms.Resize(self.data_size[::-1]),
                transforms.ToTensor(),
            ]
        )
        self.image_paths = []


        image_root = os.path.join(dataset_dir, "Image")
        self.image_paths += [
            os.path.join(image_root, p)
            for p in os.listdir(image_root)
            if p.endswith(".jpg")
        ]

        self.label_paths = []
        for p in self.image_paths:
            base, ext = os.path.splitext(p)  # 分离文件名和扩展名
            p_gt = os.path.join(
                os.path.dirname(p).replace("Image", "GT_Object"),
                os.path.basename(base) + ".png",
            )
            if os.path.exists(p_gt):
                self.label_paths.append(p_gt)

        if len(self.label_paths) != len(self.image_paths):
            set_image_paths = set(
                [os.path.splitext(p.split(os.sep)[-1])[0] for p in self.image_paths]
            )
            set_label_paths = set(
                [os.path.splitext(p.split(os.sep)[-1])[0] for p in self.label_paths]
            )
            print("Path diff:", set_image_paths - set_label_paths)
            raise ValueError(
                f"There are different numbers of images ({len(self.label_paths)}) and labels ({len(self.image_paths)})"
            )

        if self.is_train:
            # Get ground truth edges
            self.edge_paths = []
            for p in self.image_paths:
                base, ext = os.path.splitext(p)  # 分离文件名和扩展名
                p_gt = os.path.join(
                    os.path.dirname(p).replace("Image", "GT_Edge"),
                    os.path.basename(base) + ".png",
                )
                if os.path.exists(p_gt):
                    self.edge_paths.append(p_gt)

            if len(self.edge_paths) != len(self.image_paths):
                set_image_paths = set(
                    [os.path.splitext(p.split(os.sep)[-1])[0] for p in self.image_paths]
                )
                set_edge_paths = set(
                    [os.path.splitext(p.split(os.sep)[-1])[0] for p in self.edge_paths]
                )
                print("Path diff:", set_image_paths - set_edge_paths)
                raise ValueError(
                    f"There are different numbers of images ({len(self.edge_paths)}) and edges ({len(self.image_paths)})"
                )

        if self.load_all:
            self.images_loaded, self.labels_loaded = [], []
            if self.is_train:
                self.edges_loaded = []
            for i, (image_path, label_path) in enumerate(
                tqdm(
                    zip(self.image_paths, self.label_paths),
                    total=len(self.image_paths),
                )
            ):
                _image = get_image(image_path, size=self.data_size, color_type="rgb")
                _label = get_image(label_path, size=self.data_size, color_type="gray")

                self.images_loaded.append(_image)
                self.labels_loaded.append(_label)

                if self.is_train:
                    _edge = get_image(
                        self.edge_paths[i], size=self.data_size, color_type="gray"
                    )
                    self.edges_loaded.append(_edge)
    def __getitem__(self, index):

        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
            if self.is_train:
                edge = self.edges_loaded[index]
        else:
            image = get_image(
                self.image_paths[index], size=self.data_size, color_type="rgb"
            )
            label = get_image(
                self.label_paths[index], size=self.data_size, color_type="gray"
            )
            if self.is_train:
                edge = get_image(
                    self.edge_paths[index], size=self.data_size, color_type="gray"
                )

        # loading image and label
        if self.is_train:
            image, label, edge = preproc(
                image,
                label,
                edge,
                preproc_methods=self.preproc_methods,
            )

            edge = self.transform_label(edge)

        image, label = (
            self.transform_image(image),
            self.transform_label(label),
        )

        if self.is_train:
            return image, label, edge
        else:
            return image, label, self.label_paths[index]

    def __len__(self):
        return len(self.image_paths)
