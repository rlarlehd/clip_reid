import os.path as osp
from .bases import BaseImageDataset


class MSMT17_QuerySwap(BaseImageDataset):
    dataset_dir = 'msmt17_queryswap'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(MSMT17_QuerySwap, self).__init__()
        self.pid_begin = pid_begin

        # root는 /home/dicia/JWChu/clip_reid/bg_eval_datasets 로 들어온다고 가정
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.orig_root = '/home/dicia/JWChu/datasets/msmt17'
        # self.swap_root = '/home/dicia/JWChu/clip_reid/bg_eval_datasets/msmt17_bgswap'
        self.swap_root = '/home/dicia/JWChu/datasets/msmt17_bgswap'

        self.train_dir = osp.join(self.orig_root, 'train')
        self.orig_test_dir = osp.join(self.orig_root, 'test')
        self.swap_test_dir = osp.join(self.swap_root, 'test')

        self.list_train_path = osp.join(self.orig_root, 'list_train.txt')
        self.list_val_path = osp.join(self.orig_root, 'list_val.txt')
        self.list_query_path = osp.join(self.orig_root, 'list_query.txt')
        self.list_gallery_path = osp.join(self.orig_root, 'list_gallery.txt')

        train = self._process_dir(self.train_dir, self.list_train_path)
        val = self._process_dir(self.train_dir, self.list_val_path)
        train += val

        # query만 swap
        query = self._process_dir(self.swap_test_dir, self.list_query_path)
        gallery = self._process_dir(self.orig_test_dir, self.list_gallery_path)

        if verbose:
            print("=> MSMT17 QuerySwap loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _process_dir(self, dir_path, list_path):
        dataset = []
        pid_container = set()

        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        for img_info in lines:
            items = img_info.strip().split()
            if len(items) < 2:
                continue

            img_rel_path, pid = items[0], int(items[1])
            camid = int(osp.basename(img_rel_path).split('_')[2])
            img_path = osp.join(dir_path, img_rel_path)

            dataset.append((img_path, self.pid_begin + pid, camid - 1, 0))
            pid_container.add(pid)

        for idx, pid in enumerate(sorted(pid_container)):
            assert idx == pid

        return dataset
