import os.path as osp
from .bases import BaseImageDataset


class MSMT17_AugBGSwap(BaseImageDataset):
    """
    Train: Original train + BG-swap train
    Query/Gallery: Original MSMT17 test
    """

    dataset_dir = 'msmt17_aug_bgswap'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(MSMT17_AugBGSwap, self).__init__()
        self.pid_begin = pid_begin

        self.orig_root = '/home/dicia/JWChu/datasets/msmt17'
        self.swap_root = '/home/dicia/JWChu/clip_reid/bg_eval_datasets/msmt17_train_bgswap'

        self.orig_train_dir = osp.join(self.orig_root, 'train')
        self.swap_train_dir = osp.join(self.swap_root, 'train')
        self.test_dir = osp.join(self.orig_root, 'test')

        self.list_train_path = osp.join(self.orig_root, 'list_train.txt')
        self.list_val_path = osp.join(self.orig_root, 'list_val.txt')
        self.list_query_path = osp.join(self.orig_root, 'list_query.txt')
        self.list_gallery_path = osp.join(self.orig_root, 'list_gallery.txt')

        self._check_before_run()

        # 원본 train + val
        train_orig = self._process_dir(self.orig_train_dir, self.list_train_path, domain=0, check_exists=True)
        val_orig = self._process_dir(self.orig_train_dir, self.list_val_path, domain=0, check_exists=True)

        # swap train + val
        train_swap = self._process_dir(self.swap_train_dir, self.list_train_path, domain=1, check_exists=True)
        val_swap = self._process_dir(self.swap_train_dir, self.list_val_path, domain=1, check_exists=True)

        train = train_orig + val_orig + train_swap + val_swap

        query = self._process_dir(self.test_dir, self.list_query_path, domain=0, check_exists=True)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path, domain=0, check_exists=True)

        if verbose:
            print("=> MSMT17 AugBGSwap loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        for p in [
            self.orig_root,
            self.swap_root,
            self.orig_train_dir,
            self.swap_train_dir,
            self.test_dir,
            self.list_train_path,
            self.list_val_path,
            self.list_query_path,
            self.list_gallery_path,
        ]:
            if not osp.exists(p):
                raise RuntimeError("'{}' is not available".format(p))

    def _process_dir(self, dir_path, list_path, domain=0, check_exists=False):
        dataset = []
        pid_container = set()

        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        for img_info in lines:
            items = img_info.strip().split()
            if len(items) < 2:
                continue

            img_rel_path = items[0]
            pid = int(items[1])

            img_name = osp.basename(img_rel_path)
            camid = int(img_name.split('_')[2])

            img_path = osp.join(dir_path, img_rel_path)

            if check_exists and not osp.exists(img_path):
                raise FileNotFoundError(img_path)

            # 반환 형식: img_path, pid, camid, trackid
            # domain은 현재 loader에서 쓰지 않으므로 trackid 자리에 넣지 않고 0 유지
            dataset.append((img_path, self.pid_begin + pid, camid - 1, 0))
            pid_container.add(pid)

        for idx, pid in enumerate(sorted(pid_container)):
            assert idx == pid, "PID is not continuous: expected {}, got {}".format(idx, pid)

        return dataset
