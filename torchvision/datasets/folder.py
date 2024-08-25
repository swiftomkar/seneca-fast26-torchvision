import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union

from PIL import Image

from .vision import VisionDataset

import psutil
import io
import PIL
import redis
import multiprocessing as mp
from multiprocessing import shared_memory, process, Manager
from multiprocessing.managers import SharedMemoryManager
import torch.multiprocessing as multiprocessing
import queue
import pickle
import numpy as np
import torch
import sys
import datetime
import logging
import threading
import random
import time
import copy
import multiprocessing as mp
import concurrent.futures
import io

def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            sample_port: Optional[int] = 6378,
            tensor_port: Optional[int] = 6380
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        # self.key_id_map = redis.Redis(port=6378)
        # self.key_id_map_ppd = redis.Redis(port=6380)
        self.avg_transform_time = 0

        # self.key_id_map = memory_manager_xxx("shm_transformed_tensor_big", 1_000_000_000)

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Args:
        #    index (int): Index
        #
        # Returns:
        #    tuple: (sample, target) where target is class_index of the target class.
        path, target = self.samples[index]
        start_time = datetime.datetime.now()
        sample = self.loader(path)
        sample.filename = path.split('/')[-1]
        read_time = (datetime.datetime.now() - start_time).total_seconds()
        transform_start_time = datetime.datetime.now()
        if self.transform is not None:
            self.transform(sample)
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        trans_time = (datetime.datetime.now() - transform_start_time).total_seconds()
        total_time = (datetime.datetime.now() - start_time).total_seconds()
        self.avg_transform_time += trans_time
        if index % 100000 == 0:
            # print(str(read_time/total_time)+", "+ str(trans_time/total_time))
            print(str(read_time) + ", " + str(trans_time))
            print("total_time= " + str((datetime.datetime.now() - start_time).total_seconds()))
            print("total transform time: " + str(self.avg_transform_time))
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class BRDatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            random_transforms: Optional[Callable] = None,
            static_transforms: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            sample_port: Optional[int] = 6378,
            tensor_port: Optional[int] = 6380,
            decoded_port: Optional[int] = 6379,
            raw_cache_host: Optional[str] = 'localhost',
            tensor_cache_host: Optional[str] = 'localhost',
            decoded_cache_host: Optional[str] = 'localhost',
            xtreme_speed: Optional[bool] = True,
            initial_cache_size: Optional[int] = 100,
            initial_cache_split: Optional[list] = [33, 33, 33]

    ) -> None:
        super(BRDatasetFolder, self).__init__(root, transform=transform,
                                              target_transform=target_transform)
        print("--- BRDatasetFolder: ", initial_cache_size, initial_cache_split, " ---")
        self.random_transforms = random_transforms
        self.static_transofrms = static_transforms
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.sample_port = sample_port
        self.tensor_port = tensor_port
        self.decoded_port = decoded_port
        self.sample_host = raw_cache_host
        self.decoded_host = decoded_cache_host
        self.tensor_host = tensor_cache_host
        self.cache_size = 7000000  # 1170000#780000
        self.sample_cache = redis.Redis(host=self.sample_host, port=self.sample_port)
        self.tensor_cache = redis.Redis(host=self.tensor_host, port=self.tensor_port)
        self.decoded_cache = redis.Redis(host=self.decoded_host, port=self.decoded_port)

        self.raw_cache_alloc = initial_cache_split[0]
        self.decoded_cache_alloc = initial_cache_split[1]
        self.augmented_cache_alloc = initial_cache_split[2]

        self.raw_cache_alloc_base = self.raw_cache_alloc
        self.decoded_cache_alloc_base = self.decoded_cache_alloc
        self.augmented_cache_alloc_base = self.augmented_cache_alloc

        self.cache_alloc = initial_cache_size * 1000000000

        tensor_cache_size = int(((initial_cache_split[2] / 100) * self.cache_alloc) / 2)
        decoded_cache_size = int((initial_cache_split[1] / 100) * self.cache_alloc) + tensor_cache_size
        sample_cache_size = int((initial_cache_split[0] / 100) * self.cache_alloc)
        self.sample_cache.execute_command('CONFIG', 'SET', 'maxmemory', sample_cache_size)

        self.decoded_cache.execute_command('CONFIG', 'SET', 'maxmemory', decoded_cache_size)

        self.tensor_cache.execute_command('CONFIG', 'SET', 'maxmemory', tensor_cache_size)

        self.sample_cache_count = redis.Redis(host=self.sample_host, port=6388)
        self.tensor_cache_count = redis.Redis(host=self.tensor_host, port=6390)
        self.decoded_cache_count = redis.Redis(host=self.decoded_host, port=6389)
        # eviction_thread = threading.Thread(target=self.evict)
        # eviction_thread.start()

        self.tensor_size = 0
        self.raw_img_size = 0
        self.weights_set = False
        self.weights = [100, 0]
        self.xtreme_speed = xtreme_speed

        self.tensor_cache_max_ref_count = 5
        self.decode_cache_max_ref_count = 5000000
        self.sample_cache_max_ref_count = 5000000

        self.get_form_counter = 0
        self.evict_augmented = False
        self.evict_decoded = False
        self.evict_raw = False

        self.tensor_cache_info = None
        # self.missing_samples = multiprocessing.Queue()

        # thread = multiprocessing.Process(target=self.redis_put_thr, args=(self.missing_samples,),
        #                          name="redis_image_put")

        # thread = threading.Thread(target=self.redis_put_thr, args=(self.missing_samples,),
        #                          name="redis_image_put")
        # thread.start()

        """
        self.missing_samples = []
        for _ in range(200):
            _missing_samples = multiprocessing.Queue()

            thread = threading.Thread(target=self.redis_put_thr, args=(_missing_samples,),
                                  name="redis_image_put")
            thread.daemon = True
            thread.start()
            self.missing_samples.append(_missing_samples)
        """

    def get_running_instances(self):
        instances = 0
        starting_port = 6381
        while 1:
            try:
                client = redis.Redis(port=starting_port)
                client.ping()
                instances += 1
                starting_port += 1
            except:
                break
        return instances

    def union_of_lists(self, lists):
        sets = [set(lst) for lst in lists]
        common_elements = set.intersection(*sets)
        return list(common_elements)

    def evict(self):
        print("eviction controller started")
        sample_cache = redis.Redis(host=self.sample_host, port=self.sample_port)
        decoded_cache = redis.Redis(host=self.decoded_host, port=self.decoded_port)
        tensor_cache = redis.Redis(host=self.tensor_host, port=self.tensor_port)
        while (1):
            try:

                total_mem_used = tensor_cache.dbsize() + sample_cache.dbsize() + decoded_cache.dbsize()
                # dividing tensor_cache.dbsize() by 2 because tensor cache now holds the reference count for each key
                print('total memory used:', total_mem_used)
            except:
                print("ERROR in folder.py eviction thread")
                total_mem_used = 0
            if total_mem_used > self.cache_size:
                print('finding eviction candidates')
                running_instances = self.get_running_instances()
                # print("eviction started on ", running_instances ," instances")
                # flush granularity in one call: 5000 keys. This can be parametrzed further

                starting_port = 6381
                list = []
                for i in range(running_instances):
                    port = starting_port + i
                    redis_db_keys = redis.Redis(port=port).keys('*')
                    if len(redis_db_keys) > 0:
                        list.append(redis_db_keys)
                to_evict = self.union_of_lists(list)
                num_keys_to_evict = total_mem_used - self.cache_size
                to_evict = to_evict[:num_keys_to_evict]
                # for i in to_evict:
                if len(to_evict) > 0:
                    print("will evict ", len(to_evict), ' keys')
                    sample_cache.delete(*to_evict)
                    decoded_cache.delete(*to_evict)
                    tensor_cache.delete(*to_evict)
                # sample_cache.close()
                # tensor_cache.close()
            time.sleep(120)
            # print("eviction ended, evicted ",len(to_evict), " samples")

    def redis_put_thr_(self, image, index):
        byte_stream = io.BytesIO()
        image.save(byte_stream, format=image.format)
        byte_image = byte_stream.getvalue()

        sample_cache = self.sample_cache
        sample_cache_count = self.sample_cache_count
        try:
            sample_cache.set(index, byte_image)
            sample_cache_count.set(index, 0)
        except:
            return None
        return None

    def redis_put_tensor_(self, tensor, index):
        tensor_bytes = pickle.dumps(tensor)
        tensor_cache = self.tensor_cache
        tensor_cache_count = self.tensor_cache_count
        try:
            tensor_cache.set(index, tensor_bytes)
            tensor_cache_count.set(index, 0)
        except:
            return None
        return None

    def redis_put_decoded_(self, tensor, index):
        tensor_bytes = pickle.dumps(tensor)
        decoded_cache = self.decoded_cache
        decoded_cache_count = self.decoded_cache_count
        try:
            decoded_cache.set(index, tensor_bytes)
            decoded_cache_count.set(index, 0)
        except:
            return None
        return None

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def is_memory_close_to_limit(self, host):
        # Connect to Redis
        result = []
        for port in [6378, 6376, 6380]:
            r = redis.Redis(host=host, port=port)

            # Get info
            info = r.info()

            # Extract memory-related information
            used_memory = info['used_memory']
            max_memory = info['maxmemory']

            # Check if memory usage is close to the limit (within 5%)
            threshold = 0.05 * max_memory
            is_close_to_limit = used_memory >= (max_memory - threshold)
            result.append(0 if is_close_to_limit else 1)

        return result

    def model_impl(self, ds_size, infl_fac):
        def highest_index_with_max(lst):
            max_value = max(lst)
            max_index = len(lst) - 1 - lst[::-1].index(max_value)
            return max_index

        ds_size = ds_size * 1000000000
        b_gpu = 1050
        b_cpu = 800
        b_ssd = 500
        s_mem = 115_000000000
        nvme_bw = 500
        rt_size = 140000
        ppt_size_meas = 2000000
        x = infl_fac  # ppt_size_meas/rt_size
        # print(x)
        ppt_size = rt_size * x
        dt = ds_size / rt_size

        bw_list = []
        for h in range(0, 110, 10):
            h = h / 100
            remaining_tensors = dt

            tensor_ppd = min(remaining_tensors, ((h * s_mem) / ppt_size))
            remaining_tensors -= tensor_ppd
            tensor_rd = min(remaining_tensors, (((1 - h) * s_mem) / rt_size))
            remaining_tensors -= tensor_rd
            tensor_storage = remaining_tensors

            b_perf = (((tensor_ppd / dt) * b_gpu) + \
                      ((tensor_rd / dt) * min(b_cpu, b_gpu)) + \
                      ((tensor_storage / dt) * min(b_cpu, nvme_bw, b_gpu)))
            bw_list.append(b_perf)
        # print(bw_list)
        ans = (highest_index_with_max(bw_list)) * 10
        # print(ans)
        return ans

    def get_form(self):
        self.get_form_counter += 1
        if self.get_form_counter >= 5000 or self.tensor_cache_info is None:
            self.tensor_cache_info = self.tensor_cache.info()
            self.decoded_cache_info = self.decoded_cache.info()
            self.raw_cache_info = self.sample_cache.info()
            self.get_form_counter = 0

        tensor_maxmem = int(self.tensor_cache_info['maxmemory'])
        decoded_maxmem = int(self.decoded_cache_info['maxmemory'])
        raw_maxmem = int(self.raw_cache_info['maxmemory'])

        tensor_used_mem = self.tensor_cache_info['used_memory']
        decoded_used_mem = self.decoded_cache_info['used_memory']
        raw_used_mem = self.raw_cache_info['used_memory']

        if tensor_used_mem < tensor_maxmem and self.evict_augmented == False:
            return 2
        elif decoded_used_mem < decoded_maxmem and self.evict_decoded == False:
            return 1
        elif raw_used_mem < raw_maxmem and self.evict_raw == False:
            return 0
        else:
            return None

    def is_space_available(self, cache):
        cache_info = cache.info()

        maxmem = int(cache_info['maxmemory'])

        used_mem = cache_info['used_memory']
        return maxmem > used_mem

    def handle_cache_miss(self, path, index):
        miss_rate_record = redis.Redis(port=6377)
        miss_rate_record.incr('miss_rate')
        if self.raw_img_size == 0:
            self.raw_img_size = os.path.getsize(path)
        if self.raw_img_size != 0 and self.tensor_size != 0:
            if self.weights[0] == 100:
                # self.weights_set = True
                # print(self.tensor_size/self.raw_img_size)
                pre_proc_perc = self.model_impl(150, self.tensor_size / self.raw_img_size)
                self.weights = [100 - pre_proc_perc, pre_proc_perc]
                # print(self.weights)
            # weights = [0, 100]

        self.weights = [33, 33, 34]  # overriding the model for testing
        # is_cache_near_full = self.is_memory_close_to_limit(self.decoded_host)
        # self.weights = (np.array(self.weights)*np.array(is_cache_near_full)).tolist()
        # print("weights:",self.weights)
        form_options = ['raw', 'dec', 'p']

        if sum(self.weights) == 0:
            print('here')
            form = None
        else:
            chosen_form_idx = self.get_form()#self.request_split(self.weights)
            if chosen_form_idx != None:
                form = form_options[chosen_form_idx]  # (can be one of 'raw' or 'p')
            else:
                form = "no cache"
        if form == 'raw':
            image = Image.open(path)
            image_copy = copy.copy(image)
            thread = threading.Thread(target=self.redis_put_thr_, args=(image, index),
                                      name="redis_image_put")
            thread.start()
            image = image_copy.convert('RGB')
            return (image, 'raw')
        elif form == 'p':
            image = self.loader(path)
            return (image, 'raw_differed')
        elif form == 'dec':
            image = self.loader(path)
            return (image, 'raw_dec_differed')
        else:
            image = Image.open(path)
            image = image.convert('RGB')
            return (image, 'raw')

    def cache_and_get(self, path, index):
        request_rate_record = redis.Redis(port=6377)
        request_rate_record.incr('requests')
        if self.tensor_cache.exists(index):
            byte_tensor = self.tensor_cache.get(index)
            if byte_tensor != None:
                tensor = pickle.loads(byte_tensor)
                ref_count = self.tensor_cache_count.incr(index)
                if ref_count >= self.tensor_cache_max_ref_count:
                    self.tensor_cache_count.publish('delete', index)
                return (tensor, 'p')
            else:
                print('[WARN] Possibly corrupt data in redis tensor cache')
                return self.handle_cache_miss(path, index)

        elif self.decoded_cache.exists(index):
            byte_tensor = self.decoded_cache.get(index)
            if byte_tensor != None:
                tensor = pickle.loads(byte_tensor)
                tensor = self.random_transforms(tensor)
                if self.is_space_available(self.tensor_cache):  # is there free space in augmented cache?
                    self.redis_put_tensor_(tensor, index)
                """
                # MODEL IMPLEMENTATION HERE
                self.weights = [33, 33, 34]  # overriding the model for testing
                # is_cache_near_full = self.is_memory_close_to_limit(self.decoded_host)
                # self.weights = (np.array(self.weights) * np.array(is_cache_near_full)).tolist()
                # print("weights:",self.weights)
                form_options = ['raw', 'dec', 'p']
                chosen_form_idx = self.get_form()#self.request_split(self.weights)
                form = form_options[chosen_form_idx]
                if form == 'p':
                    thread = threading.Thread(target=self.redis_put_tensor_, args=(tensor, index),
                                              name="redis_tensor_put")
                    thread.start()
                """
                ref_count = self.decoded_cache_count.incr(index)
                if ref_count >= self.decode_cache_max_ref_count:
                    self.decoded_cache_count.publish('delete', index)

                return (tensor, 'p')
            else:
                print('[WARN] Possibly corrupt data in redis decoded cache')
                return self.handle_cache_miss(path, index)

        # elif self.sample_cache.exists(index) and (ref_count_value := self.sample_cache.get(str(index) + '_ref_count')) is not None and int(ref_count_value or 0) < self.max_ref_count:
        elif self.sample_cache.exists(index):
            byte_image = self.sample_cache.get(index)
            if byte_image != None:
                sample = Image.open(io.BytesIO(byte_image))
                sample = sample.convert('RGB')
                # sample = pickle.loads(byte_image)
                # self.sample_cache.incr(str(index) + '_ref_count')
                ref_count = self.sample_cache_count.incr(index)
                if ref_count >= self.sample_cache_max_ref_count:
                    self.sample_cache_count.publish('delete', index)
                return (sample, 'raw')
            else:
                print('[WARN] Possibly corrupt data in redis sample cache')
                return self.handle_cache_miss(path, index)
        else:  # data was not found in redis cache
            return self.handle_cache_miss(path, index)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        # Args:
        #    index (int): Index

        # Returns:
        #    tuple: (sample, target) where target is class_index of the target class.
        # print(worker_id)

        path, target = self.samples[index]
        sample, type = self.cache_and_get(path, index)
        if sample != None:
            if type == "raw":
                sample.filename = path.split('/')[-1]
                if self.transform is not None:
                    sample = self.static_transofrms(sample)
                    if self.is_space_available(self.decoded_cache):
                        self.redis_put_decoded_(sample, index)
                    sample = self.random_transforms(sample)
                    if self.is_space_available(self.tensor_cache):
                        self.redis_put_tensor_(sample, index)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target
            elif type == 'p':
                sample.filename = path.split('/')[-1]
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target
            elif type == 'raw_differed':
                sample.filename = path.split('/')[-1]
                if self.static_transofrms is not None:
                    sample = self.static_transofrms(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                thread = threading.Thread(target=self.redis_put_decoded_, args=(sample, index),
                                          name="redis_decoded_put")
                thread.start()

                if self.random_transforms is not None:
                    sample = self.random_transforms(sample)

                thread = threading.Thread(target=self.redis_put_tensor_, args=(sample, index),
                                          name="redis_tensor_put")
                thread.start()
                return sample, target
            elif type == "raw_dec_differed":
                sample.filename = path.split('/')[-1]
                if self.static_transofrms is not None:
                    sample = self.static_transofrms(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)

                thread = threading.Thread(target=self.redis_put_decoded_, args=(sample, index),
                                          name="redis_decoded_put")
                thread.start()
                sample = self.random_transforms(sample)
                return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class CachedDatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            random_transforms: Optional[Callable] = None,
            static_transforms: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            sample_port: Optional[int] = 6378,
            tensor_port: Optional[int] = 6380,
            decoded_port: Optional[int] = 6379,
            raw_cache_host: Optional[str] = 'localhost',
            tensor_cache_host: Optional[str] = 'localhost',
            decoded_cache_host: Optional[str] = 'localhost',
            xtreme_speed: Optional[bool] = True,
            initial_cache_size: Optional[int] = 100,
            initial_cache_split: Optional[list] = [33, 33, 33]

    ) -> None:
        super(CachedDatasetFolder, self).__init__(root, transform=transform,
                                                  target_transform=target_transform)
        print("--- CachedDatasetFolder-NEW: ", initial_cache_size, initial_cache_split, " ---")
        self.random_transforms = random_transforms
        self.static_transofrms = static_transforms
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.sample_port = sample_port
        self.tensor_port = tensor_port
        self.decoded_port = decoded_port
        self.sample_host = raw_cache_host
        self.decoded_host = decoded_cache_host
        self.tensor_host = tensor_cache_host
        self.cache_size = 780000  # 1170000#780000
        self.sample_cache = redis.Redis(host=self.sample_host, port=self.sample_port)
        self.tensor_cache = redis.Redis(host=self.tensor_host, port=self.tensor_port)
        self.decoded_cache = redis.Redis(host=self.decoded_host, port=self.decoded_port)

        self.raw_cache_alloc = initial_cache_split[0]
        self.decoded_cache_alloc = initial_cache_split[1]
        self.augmented_cache_alloc = initial_cache_split[2]

        self.raw_cache_alloc_base = self.raw_cache_alloc
        self.decoded_cache_alloc_base = self.decoded_cache_alloc
        self.augmented_cache_alloc_base = self.augmented_cache_alloc

        self.cache_alloc = initial_cache_size * 1000000000

        tensor_cache_size = int(((initial_cache_split[2] / 100) * self.cache_alloc) / 2)
        decoded_cache_size = int((initial_cache_split[1] / 100) * self.cache_alloc) + tensor_cache_size
        sample_cache_size = int((initial_cache_split[0] / 100) * self.cache_alloc)
        self.sample_cache.execute_command('CONFIG', 'SET', 'maxmemory', sample_cache_size)

        self.decoded_cache.execute_command('CONFIG', 'SET', 'maxmemory', decoded_cache_size)

        self.tensor_cache.execute_command('CONFIG', 'SET', 'maxmemory', tensor_cache_size)

        self.sample_cache_count = redis.Redis(host=self.sample_host, port=6388)
        self.tensor_cache_count = redis.Redis(host=self.tensor_host, port=6390)
        self.decoded_cache_count = redis.Redis(host=self.decoded_host, port=6389)
        # eviction_thread = threading.Thread(target=self.evict)
        # eviction_thread.start()

        self.tensor_size = 0
        self.raw_img_size = 0
        self.weights_set = False
        self.weights = [100, 0]
        self.xtreme_speed = xtreme_speed

        self.tensor_cache_max_ref_count = 3
        self.decode_cache_max_ref_count = 5000000
        self.sample_cache_max_ref_count = 5000000

        self.total_fetch_time = 0
        self.total_raw_processing_time = 0
        self.total_augmneted_processing_time = 0

        self.evict_augmented = False
        self.evict_decoded = False
        self.evict_raw = False

        self.get_form_counter = 0
        self.tensor_cache_info = None
        # self.missing_samples = multiprocessing.Queue()

        # thread = multiprocessing.Process(target=self.redis_put_thr, args=(self.missing_samples,),
        #                          name="redis_image_put")

        # thread = threading.Thread(target=self.redis_put_thr, args=(self.missing_samples,),
        #                          name="redis_image_put")
        # thread.start()

        """
        self.missing_samples = []
        for _ in range(200):
            _missing_samples = multiprocessing.Queue()

            thread = threading.Thread(target=self.redis_put_thr, args=(_missing_samples,),
                                  name="redis_image_put")
            thread.daemon = True
            thread.start()
            self.missing_samples.append(_missing_samples)
        """

    def get_running_instances(self):
        instances = 0
        starting_port = 6381
        while 1:
            try:
                client = redis.Redis(port=starting_port)
                client.ping()
                instances += 1
                starting_port += 1
            except:
                break
        return instances

    def union_of_lists(self, lists):
        sets = [set(lst) for lst in lists]
        common_elements = set.intersection(*sets)
        return list(common_elements)

    def evict(self):
        print("eviction controller started")
        sample_cache = redis.Redis(host=self.sample_host, port=self.sample_port)
        decoded_cache = redis.Redis(host=self.decoded_host, port=self.decoded_port)
        tensor_cache = redis.Redis(host=self.tensor_host, port=self.tensor_port)
        while (1):
            try:

                total_mem_used = tensor_cache.dbsize() / 2 + \
                                 sample_cache.dbsize()  # dividing tensor_cache.dbsize() by 2 because tensor cache now holds the reference count for each key
                print('total memory used:', total_mem_used)
            except:
                print("ERROR in folder.py eviction thread")
                total_mem_used = 0
            if total_mem_used > self.cache_size:
                print('finding eviction candidates')
                running_instances = self.get_running_instances()
                # print("eviction started on ", running_instances ," instances")
                # flush granularity in one call: 5000 keys. This can be parametrzed further

                starting_port = 6381
                list = []
                for i in range(running_instances):
                    port = starting_port + i
                    redis_db_keys = redis.Redis(port=port).keys('*')
                    if len(redis_db_keys) > 0:
                        list.append(redis_db_keys)
                to_evict = self.union_of_lists(list)
                num_keys_to_evict = total_mem_used - self.cache_size
                to_evict = to_evict[:num_keys_to_evict]
                # for i in to_evict:
                if len(to_evict) > 0:
                    print("will evict ", len(to_evict), ' keys')
                    sample_cache.delete(*to_evict)
                    tensor_cache.delete(*to_evict)
                # sample_cache.close()
                # tensor_cache.close()
            time.sleep(120)
            # print("eviction ended, evicted ",len(to_evict), " samples")

    def redis_put_thr_(self, image, index):
        # index = index
        # image = Image.open(path)
        # print(self.missing_samples.qsize())
        # image = self.loader(path)

        byte_stream = io.BytesIO()
        image.save(byte_stream, format=image.format)
        # print("SIZE of RAW image--->",byte_stream.tell())
        # byte_stream.seek(0)
        # byte_image = byte_stream.read()
        byte_image = byte_stream.getvalue()
        # byte_image = pickle.dumps(image)

        sample_cache = self.sample_cache  # redis.Redis(host=self.sample_host, port=self.sample_port)
        sample_cache_count = self.sample_cache_count
        try:
            sample_cache.set(index, byte_image)
            sample_cache_count.set(index, 0)
            # sample_cache.set(str(index)+'_ref_count', 0)
        except:
            return None
        # sample_cache.close()
        return None

    def redis_put_tensor_(self, tensor, index):
        tensor_bytes = pickle.dumps(tensor)
        # print("SIZE of processed tensor>", len(tensor_bytes))
        tensor_cache = self.tensor_cache  # redis.Redis(host=self.tensor_host, port=self.tensor_port)
        tensor_cache_count = self.tensor_cache_count
        try:
            tensor_cache.set(index, tensor_bytes)
            tensor_cache_count.set(index, 0)
            # tensor_cache.set(str(index)+'_ref_count', 0)
        except:
            return None
        # tensor_cache.close()
        return None

    def redis_put_decoded_(self, tensor, index):
        tensor_bytes = pickle.dumps(tensor)
        decoded_cache = self.decoded_cache  # redis.Redis(host=self.tensor_host, port=self.tensor_port)
        decoded_cache_count = self.decoded_cache_count
        try:
            decoded_cache.set(index, tensor_bytes)
            decoded_cache_count.set(index, 0)
            # decoded_cache.set(str(index)+'_ref_count', 0)
        except:
            return None
        # tensor_cache.close()
        return None

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def get_form(self):
        self.get_form_counter += 1
        if self.get_form_counter >= 5000 or self.tensor_cache_info is None:
            self.tensor_cache_info = self.tensor_cache.info()
            self.decoded_cache_info = self.decoded_cache.info()
            self.raw_cache_info = self.sample_cache.info()
            self.get_form_counter = 0

        tensor_maxmem = int(self.tensor_cache_info['maxmemory'])
        decoded_maxmem = int(self.decoded_cache_info['maxmemory'])
        raw_maxmem = int(self.raw_cache_info['maxmemory'])

        tensor_used_mem = self.tensor_cache_info['used_memory']
        decoded_used_mem = self.decoded_cache_info['used_memory']
        raw_used_mem = self.raw_cache_info['used_memory']

        if tensor_used_mem < tensor_maxmem and self.evict_augmented == False:
            return 2
        elif decoded_used_mem < decoded_maxmem and self.evict_decoded == False:
            return 1
        elif raw_used_mem < raw_maxmem and self.evict_raw == False:
            return 0
        else:
            return None

    def is_space_available(self, cache):
        cache_info = cache.info()

        maxmem = int(cache_info['maxmemory'])

        used_mem = cache_info['used_memory']
        return maxmem > used_mem


    def request_split(self, weights):
        total = sum(weights)
        r = random.uniform(0, total)
        upto = 0
        for i, w in enumerate(weights):
            if upto + w >= r:
                return i
            upto += w

    def is_memory_close_to_limit(self, host):
        # Connect to Redis
        result = []
        for port in [6378, 6376, 6380]:
            r = redis.Redis(host=host, port=port)

            # Get info
            info = r.info()

            # Extract memory-related information
            used_memory = info['used_memory']
            max_memory = info['maxmemory']

            # Check if memory usage is close to the limit (within 5%)
            threshold = 0.05 * max_memory
            is_close_to_limit = used_memory >= (max_memory - threshold)
            result.append(0 if is_close_to_limit else 1)

        return result

    def model_impl(self, ds_size, infl_fac):
        def highest_index_with_max(lst):
            max_value = max(lst)
            max_index = len(lst) - 1 - lst[::-1].index(max_value)
            return max_index

        ds_size = ds_size * 1000000000
        b_gpu = 1050
        b_cpu = 800
        b_ssd = 500
        s_mem = 115_000000000
        nvme_bw = 500
        rt_size = 140000
        ppt_size_meas = 2000000
        x = infl_fac  # ppt_size_meas/rt_size
        # print(x)
        ppt_size = rt_size * x
        dt = ds_size / rt_size

        bw_list = []
        for h in range(0, 110, 10):
            h = h / 100
            remaining_tensors = dt

            tensor_ppd = min(remaining_tensors, ((h * s_mem) / ppt_size))
            remaining_tensors -= tensor_ppd
            tensor_rd = min(remaining_tensors, (((1 - h) * s_mem) / rt_size))
            remaining_tensors -= tensor_rd
            tensor_storage = remaining_tensors

            b_perf = (((tensor_ppd / dt) * b_gpu) + \
                      ((tensor_rd / dt) * min(b_cpu, b_gpu)) + \
                      ((tensor_storage / dt) * min(b_cpu, nvme_bw, b_gpu)))
            bw_list.append(b_perf)
        # print(bw_list)
        ans = (highest_index_with_max(bw_list)) * 10
        # print(ans)
        return ans

    def handle_cache_miss(self, path, index):
        miss_rate_record = redis.Redis(port=6377)
        miss_rate_record.incr('miss_rate')
        if self.raw_img_size == 0:
            self.raw_img_size = os.path.getsize(path)
        # weights=[100,0]
        if self.raw_img_size != 0 and self.tensor_size != 0:
            if self.weights[0] == 100:
                # self.weights_set = True
                # print(self.tensor_size/self.raw_img_size)
                pre_proc_perc = self.model_impl(150, self.tensor_size / self.raw_img_size)
                self.weights = [100 - pre_proc_perc, pre_proc_perc]
                # print(self.weights)
            # weights = [0, 100]

        self.weights = [00, 0, 100]  # overriding the model for testing
        # is_cache_near_full = self.is_memory_close_to_limit(self.decoded_host)
        # self.weights = (np.array(self.weights)*np.array(is_cache_near_full)).tolist()
        # print("weights:",self.weights)
        form_options = ['raw', 'dec', 'p']
        # form_options = ['raw', 'p', 'dec'] #whenever data is stored in augmented form, it should always exist in decoded cache
        if sum(self.weights) == 0:
            print('here')
            form = None
        else:
            chosen_form_idx = self.get_form()  # self.request_split(self.weights)
            if chosen_form_idx != None:
                form = form_options[chosen_form_idx]  # (can be one of 'raw' or 'p')
            else:
                form = "no cache"
        if form == 'raw':
            # self.redis_put_thr_(path, index)
            # self.missing_samples.put((path, index))

            # image = self.loader(path)
            image = Image.open(path)
            image_copy = copy.copy(image)
            thread = threading.Thread(target=self.redis_put_thr_, args=(image, index),
                                      name="redis_image_put")
            thread.start()
            # self.redis_put_thr_(path, index)
            image = image_copy.convert('RGB')

            # print("Index: ", index)
            # sample = image #image.convert('RGB')
            return (image, 'raw')
        elif form == 'p':
            image = self.loader(path)
            return (image, 'raw_differed')
        elif form == 'dec':
            image = self.loader(path)
            return (image, 'raw_dec_differed')
        else:
            # print("[ERROR]")
            image = Image.open(path)
            image = image.convert('RGB')
            return (image, 'raw')

    def cache_and_get(self, path, index):
        request_rate_record = redis.Redis(port=6377)
        request_rate_record.incr('requests')
        # if self.tensor_cache.exists(index) and (ref_count_value := self.tensor_cache.get(str(index) + '_ref_count')) is not None and int(ref_count_value or 0) < self.max_ref_count:
        if self.tensor_cache.exists(index):
            # print("Dont be here")
            byte_tensor = self.tensor_cache.get(index)
            if byte_tensor != None:
                tensor = pickle.loads(byte_tensor)
                # tensor = self.random_transform_pick(tensor)
                # put the newly transofrmed tensor back into cache and also reset the ref count
                # self.tensor_cache.incr(str(index) + '_ref_count')
                ref_count = self.tensor_cache_count.incr(index)
                if ref_count >= self.tensor_cache_max_ref_count:
                    self.tensor_cache_count.publish('delete', index)
                return (tensor, 'p')
            else:
                # print('[WARN] Possibly corrupt data in redis tensor cache')
                return self.handle_cache_miss(path, index)
        # elif self.decoded_cache.exists(index) and (ref_count_value := self.decoded_cache.get(str(index) + '_ref_count')) is not None and int(ref_count_value or 0) < self.max_ref_count:
        elif self.decoded_cache.exists(index):
            byte_tensor = self.decoded_cache.get(index)
            if byte_tensor != None:
                tensor = pickle.loads(byte_tensor)
                tensor = self.random_transforms(tensor)
                # put the newly transofrmed tensor back into tensor cache only if ... cache and also reset the ref count
                if self.is_space_available(self.tensor_cache):  # is there free space in augmented cache?
                    self.redis_put_tensor_(tensor, index)
                """
                # MODEL IMPLEMENTATION HERE
                self.weights = [00, 00, 100]  # overriding the model for testing
                # is_cache_near_full = self.is_memory_close_to_limit(self.decoded_host)
                # self.weights = (np.array(self.weights) * np.array(is_cache_near_full)).tolist()
                # print("weights:",self.weights)
                form_options = ['raw', 'dec', 'p']
                # form_options = ['raw', 'p', 'dec']  # whenever data is stored in augmented form, it should always exist in decoded cache
                chosen_form_idx = self.request_split(self.weights)
                form = form_options[chosen_form_idx]
                if form == 'p':
                    thread = threading.Thread(target=self.redis_put_tensor_, args=(tensor, index),
                                              name="redis_tensor_put")
                    thread.start()
                # self.decoded_cache.incr(str(index) + '_ref_count')
                """
                ref_count = self.decoded_cache_count.incr(index)
                if ref_count >= self.decode_cache_max_ref_count:
                    self.decoded_cache_count.publish('delete', index)

                return (tensor, 'p')
            else:
                print('[WARN] Possibly corrupt data in redis decoded cache')
                return self.handle_cache_miss(path, index)

        # elif self.sample_cache.exists(index) and (ref_count_value := self.sample_cache.get(str(index) + '_ref_count')) is not None and int(ref_count_value or 0) < self.max_ref_count:
        elif self.sample_cache.exists(index):
            byte_image = self.sample_cache.get(index)
            if byte_image != None:
                sample = Image.open(io.BytesIO(byte_image))
                sample = sample.convert('RGB')
                # sample = pickle.loads(byte_image)
                # self.sample_cache.incr(str(index) + '_ref_count')
                ref_count = self.sample_cache_count.incr(index)
                if ref_count >= self.sample_cache_max_ref_count:
                    self.sample_cache_count.publish('delete', index)
                return (sample, 'raw')
            else:
                print('[WARN] Possibly corrupt data in redis sample cache')
                return self.handle_cache_miss(path, index)
        else:  # data was not found in redis cache
            return self.handle_cache_miss(path, index)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        # Args:
        #    index (int): Index

        # Returns:
        #    tuple: (sample, target) where target is class_index of the target class.
        # print(worker_id)

        path, target = self.samples[index]

        fetch_start_time = time.time()
        sample, type = self.cache_and_get(path, index)
        self.total_fetch_time += (time.time() - fetch_start_time)

        # t1=threading.Thread(self.set_seen_samples, args=(index,))
        # t1.start()
        # print("fetch time:", self.total_fetch_time)
        # print("pre-processing time:", self.total_raw_processing_time)
        # print("augmented processing time:", self.total_augmneted_processing_time)
        if sample != None:
            if type == "raw":

                # end_time_get = datetime.datetime.now() - start_time
                # print("fetch time= "+str(end_time_get.total_seconds()))
                sample.filename = path.split('/')[-1]
                # start_time_transform = time.time()
                if self.transform is not None:
                    sample = self.static_transofrms(sample)
                    if self.is_space_available(self.decoded_cache):
                        self.redis_put_decoded_(sample, index)
                    sample = self.random_transforms(sample)
                    if self.is_space_available(self.tensor_cache):
                        self.redis_put_tensor_(sample, index)

                    #sample = self.transform(sample)
                    # sample = self.random_transform_pick(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                # self.cache_and_evict_r_ppd(sample, index)
                # self.total_raw_processing_time+=(time.time()-start_time_transform)

                # if self.tensor_size == 0:
                #    self.tensor_size=sample.element_size() * sample.nelement()

                # print("tensor size: ", sample.element_size() * sample.nelement())

                # end_time_trans = datetime.datetime.now() - start_time_transform
                # print("transform time= "+str(end_time_trans.total_seconds()))
                # print("total_time= "+ str((datetime.datetime.now()-start_time).total_seconds()))
                return sample, target
            elif type == 'p':
                sample.filename = path.split('/')[-1]

                # if self.tensor_size == 0:
                #    self.tensor_size=sample.element_size() * sample.nelement()
                # start_time_augmnet = time.time()
                if self.target_transform is not None:
                    target = self.target_transform(target)
                # self.total_augmneted_processing_time+=(time.time()-start_time_augmnet)
                # if self.random_transform_pick is not None:
                #    sample = self.random_transform_pick(sample)
                return sample, target
            elif type == 'raw_differed':
                sample.filename = path.split('/')[-1]
                # start_time_transform = datetime.datetime.now()
                # time_proc_start = time.time()
                if self.static_transofrms is not None:
                    sample = self.static_transofrms(sample)
                    # sample = self.random_transform_pick(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                # self.total_raw_processing_time+=time.time()-time_proc_start
                # sample_copy = copy.copy(sample)
                thread = threading.Thread(target=self.redis_put_decoded_, args=(sample, index),
                                          name="redis_decoded_put")
                thread.start()

                # if self.tensor_size == 0:
                #    self.tensor_size=sample.element_size() * sample.nelement()
                if self.random_transforms is not None:
                    sample = self.random_transforms(sample)
                # sample_copy_2 = copy.copy(sample_copy)

                thread = threading.Thread(target=self.redis_put_tensor_, args=(sample, index),
                                          name="redis_tensor_put")
                thread.start()
                return sample, target
            elif type == "raw_dec_differed":
                sample.filename = path.split('/')[-1]
                # start_time_transform = datetime.datetime.now()
                if self.static_transofrms is not None:
                    sample = self.static_transofrms(sample)
                    # sample = self.random_transform_pick(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)

                # if self.tensor_size == 0:
                #    self.tensor_size = sample.element_size() * sample.nelement()

                thread = threading.Thread(target=self.redis_put_decoded_, args=(sample, index),
                                          name="redis_decoded_put")
                thread.start()
                sample = self.random_transforms(sample)
                return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class CachedDatasetFolder_old(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            sample_port: Optional[int] = 6378,
            tensor_port: Optional[int] = 6380
    ) -> None:
        super(CachedDatasetFolder, self).__init__(root, transform=transform,
                                                  target_transform=target_transform)
        print("CachedDatasetFolder")
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.sample_port = sample_port
        self.tensor_port = tensor_port

        self.sample_cache = redis.Redis(port=self.sample_port)
        self.tensor_cache = redis.Redis(port=self.tensor_port)

        self.max_cache_gb = 141000  # (10 *(10**9))

        self.tensor_size = 0
        self.raw_img_size = 0
        self.weights_set = False
        self.weights = [100, 0]

    def redis_put_thr_(self, image, index):
        if self.sample_cache.dbsize() + self.tensor_cache.dbsize() <= self.max_cache_gb:
            # if self.sample_cache.memory_stats()['total.allocated'] +self.tensor_cache.memory_stats()['total.allocated'] <= self.max_cache_gb:
            # index = index
            # image = Image.open(path)
            # print(self.missing_samples.qsize())
            # image = self.loader(path)

            byte_stream = io.BytesIO()
            image.save(byte_stream, format=image.format)
            # byte_stream.seek(0)
            # byte_image = byte_stream.read()
            byte_image = byte_stream.getvalue()
            # byte_image = pickle.dumps(image)

            sample_cache = redis.Redis(port=self.sample_port)
            try:
                sample_cache.set(index, byte_image)
            except:
                return None
            return None
        else:
            return None

    def redis_put_tensor_(self, tensor, index):
        if self.sample_cache.dbsize() + self.tensor_cache.dbsize() <= self.max_cache_gb:
            # if self.sample_cache.memory_stats()['total.allocated'] + self.tensor_cache.memory_stats()[
            #    'total.allocated'] <= self.max_cache_gb:
            tensor_bytes = pickle.dumps(tensor)
            tensor_cache = redis.Redis(port=self.tensor_port)
            try:
                tensor_cache.set(index, tensor_bytes)
            except:
                return None
            return None
        else:
            return None

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def request_split(self, weights):
        total = sum(weights)
        r = random.uniform(0, total)
        upto = 0
        for i, w in enumerate(weights):
            if upto + w >= r:
                return i
            upto += w

    def model_impl(self, ds_size, infl_fac):
        def highest_index_with_max(lst):
            max_value = max(lst)
            max_index = len(lst) - 1 - lst[::-1].index(max_value)
            return max_index

        ds_size = ds_size * 1000000000
        b_gpu = 1050
        b_cpu = 800
        b_ssd = 500
        s_mem = 115_000000000
        nvme_bw = 500
        rt_size = 140000
        ppt_size_meas = 2000000
        x = infl_fac  # ppt_size_meas/rt_size
        # print(x)
        ppt_size = rt_size * x
        dt = ds_size / rt_size

        bw_list = []
        for h in range(0, 110, 10):
            h = h / 100
            remaining_tensors = dt

            tensor_ppd = min(remaining_tensors, ((h * s_mem) / ppt_size))
            remaining_tensors -= tensor_ppd
            tensor_rd = min(remaining_tensors, (((1 - h) * s_mem) / rt_size))
            remaining_tensors -= tensor_rd
            tensor_storage = remaining_tensors

            b_perf = (((tensor_ppd / dt) * b_gpu) + \
                      ((tensor_rd / dt) * min(b_cpu, b_gpu)) + \
                      ((tensor_storage / dt) * min(b_cpu, nvme_bw, b_gpu)))
            bw_list.append(b_perf)
        # print(bw_list)
        ans = (highest_index_with_max(bw_list)) * 10
        # print(ans)
        return ans

    def handle_cache_miss(self, path, index):
        miss_rate_record = redis.Redis(port=6377)
        miss_rate_record.incr('miss_rate')
        if self.raw_img_size == 0:
            self.raw_img_size = os.path.getsize(path)
        # weights=[100,0]
        if self.raw_img_size != 0 and self.tensor_size != 0:
            if self.weights[0] == 100:
                # self.weights_set = True
                # print(self.tensor_size/self.raw_img_size)
                pre_proc_perc = self.model_impl(25, self.tensor_size / self.raw_img_size)
                self.weights = [100 - pre_proc_perc, pre_proc_perc]
                # print(self.weights)
            # weights = [0, 100]

        # weights = [50, 50]
        # print("weights:",self.weights)
        # weights = [0, 100]
        form_options = ['raw', 'p']
        chosen_form_idx = self.request_split(self.weights)
        form = form_options[chosen_form_idx]  # (can be one of 'raw' or 'p')
        if form == 'raw':
            # self.redis_put_thr_(path, index)
            # self.missing_samples.put((path, index))

            # image = self.loader(path)
            image = Image.open(path)
            image_copy = copy.copy(image)
            thread = threading.Thread(target=self.redis_put_thr_, args=(image, index),
                                      name="redis_image_put")
            thread.start()
            # self.redis_put_thr_(path, index)
            image = image_copy.convert('RGB')

            # print("Index: ", index)
            # sample = image #image.convert('RGB')
            return (image, 'raw')
        else:
            image = self.loader(path)
            return (image, 'raw_differed')

    def cache_and_get(self, path, index):
        request_rate_record = redis.Redis(port=6377)
        request_rate_record.incr('requests')
        if self.tensor_cache.exists(index):
            byte_tensor = self.tensor_cache.get(index)
            if byte_tensor != None:
                tensor = pickle.loads(byte_tensor)  # torch.tensor(list(byte_tensor), dtype=torch.uint8)
                return (tensor, 'p')
            else:
                return self.handle_cache_miss(path, index)
        elif self.sample_cache.exists(index):
            byte_image = self.sample_cache.get(index)
            if byte_image != None:
                sample = Image.open(io.BytesIO(byte_image))
                sample = sample.convert('RGB')
                # sample = pickle.loads(byte_image)
                return (sample, 'raw')
            else:
                return self.handle_cache_miss(path, index)

        else:  # data was not found in redis cache
            return self.handle_cache_miss(path, index)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        # Args:
        #    index (int): Index

        # Returns:
        #    tuple: (sample, target) where target is class_index of the target class.
        # print(worker_id)

        path, target = self.samples[index]

        # start_time = datetime.datetime.now()
        sample, type = self.cache_and_get(path, index)
        # t1=threading.Thread(self.set_seen_samples, args=(index,))
        # t1.start()
        if sample != None:
            if type == "raw":

                # end_time_get = datetime.datetime.now() - start_time
                # print("fetch time= "+str(end_time_get.total_seconds()))
                sample.filename = path.split('/')[-1]
                # start_time_transform = datetime.datetime.now()
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                # self.cache_and_evict_r_ppd(sample, index)
                if self.tensor_size == 0:
                    self.tensor_size = sample.element_size() * sample.nelement()

                # end_time_trans = datetime.datetime.now() - start_time_transform
                # print("transform time= "+str(end_time_trans.total_seconds()))
                # print("total_time= "+ str((datetime.datetime.now()-start_time).total_seconds()))
                return sample, target
            elif type == 'p':
                sample.filename = path.split('/')[-1]
                if self.tensor_size == 0:
                    self.tensor_size = sample.element_size() * sample.nelement()

                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target
            elif type == 'raw_differed':
                sample.filename = path.split('/')[-1]
                # start_time_transform = datetime.datetime.now()
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                if self.tensor_size == 0:
                    self.tensor_size = sample.element_size() * sample.nelement()
                thread = threading.Thread(target=self.redis_put_tensor_, args=(sample, index),
                                          name="redis_tensor_put")
                thread.start()
                return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class QuiverDatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            sample_port: Optional[int] = 6378,
            tensor_port: Optional[int] = 6380
    ) -> None:
        super(QuiverDatasetFolder, self).__init__(root, transform=transform,
                                                  target_transform=target_transform)
        print("QuiverDatasetFolder")
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.sample_cache = redis.Redis(port=6378)
        self.tensor_cache = redis.Redis(port=6380)
        self.chunk_size = 141000 / 2  # 236471
        # self.lock = multiprocessing.Lock()
        # eviction_thread = mp.Process(target=self.evict)#threading.Thread(target=self.evict)

        # eviction_thread.start()

    def evict(self):
        all_keys = set()
        # chunk size = 20000 lets say
        chunk_size = self.chunk_size
        print('quiver evictions thread started')
        while (1):
            second_chunk = redis.Redis(port=6380)
            sample_cache = redis.Redis(port=6378)
            second_chunk_size = second_chunk.dbsize()
            if second_chunk_size > 0:  # = chunk_size:
                # print('quiver evictions 1')
                for key in second_chunk.keys():
                    try:
                        sample_cache.set(key, second_chunk.get(key))
                        all_keys.add(key)
                        # sample_cache.zadd('timestamp', {key: time.time()})
                        second_chunk.delete(key)
                    except:
                        pass
            num_keys = redis.Redis(port=6378).dbsize()
            cache_size = redis.Redis(port=6378).memory_stats()['total.allocated']
            with self.lock:
                if num_keys >= 2.5 * self.chunk_size:
                    print('eviction started')
                    keys_with_idle_times = [(key, sample_cache.object('idletime', key)) for key in all_keys]
                    # keys_by_insert_time = sample_cache.zrange('timestamp', 0,-1)

                    keys = []
                    for i in keys_with_idle_times:
                        if i != None:
                            keys.append(i)
                    # keys_with_idle_times.sort(key=lambda x: x[1], reverse=False)
                    # keys.sort(key=lambda x: x[1], reverse=False)
                    # keys_lru = [key for key, _ in keys_with_idle_times]
                    keys_lru = [key for key, _ in keys]
                    print("EVICTING KEYS:", len(keys_lru))
                    # sample_cache.delete(*keys_lru[:int(chunk_size)])

                    sample_cache.delete(*keys_lru[:int(chunk_size / 3)])

    def redis_put_thr_(self, path, index):
        # if redis.Redis(port=6378).dbsize() < self.chunk_size*2:
        index = index
        image = Image.open(path)
        # print(self.missing_samples.qsize())
        # image = self.loader(path)

        byte_stream = io.BytesIO()
        image.save(byte_stream, format=image.format)
        # byte_stream.seek(0)
        # byte_image = byte_stream.read()
        byte_image = byte_stream.getvalue()
        # byte_image = pickle.dumps(image)

        sample_cache = redis.Redis(
            port=6380)  # this only puts items in the chunking db. It will be moved to the active db later
        try:
            sample_cache.set(self.root + str(index), byte_image)
        except:
            return None
        return None

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def try_get_from_cache(self, index):
        request_rate_record = redis.Redis(port=6377)
        request_rate_record.incr('requests')
        path, target = self.samples[index]
        byte_image = self.sample_cache.get(self.root + str(index))
        if byte_image != None:
            sample = Image.open(io.BytesIO(byte_image))
            sample = sample.convert('RGB')
            sample.filename = path.split('/')[-1]
            if self.transform is not None:
                self.transform(sample)
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target
        else:
            return None

    def get_from_disk(self, index):
        miss_rate_record = redis.Redis(port=6377)
        miss_rate_record.incr('miss_rate')
        path, target = self.samples[index]
        start_time = datetime.datetime.now()
        sample = self.loader(path)
        sample.filename = path.split('/')[-1]
        self.redis_put_thr_(path, index)
        read_time = (datetime.datetime.now() - start_time).total_seconds()
        transform_start_time = datetime.datetime.now()
        if self.transform is not None:
            self.transform(sample)
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        trans_time = (datetime.datetime.now() - transform_start_time).total_seconds()
        total_time = (datetime.datetime.now() - start_time).total_seconds()
        # print(str(read_time/total_time)+", "+ str(trans_time/total_time))
        # print("total_time= " + str((datetime.datetime.now() - start_time).total_seconds()))
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class MinioDatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            sample_port: Optional[int] = 6378,
            tensor_port: Optional[int] = 6380,
            raw_cache_host: Optional[str] = "127.0.0.1",
            tensor_cache_host: Optional[str] = "127.0.0.1"
    ) -> None:
        super(MinioDatasetFolder, self).__init__(root, transform=transform,
                                                 target_transform=target_transform)
        print("MinioDatasetFolder")
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.raw_cache_host = raw_cache_host
        self.tensor_cache_host = tensor_cache_host

        self.sample_cache = redis.Redis(host=raw_cache_host, port=sample_port)
        self.tensor_cache = redis.Redis(host=tensor_cache_host, port=tensor_port)
        self.max_cache_gb = 4500000  # 425000#280000#141000#(10 * (10 ** 9))

    def redis_put_thr_(self, image, index):
        # index = index
        # image = Image.open(path)
        # print(self.missing_samples.qsize())
        # image = self.loader(path)

        if int(self.sample_cache.dbsize()) <= self.max_cache_gb:

            byte_stream = io.BytesIO()
            image.save(byte_stream, format=image.format)
            # byte_stream.seek(0)
            # byte_image = byte_stream.read()
            byte_image = byte_stream.getvalue()
            # byte_image = pickle.dumps(image)

            sample_cache = redis.Redis(host=self.raw_cache_host, port=6378)
            try:
                sample_cache.set(self.root + str(index), byte_image)
                sample_cache.close()
            except:
                return None
            return None
        else:
            return None

    def redis_put_tensor_(self, tensor, index):
        tensor_bytes = pickle.dumps(tensor)
        tensor_cache = redis.Redis(host=self.tensor_cache_host, port=6380)
        try:
            tensor_cache.set(self.root + str(index), tensor_bytes)
        except:
            return None
        return None

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def request_split(self, weights):
        total = sum(weights)
        r = random.uniform(0, total)
        upto = 0
        for i, w in enumerate(weights):
            if upto + w >= r:
                return i
            upto += w

    def handle_cache_miss(self, path, index):
        miss_rate_record = redis.Redis(port=6377)
        miss_rate_record.incr('miss_rate')
        form = 'raw'
        if form == 'raw':
            # self.redis_put_thr_(path, index)
            # self.missing_samples.put((path, index))

            # image = self.loader(path)
            image = Image.open(path)
            image_copy = copy.copy(image)
            thread = threading.Thread(target=self.redis_put_thr_, args=(image, index),
                                      name="redis_image_put")
            thread.start()
            # self.redis_put_thr_(path, index)
            image = image_copy.convert('RGB')

            # print("Index: ", index)
            # sample = image #image.convert('RGB')
            return (image, 'raw')
        else:
            image = self.loader(path)
            return (image, 'raw_differed')

    def cache_and_get(self, path, index):
        request_rate_record = redis.Redis(port=6377)
        request_rate_record.incr('requests')
        if self.tensor_cache.exists(self.root + str(index)):
            byte_tensor = self.tensor_cache.get(self.root + str(index))
            if byte_tensor != None:
                tensor = pickle.loads(byte_tensor)  # torch.tensor(list(byte_tensor), dtype=torch.uint8)
                return (tensor, 'p')
            else:
                return self.handle_cache_miss(path, index)
        elif self.sample_cache.exists(self.root + str(index)):
            byte_image = self.sample_cache.get(self.root + str(index))
            if byte_image != None:
                sample = Image.open(io.BytesIO(byte_image))
                sample = sample.convert('RGB')
                # sample = pickle.loads(byte_image)
                return (sample, 'raw')
            else:
                return self.handle_cache_miss(path, index)

        else:  # data was not found in redis cache
            return self.handle_cache_miss(path, index)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        # Args:
        #    index (int): Index

        # Returns:
        #    tuple: (sample, target) where target is class_index of the target class.
        # print(worker_id)

        path, target = self.samples[index]

        # start_time = datetime.datetime.now()
        sample, type = self.cache_and_get(path, index)
        # t1=threading.Thread(self.set_seen_samples, args=(index,))
        # t1.start()
        if sample != None:
            if type == "raw":

                # end_time_get = datetime.datetime.now() - start_time
                # print("fetch time= "+str(end_time_get.total_seconds()))
                sample.filename = path.split('/')[-1]
                # start_time_transform = datetime.datetime.now()
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                # self.cache_and_evict_r_ppd(sample, index)

                # end_time_trans = datetime.datetime.now() - start_time_transform
                # print("transform time= "+str(end_time_trans.total_seconds()))
                # print("total_time= "+ str((datetime.datetime.now()-start_time).total_seconds()))
                return sample, target
            elif type == 'p':
                sample.filename = path.split('/')[-1]

                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target
            elif type == 'raw_differed':
                sample.filename = path.split('/')[-1]
                # start_time_transform = datetime.datetime.now()
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                thread = threading.Thread(target=self.redis_put_tensor_, args=(sample, index),
                                          name="redis_tensor_put")
                thread.start()
                return sample, target

    def __len__(self) -> int:
        return len(self.samples)


import pandas


class BBModelDatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            random_transforms: Optional[Callable] = None,
            static_transforms: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            sample_port: Optional[int] = 6378,
            tensor_port: Optional[int] = 6380,
            decoded_port: Optional[int] = 6379,
            raw_cache_host: Optional[str] = 'localhost',
            tensor_cache_host: Optional[str] = 'localhost',
            decoded_cache_host: Optional[str] = 'localhost',
            xtreme_speed: Optional[bool] = True,
            fetch_time: Optional[list] = None,
            initial_cache_size: Optional[int] = 100,
            initial_cache_split: Optional[list] = [33, 33, 33]

    ) -> None:
        super(BBModelDatasetFolder, self).__init__(root, transform=transform,
                                                   target_transform=target_transform)
        print("--- BBModelDatasetFolder: ", initial_cache_size, initial_cache_split, " ---")
        self.random_transforms = random_transforms
        self.static_transofrms = static_transforms
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.sample_port = sample_port
        self.tensor_port = tensor_port
        self.decoded_port = decoded_port
        self.sample_host = raw_cache_host
        self.decoded_host = decoded_cache_host
        self.tensor_host = tensor_cache_host

        self.cache_size = 7000000  # 1170000#780000

        self.sample_cache = redis.Redis(host=self.sample_host, port=self.sample_port)
        self.tensor_cache = redis.Redis(host=self.tensor_host, port=self.tensor_port)
        self.decoded_cache = redis.Redis(host=self.decoded_host, port=self.decoded_port)

        self.raw_cache_alloc = initial_cache_split[0]
        self.decoded_cache_alloc = initial_cache_split[1]
        self.augmented_cache_alloc = initial_cache_split[2]

        self.raw_cache_alloc_base = self.raw_cache_alloc
        self.decoded_cache_alloc_base = self.decoded_cache_alloc
        self.augmented_cache_alloc_base = self.augmented_cache_alloc

        self.cache_alloc = initial_cache_size * 1000000000

        tensor_cache_size = int(((initial_cache_split[2] / 100) * self.cache_alloc) / 2)
        decoded_cache_size = int((initial_cache_split[1] / 100) * self.cache_alloc) + tensor_cache_size
        sample_cache_size = int((initial_cache_split[0] / 100) * self.cache_alloc)
        self.sample_cache.execute_command('CONFIG', 'SET', 'maxmemory', sample_cache_size)

        self.decoded_cache.execute_command('CONFIG', 'SET', 'maxmemory', decoded_cache_size)

        self.tensor_cache.execute_command('CONFIG', 'SET', 'maxmemory', tensor_cache_size)

        self.tensor_cache_size = self.tensor_cache.info()['maxmemory']
        self.decoded_cache_size = self.decoded_cache.info()['maxmemory']
        self.sample_cache_size = self.sample_cache.info()['maxmemory']

        self.sample_cache_count = redis.Redis(host=self.sample_host, port=6388)
        self.tensor_cache_count = redis.Redis(host=self.tensor_host, port=6390)
        self.decoded_cache_count = redis.Redis(host=self.decoded_host, port=6389)

        self.raw_img_size = 0
        self.decoded_size = 0
        self.tensor_size = 0

        self.weights_set = False
        self.weights = [100, 0]

        self.tensor_cache_max_ref_count = 5
        self.decode_cache_max_ref_count = 5000000
        self.sample_cache_max_ref_count = 5000000

        self.time_from_augmented = fetch_time[0]
        self.time_from_decoded = fetch_time[1]
        self.time_from_raw = fetch_time[2]
        self.time_from_storage = fetch_time[3]

        self.get_form_counter = 0
        self.tensor_cache_info = None
        self.decoded_cache_info = None
        self.raw_cache_info = None

        self.evict_augmented = False
        self.evict_decoded = False
        self.evict_raw = False

        self.direction_up = True
        self.throughput_drop_count = 0
        self.cursor_dec = 0
        self.cursor_raw = 0
        self.cursor_aug = 0
        self.autotune_simple_dir = 1
        self.epoch = 0

    def redis_put_thr_(self, image, index):
        byte_stream = io.BytesIO()
        image.save(byte_stream, format=image.format)
        byte_image = byte_stream.getvalue()

        sample_cache = self.sample_cache
        sample_cache_count = self.sample_cache_count
        try:
            sample_cache.set(index, byte_image)
            sample_cache_count.set(index, 0)
        except:
            return None
        return None

    def redis_put_tensor_(self, tensor, index):
        tensor_bytes = pickle.dumps(tensor)
        tensor_cache = self.tensor_cache
        tensor_cache_count = self.tensor_cache_count
        try:
            tensor_cache.set(index, tensor_bytes)
            tensor_cache_count.set(index, 0)
        except:
            return None
        return None

    def redis_put_decoded_(self, tensor, index):
        tensor_bytes = pickle.dumps(tensor)
        decoded_cache = self.decoded_cache
        decoded_cache_count = self.decoded_cache_count
        try:
            decoded_cache.set(index, tensor_bytes)
            decoded_cache_count.set(index, 0)
        except:
            return None
        return None

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def request_split(self, weights):
        total = sum(weights)
        r = random.uniform(0, total)
        upto = 0
        for i, w in enumerate(weights):
            if upto + w >= r:
                return i
            upto += w

    def get_form(self):
        self.get_form_counter += 1
        if self.get_form_counter >= 5000 or self.tensor_cache_info is None:
            self.tensor_cache_info = self.tensor_cache.info()
            self.decoded_cache_info = self.decoded_cache.info()
            self.raw_cache_info = self.sample_cache.info()
            self.get_form_counter = 0

        tensor_maxmem = int(self.tensor_cache_info['maxmemory'])
        decoded_maxmem = int(self.decoded_cache_info['maxmemory'])
        raw_maxmem = int(self.raw_cache_info['maxmemory'])

        tensor_used_mem = self.tensor_cache_info['used_memory']
        decoded_used_mem = self.decoded_cache_info['used_memory']
        raw_used_mem = self.raw_cache_info['used_memory']

        if tensor_used_mem < tensor_maxmem and self.evict_augmented == False:
            return 2
        elif decoded_used_mem < decoded_maxmem and self.evict_decoded == False:
            return 1
        elif raw_used_mem < raw_maxmem and self.evict_raw == False:
            return 0
        else:
            return None

    def is_space_available(self, cache):
        cache_info = cache.info()

        maxmem = int(cache_info['maxmemory'])

        used_mem = cache_info['used_memory']
        return maxmem > used_mem

    def handle_cache_miss(self, path, index):
        miss_rate_record = redis.Redis(port=6377)
        miss_rate_record.incr('miss_rate')
        if self.raw_img_size == 0:
            self.raw_img_size = os.path.getsize(path)
        if self.raw_img_size != 0 and self.tensor_size != 0:
            if self.weights[0] == 100:
                print("[WARN]: Black box model is not implemented yet")

        self.weights = [20, 80, 0]  # overriding the model for testing
        form_options = ['raw', 'dec', 'p']

        if sum(self.weights) == 0:
            print('sum of cache splits should be 100 but was 0')
            form = None
        else:
            chosen_form_idx = self.get_form()  # self.request_split(self.weights)
            if chosen_form_idx != None:
                form = form_options[chosen_form_idx]  # (can be one of 'raw' or 'p')
            else:
                form = "no cache"

        if form == 'raw':
            # [BBModel] start measuring fetch time
            # fetch_start_time = time.time()*1000
            image = Image.open(path)
            # [BBModel] end measuring fetch time
            # fetch_end_time = time.time()*1000
            # fetch_time = fetch_end_time - fetch_start_time
            # self.fetch_time.update(fetch_time)

            image_copy = copy.copy(image)
            thread = threading.Thread(target=self.redis_put_thr_, args=(image, index),
                                      name="redis_image_put")
            thread.start()
            image = image_copy.convert('RGB')
            return (image, 'raw')
        elif form == 'p':
            # [BBModel] start measuring fetch time
            # fetch_start_time = time.time() * 1000
            image = self.loader(path)
            # fetch_end_time = time.time() * 1000
            # fetch_time = fetch_end_time - fetch_start_time
            # self.fetch_time.update(fetch_time)
            # [BBModel] end measuring fetch time
            return (image, 'raw_differed')
        elif form == 'dec':
            # [BBModel] start measuring fetch time
            # fetch_start_time = time.time() * 1000
            image = self.loader(path)
            # fetch_end_time = time.time() * 1000
            # fetch_time = fetch_end_time - fetch_start_time
            # self.fetch_time.update(fetch_time)
            # [BBModel] end measuring fetch time
            return (image, 'raw_dec_differed')
        else:
            # [BBModel] start measuring fetch time
            # fetch_start_time = time.time() * 1000
            image = Image.open(path)
            # fetch_end_time = time.time() * 1000
            # fetch_time = fetch_end_time - fetch_start_time
            # self.fetch_time.update(fetch_time)
            # [BBModel] end measuring fetch time
            image = image.convert('RGB')
            return (image, 'raw')

    def cache_and_get(self, path, index):
        request_rate_record = redis.Redis(port=6377)
        request_rate_record.incr('requests')
        if self.tensor_cache.exists(index):
            # time_from_augment_start = time.time()
            byte_tensor = self.tensor_cache.get(index)
            if self.evict_augmented == True:
                # self.tensor_cache.unlink(index)
                # self.decoded_cache.unlink(index)
                self.tensor_cache_count.publish('delete', index)
                self.decoded_cache_count.publish('delete', index)
                if self.tensor_cache.info()['used_memory'] <= self.tensor_cache_size:
                    print("done with deletion: aug")
                    self.evict_augmented = False
                    self.tensor_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                                                      str(self.tensor_cache_size)
                                                      )
            if byte_tensor != None:
                tensor = pickle.loads(byte_tensor)
                ref_count = self.tensor_cache_count.incr(index)
                if ref_count >= self.tensor_cache_max_ref_count:
                    self.tensor_cache_count.publish('delete', index)
                # time_from_augment_end = time.time()
                # time_from_augment = time_from_augment_end - time_from_augment_start
                # self.time_from_augmented.update(time_from_augment)
                return (tensor, 'p')
            else:
                print('[WARN] Possibly corrupt data in redis tensor cache')
                return self.handle_cache_miss(path, index)

        elif self.decoded_cache.exists(index):
            # time_from_decoded_start = time.time()
            byte_tensor = self.decoded_cache.get(index)
            if self.evict_decoded == True:
                # self.decoded_cache.unlink(index)
                self.decoded_cache_count.publish('delete', index)
                if self.decoded_cache.info()['used_memory'] <= self.decoded_cache_size:
                    print("done with deletion: dec")
                    self.evict_decoded = False
                    self.decoded_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                                                       str(self.decoded_cache_size)
                                                       )
            if byte_tensor != None:
                tensor = pickle.loads(byte_tensor)
                tensor = self.random_transforms(tensor)
                if self.is_space_available(self.tensor_cache):  # is there free space in augmented cache?
                    self.redis_put_tensor_(tensor, index)
                    # cache in augmented cache but dont delete from decoded cache
                # MODEL IMPLEMENTATION HERE
                """
                self.weights = [20, 80, 0]  # overriding the model for testing


                form_options = ['raw', 'dec', 'p']
                chosen_form_idx = self.request_split(self.weights)
                form = form_options[chosen_form_idx]
                if form =='p':
                    thread = threading.Thread(target=self.redis_put_tensor_, args=(tensor, index),
                                          name="redis_tensor_put")
                    thread.start()
                """
                ref_count = self.decoded_cache_count.incr(index)
                if ref_count >= self.decode_cache_max_ref_count:
                    self.decoded_cache_count.publish('delete', index)

                # time_from_decoded_end = time.time()
                # time_from_decoded = time_from_decoded_end - time_from_decoded_start
                # self.time_from_decoded.update(time_from_decoded)
                return (tensor, 'p')
            else:
                print('[WARN] Possibly corrupt data in redis decoded cache')
                return self.handle_cache_miss(path, index)

        # elif self.sample_cache.exists(index) and (ref_count_value := self.sample_cache.get(str(index) + '_ref_count')) is not None and int(ref_count_value or 0) < self.max_ref_count:
        elif self.sample_cache.exists(index):
            # time_from_raw_start = time.time()
            byte_image = self.sample_cache.get(index)
            if self.evict_raw == True:
                # self.sample_cache.unlink(index)
                self.sample_cache_count.publish('delete', index)
                if self.sample_cache.info()['used_memory'] <= self.sample_cache_size:
                    self.evict_raw = False
                    self.sample_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                                                      str(self.sample_cache_size)
                                                      )
            if byte_image != None:
                sample = Image.open(io.BytesIO(byte_image))
                sample = sample.convert('RGB')
                # sample = pickle.loads(byte_image)
                # self.sample_cache.incr(str(index) + '_ref_count')
                ref_count = self.sample_cache_count.incr(index)
                if ref_count >= self.sample_cache_max_ref_count:
                    self.sample_cache_count.publish('delete', index)
                # time_from_raw_end = time.time()
                # time_from_raw = time_from_raw_end - time_from_raw_start
                # self.time_from_raw.update(time_from_raw)
                return (sample, 'raw')
            else:
                print('[WARN] Possibly corrupt data in redis sample cache')
                return self.handle_cache_miss(path, index)
        else:  # data was not found in redis cache
            # time_from_storage_start = time.time()
            sample, form = self.handle_cache_miss(path, index)
            # time_from_storage_end = time.time()
            # time_from_storage = time_from_storage_end - time_from_storage_start
            # self.time_from_storage.update(time_from_storage)
            return (sample, form)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        # Args:
        #    index (int): Index

        # Returns:
        #    tuple: (sample, target) where target is class_index of the target class.
        # print(worker_id)

        path, target = self.samples[index]
        if self.epoch == 0:
            fetch_time_s = time.time()
        sample, type = self.cache_and_get(path, index)
        if self.epoch == 0:
            fetch_time_e = time.time()
            fetch_time = fetch_time_e - fetch_time_s
            self.time_from_storage.update(fetch_time)
        if index % (50000) == 0:
            # self.time_from_storage.update_averages([self.time_from_augmented,
            #                                 #self.time_from_decoded,
            #                                 self.time_from_raw,
            #                                 self.time_from_storage])
            pass

        if sample != None:
            if type == "raw":
                # time_from_raw_start = time.time()
                sample.filename = path.split('/')[-1]
                if self.transform is not None:
                    if self.epoch == 0:
                        static_trans_time_s = time.time()
                    sample = self.static_transofrms(sample)
                    if self.epoch == 0:
                        static_trans_time_e = time.time()
                        static_trans_time = static_trans_time_e - static_trans_time_s
                        self.time_from_raw.update(static_trans_time)
                    if self.is_space_available(self.decoded_cache):
                        self.redis_put_decoded_(sample, index)
                    if self.epoch == 0:
                        random_trans_time_s = time.time()
                    sample = self.random_transforms(sample)
                    if self.epoch == 0:
                        random_trans_time_e = time.time()
                        random_trans_time = random_trans_time_e - random_trans_time_s
                        self.time_from_augmented.update(random_trans_time)
                    if self.is_space_available(self.tensor_cache):
                        self.redis_put_tensor_(sample, index)

                    # sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                # time_from_raw_end = time.time()
                # time_from_raw = time_from_raw_end - time_from_raw_start
                # self.time_from_raw.update(time_from_raw)
                return sample, target
            elif type == 'p':
                sample.filename = path.split('/')[-1]
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target
            elif type == 'raw_differed':
                # time_from_storage_start =time.time()
                sample.filename = path.split('/')[-1]
                if self.static_transofrms is not None:
                    sample = self.static_transofrms(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                thread = threading.Thread(target=self.redis_put_decoded_, args=(sample, index),
                                          name="redis_decoded_put")
                thread.start()

                if self.random_transforms is not None:
                    sample = self.random_transforms(sample)

                thread = threading.Thread(target=self.redis_put_tensor_, args=(sample, index),
                                          name="redis_tensor_put")
                thread.start()
                # time_from_storage_end = time.time()
                # time_from_storage = time_from_storage_end - time_from_storage_start
                # self.time_from_storage.update(time_from_storage)
                return sample, target
            elif type == "raw_dec_differed":
                # time_from_storage_start = time.time()
                sample.filename = path.split('/')[-1]
                if self.static_transofrms is not None:
                    sample = self.static_transofrms(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)

                thread = threading.Thread(target=self.redis_put_decoded_, args=(sample, index),
                                          name="redis_decoded_put")
                thread.start()
                sample = self.random_transforms(sample)
                # time_from_storage_end = time.time()
                # time_from_storage = time_from_storage_end - time_from_storage_start
                # self.time_from_storage.update(time_from_storage)
                return sample, target

    def tune_allocations_storm(self, cache_size, cache_split):
        print("new memory allocation:", cache_size)
        print("new cache split:", cache_split)
        self.sample_cache = redis.Redis(host=self.sample_host, port=self.sample_port)
        self.tensor_cache = redis.Redis(host=self.tensor_host, port=self.tensor_port)
        self.decoded_cache = redis.Redis(host=self.decoded_host, port=self.decoded_port)

        tensor_cache_size = int((((cache_split[2] / 100) * cache_size) / 2))
        decoded_cache_size = int(((cache_split[1] / 100) * cache_size + ((cache_split[2] / 100) * cache_size) / 2))
        sample_cache_size = int(((cache_split[0] / 100) * cache_size))
        print(sample_cache_size, decoded_cache_size, tensor_cache_size)

        self.tensor_cache_size = tensor_cache_size
        self.decoded_cache_size = decoded_cache_size
        self.sample_cache_size = sample_cache_size

        self.sample_cache.execute_command('CONFIG', 'SET', 'maxmemory', sample_cache_size)

        self.decoded_cache.execute_command('CONFIG', 'SET', 'maxmemory', decoded_cache_size)

        self.tensor_cache.execute_command('CONFIG', 'SET', 'maxmemory', tensor_cache_size)

    def move_samples(self, destination, approx_samples_to_add):
        print("Number of samples to move:", approx_samples_to_add, "destination:", destination)

        def process_decoded_key(key):
            decoded_sample = self.decoded_cache.get(key)
            decoded_sample = pickle.loads(decoded_sample)
            augmented_sample = self.random_transforms(decoded_sample)
            augmented_sample = pickle.dumps(augmented_sample)
            try:
                if not self.tensor_cache.exists(key):
                    self.tensor_cache.set(key, augmented_sample)
            except:
                pass

        def process_raw_key(key):
            raw_sample = self.sample_cache.get(key)
            raw_sample = Image.open(io.BytesIO(raw_sample))
            raw_sample = raw_sample.convert('RGB')
            static_sample = self.static_transofrms(raw_sample)
            static_sample_pickled = pickle.dumps(static_sample)
            # try:
            if not self.decoded_cache.exists(key):
                self.decoded_cache.set(key, static_sample_pickled)
            # except:
            #    pass
            augmented_sample = self.random_transforms(static_sample)
            augmented_sample = pickle.dumps(augmented_sample)
            # try:
            if not self.tensor_cache.exists(key):
                self.tensor_cache.set(key, augmented_sample)
                # self.sample_cache.unlink(key)
            # except:
            #    pass

        def process_sample_key(key):
            raw_sample = self.sample_cache.get(key)
            raw_sample = Image.open(io.BytesIO(raw_sample))
            raw_sample = raw_sample.convert('RGB')
            decoded_sample = self.static_transofrms(raw_sample)
            decoded_sample = pickle.dumps(decoded_sample)
            try:
                if not self.decoded_cache.exists(key):
                    self.decoded_cache.set(key, decoded_sample)
                    # self.sample_cache.unlink(key)
            except:
                pass

        if destination == "augmented" and approx_samples_to_add > 0:

            attempts = 0
            while attempts < 3 and self.tensor_cache.dbsize() < len(self) and self.tensor_cache.info()[
                'used_memory'] < (0.95 * self.tensor_cache.info()['maxmemory']):

                attempts += 1
                dbsize_init = self.tensor_cache.dbsize()
                if self.decoded_cache.dbsize() > 0:
                    keys_pp = self.decoded_cache.scan(cursor=self.cursor_dec, count=approx_samples_to_add)
                    self.cursor_dec = keys_pp[0]
                    keys_pp = keys_pp[1][:approx_samples_to_add]
                    print("Start bulk loading data A", len(keys_pp))

                    t_a = time.time()
                    # for key in keys_pp[1]:
                    #    process_decoded_key(key)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                        executor.map(process_decoded_key, keys_pp, chunksize=32)
                    print("Completed bulk loading data A in ", time.time() - t_a, "added samples:",
                          self.tensor_cache.dbsize() - dbsize_init)
                else:
                    print('data A empty')
                dbsize_end = self.tensor_cache.dbsize()
                keys_added = dbsize_end - dbsize_init
                if keys_added < approx_samples_to_add:
                    keys_pp_raw = self.sample_cache.scan(cursor=self.cursor_raw,
                                                         count=int(approx_samples_to_add - keys_added))
                    self.cursor_raw = keys_pp_raw[0]
                    keys_pp_raw = keys_pp_raw[1][:approx_samples_to_add]
                    print("Start bulk loading data B", len(keys_pp_raw))

                    t_b = time.time()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                        executor.map(process_raw_key, keys_pp_raw, chunksize=32)
                        # x=executor.map(process_raw_key_chunk, chunks_pp_raw)
                    print("Completed bulk loading data B in", time.time() - t_b, "added samples:",
                          self.tensor_cache.dbsize() - dbsize_init)
                dbsize_end = self.tensor_cache.dbsize()
                keys_added = dbsize_end - dbsize_init
                approx_samples_to_add = int(approx_samples_to_add - keys_added)

        elif destination == "decoded" and approx_samples_to_add > 0:

            attempts = 0
            while attempts < 3 and self.tensor_cache.dbsize() < len(self) and self.decoded_cache.info()[
                'used_memory'] < (0.95 * self.decoded_cache.info()['maxmemory']):
                # (0.95 * self.decoded_cache.info()['maxmemory'] <= self.decoded_cache.info()['used_memory'] <= 1.05*self.decoded_cache.info()['maxmemory']):
                attempts += 1
                dbsize_init = self.decoded_cache.dbsize()
                keys_pp = self.sample_cache.scan(cursor=self.cursor_raw, count=approx_samples_to_add)
                self.cursor_raw = keys_pp[0]
                keys_pp = keys_pp[1][:approx_samples_to_add]
                print("Start bulk loading data C", len(keys_pp))

                t_c = time.time()
                # for key in keys_pp[1]:
                #    process_sample_key(key)
                with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                    executor.map(process_sample_key, keys_pp, chunksize=32)
                    # x=executor.map(process_sample_key_chunk, chunks_pp)
                print("Completed bulk loading data in cache C in ", time.time() - t_c, "added samples:",
                      self.decoded_cache.dbsize() - dbsize_init)
                approx_samples_to_add = int(approx_samples_to_add - (self.decoded_cache.dbsize() - dbsize_init))

        elif destination == "raw":
            print("to do")

    def tune_allocations_smooth(self, cache_size, cache_split):
        print("tune_allocations_smooth")
        print("new memory allocation:", cache_size)
        print("new cache split:", cache_split)
        self.sample_cache = redis.Redis(host=self.sample_host, port=self.sample_port)
        self.tensor_cache = redis.Redis(host=self.tensor_host, port=self.tensor_port)
        self.decoded_cache = redis.Redis(host=self.decoded_host, port=self.decoded_port)

        self.raw_cache_info = self.sample_cache.info()
        self.tensor_cache_info = self.tensor_cache.info()
        self.decoded_cache_info = self.decoded_cache.info()

        tensor_cache_size = int((((cache_split[2] / 100) * cache_size) / 2))
        tensor_cache_size_decoded_portion = int((((cache_split[2] / 100) * cache_size) / 2))

        decoded_cache_size = int((cache_split[1] / 100) * cache_size)
        sample_cache_size = int(((cache_split[0] / 100) * cache_size))
        print(sample_cache_size, decoded_cache_size, tensor_cache_size)

        self.tensor_cache_size = tensor_cache_size
        self.decoded_cache_size = decoded_cache_size + tensor_cache_size_decoded_portion
        self.sample_cache_size = sample_cache_size

        if self.tensor_cache_size < self.tensor_cache_info['maxmemory']:
            if tensor_cache_size > self.tensor_cache_info['used_memory']:
                self.tensor_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                                                  self.tensor_cache_size
                                                  )
                self.decoded_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                                                   self.decoded_cache_size
                                                   )
                self.tensor_cache_info = self.tensor_cache.info()
                self.decoded_cache_info = self.decoded_cache.info()
            else:
                # self.evict_augmented = True
                self.tensor_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                                                  self.tensor_cache_size
                                                  )
                self.decoded_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                                                   self.decoded_cache_size
                                                   )
                self.tensor_cache_info = self.tensor_cache.info()
                self.decoded_cache_info = self.decoded_cache.info()
                # self.decoded_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                #                                   str(decoded_cache_size)
                #                                   )
                print("evict samples as they are used: AUG")
        if self.tensor_cache_size > self.tensor_cache_info['maxmemory']:
            prev_cache_size = self.tensor_cache_info['maxmemory']
            self.tensor_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                                              self.tensor_cache_size
                                              )
            self.decoded_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                                               self.decoded_cache_size
                                               )
            self.tensor_cache_info = self.tensor_cache.info()
            # self.decoded_cache_info = self.decoded_cache.info()
            if self.tensor_cache.dbsize() > 1:
                avg_sample_size = self.tensor_cache_info['used_memory'] / self.tensor_cache.dbsize()  # could be 0
                memory_bytes_increased = self.tensor_cache_size - prev_cache_size
                approx_samples_to_add = int(memory_bytes_increased / avg_sample_size)
                approx_samples_to_add = min(approx_samples_to_add, (len(self.samples) - self.tensor_cache.dbsize()))
                self.move_samples('augmented', approx_samples_to_add)
            else:
                """
                keys_init = self.sample_cache.scan(cursor=0, match='*', count=50)
                keys_pp = keys_init[1][:50]
                for key in keys_pp:
                    raw_sample = self.sample_cache.get(key)
                    raw_sample = Image.open(io.BytesIO(raw_sample))
                    raw_sample = raw_sample.convert('RGB')
                    augmented_sample = self.transform(raw_sample)
                    augmented_sample = pickle.dumps(augmented_sample)
                    try:
                        self.tensor_cache.set(key, augmented_sample)
                    except:
                        pass
                """
                avg_sample_size = 200000  # 15*(self.raw_cache_info['used_memory'] / self.sample_cache.dbsize())  # could be 0
                memory_bytes_increased = self.tensor_cache_size - prev_cache_size
                # print("--->", self.decoded_cache.dbsize(), self.decoded_cache_info['used_memory'])
                # print("bytes inc:", memory_bytes_increased, "avg samp size:", avg_sample_size)
                approx_samples_to_add = int(memory_bytes_increased / avg_sample_size)
                # print("1) samples to move:", approx_samples_to_add)
                approx_samples_to_add = min(approx_samples_to_add, (len(self.samples) - self.decoded_cache.dbsize()))
                self.move_samples('augmented', approx_samples_to_add)
        ###############
        if self.decoded_cache_size < self.decoded_cache_info['maxmemory']:  # tensor memory shrinking
            if decoded_cache_size > self.decoded_cache_info['used_memory']:
                self.decoded_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                                                   self.decoded_cache_size
                                                   )
                self.decoded_cache_info = self.decoded_cache.info()
            else:
                # self.decoded_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                #                                   str(decoded_cache_size)
                #                                   )
                # self.evict_decoded = True
                self.decoded_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                                                   self.decoded_cache_size
                                                   )
                self.decoded_cache_info = self.decoded_cache.info()
                print("evict samples as they are used: DEC")

        if self.decoded_cache_size > self.decoded_cache_info['maxmemory']:
            prev_cache_size = self.decoded_cache_info['maxmemory']
            self.decoded_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                                               self.decoded_cache_size
                                               )
            self.decoded_cache_info = self.decoded_cache.info()
            if self.decoded_cache.dbsize() > 1:
                avg_sample_size = self.decoded_cache_info['used_memory'] / self.decoded_cache.dbsize()  # could be 0
                memory_bytes_increased = self.decoded_cache_size - prev_cache_size
                # print("--->", self.decoded_cache.dbsize(), self.decoded_cache_info['used_memory'])
                # print("bytes inc:", memory_bytes_increased, "avg samp size:", avg_sample_size)
                approx_samples_to_add = int(memory_bytes_increased / avg_sample_size)
                # print("1) samples to move:", approx_samples_to_add)
                approx_samples_to_add = min(approx_samples_to_add, (len(self.samples) - self.decoded_cache.dbsize()))
                self.move_samples('decoded', approx_samples_to_add)
            else:
                """
                keys_init = self.sample_cache.scan(cursor=0, match='*', count=500)
                keys_pp = keys_init[1][:500]
                for key in keys_pp:
                    raw_sample = self.sample_cache.get(key)
                    raw_sample = Image.open(io.BytesIO(raw_sample))
                    raw_sample = raw_sample.convert('RGB')
                    decoded_sample = self.static_transofrms(raw_sample)
                    decoded_sample = pickle.dumps(decoded_sample)
                    try:
                        self.decoded_cache.set(key, decoded_sample)
                    except:
                        pass
                """
                avg_sample_size = 200000  # 15*(self.raw_cache_info['used_memory'] / self.sample_cache.dbsize())  # could be 0
                memory_bytes_increased = self.decoded_cache_size - prev_cache_size
                # print("--->", self.decoded_cache.dbsize(), self.decoded_cache_info['used_memory'], decoded_cache_size)
                # print("bytes inc:", memory_bytes_increased, "avg samp size:", avg_sample_size)
                approx_samples_to_add = int(memory_bytes_increased / avg_sample_size)
                # print("1) samples to move:", approx_samples_to_add)
                approx_samples_to_add = min(approx_samples_to_add, (len(self.samples) - self.decoded_cache.dbsize()))

                self.move_samples('decoded', approx_samples_to_add)
        ################

        if self.sample_cache_size < self.raw_cache_info['maxmemory']:  # tensor memory shrinking
            if sample_cache_size > self.raw_cache_info['used_memory']:
                self.sample_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                                                  sample_cache_size
                                                  )
                self.raw_cache_info = self.sample_cache.info()
            else:
                print("evict samples as they are used: RAW")
                # self.evict_raw = True
                self.sample_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                                                  sample_cache_size
                                                  )
                self.raw_cache_info = self.sample_cache.info()
                # self.decoded_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                #                                   str(decoded_cache_size)
                #                                   )
        if self.sample_cache_size > self.raw_cache_info['maxmemory']:
            prev_cache_size = self.raw_cache_info['maxmemory']
            self.sample_cache.execute_command('CONFIG', 'SET', 'maxmemory',
                                              sample_cache_size
                                              )

    def autotune(self, datarame_file, dsi_hash, pid, epoch):
        if epoch < 1:
            self.epoch = epoch + 1
            print("no autotuning until epoch >= 1")
            return
        print("Force apply previous tuning states")
        self.tune_allocations_storm(self.cache_alloc,
                                    [self.raw_cache_alloc, self.decoded_cache_alloc,
                                     self.augmented_cache_alloc])
        print("autotune")
        df = pandas.read_csv(datarame_file)
        all_dataset_jobs = df.loc[df['dsi_hash'] == str(dsi_hash)]
        grouped_by_pid = all_dataset_jobs.groupby('pid', group_keys=True)
        aggregate_throughput_current = 0
        aggregate_throughput_previous = 0
        aggregate_throughput_previous_ma = 0
        for name, data in grouped_by_pid:
            data_1 = data.sort_values(by='profiled_timestamp', ascending=False).head(2)
            aggregate_throughput_current += data_1.iloc[0]['throughput']
            if len(data_1) > 1:
                throughput_trend_ma = data['throughput'].rolling(window=2).mean().iloc[-1]
                if not pandas.isna(throughput_trend_ma):
                    aggregate_throughput_previous += throughput_trend_ma  # data_1.iloc[1]['moving_avg']
                else:
                    aggregate_throughput_previous += data_1.iloc[1]['throughput']
                # aggregate_throughput_previous += data_1.iloc[1]['throughput']

        last_3_splits = all_dataset_jobs.tail(3)['cache_split']
        all_same = last_3_splits.nunique() == 1

        throughput_diff = aggregate_throughput_current - aggregate_throughput_previous
        print(aggregate_throughput_current, aggregate_throughput_previous)
        load_avg = psutil.getloadavg()[1]
        num_cores = os.cpu_count()
        cach_utils = []

        raw_cache_info = self.sample_cache.info()
        cach_utils.append((((raw_cache_info['maxmemory'] - raw_cache_info['used_memory']) / raw_cache_info[
            'maxmemory']) * 100, 'raw')) if raw_cache_info['maxmemory'] > 0 else -1

        dec_cache_info = self.decoded_cache.info()
        cach_utils.append((((dec_cache_info['maxmemory'] - dec_cache_info['used_memory']) / dec_cache_info[
            'maxmemory']) * 100, 'dec')) if dec_cache_info['maxmemory'] > 0 else -1

        aug_cache_info = self.tensor_cache.info()

        # cach_utils.append((((aug_cache_info['maxmemory'] - aug_cache_info['used_memory']) / aug_cache_info[
        #    'maxmemory']) * 100, 'aug')) if aug_cache_info['maxmemory'] > 0 else -1

        # if abs(aggregate_throughput_current - aggregate_throughput_previous) / aggregate_throughput_current * 100 < 4:
        #    print("Autotune stabilizing split")
        #    return
        if aggregate_throughput_current > aggregate_throughput_previous:
            if self.direction_up == True:
                print("moving up")

                # If there is a high load avg on the CPU and enough memory is available. What does really define high LA and
                # high memory.
                if ((load_avg > (num_cores + 5)) and \
                        ((raw_cache_info['maxmemory'] - raw_cache_info['used_memory'])
                         + (dec_cache_info['maxmemory'] - dec_cache_info['used_memory'])
                         + (aug_cache_info['maxmemory'] - aug_cache_info['used_memory'])) > (
                                0.65 * (self.cache_alloc))):
                    # if (load_avg > (num_cores)):
                    destination = 'aug'
                else:
                    destination = 'dec'

                step_size = min(40, 2 * ((
                                                     aggregate_throughput_current - aggregate_throughput_previous) / aggregate_throughput_current * 100))
                # step_size = ((aggregate_throughput_current - aggregate_throughput_previous) / aggregate_throughput_current * 100)

                cache_utils_sorted = sorted(cach_utils, reverse=True, key=lambda x: x[0])
                source = None
                for i in range(len(cache_utils_sorted)):
                    source = cache_utils_sorted[i][1]
                    if source == destination:
                        source = None
                    else:
                        break
                if source == None:
                    print("Source is None, will not do anything")
                    return
                    # raise Exception("Source is None")

                print("source:", source, "destination:", destination)
                self.augmented_cache_alloc_base = self.augmented_cache_alloc
                self.decoded_cache_alloc_base = self.decoded_cache_alloc
                self.raw_cache_alloc_base = self.raw_cache_alloc

                if source == 'dec':
                    decoded_cache_alloc = self.decoded_cache_alloc - step_size
                    if decoded_cache_alloc < 0:
                        step_size = step_size - (abs(decoded_cache_alloc))
                    self.decoded_cache_alloc = self.decoded_cache_alloc - step_size
                elif source == 'raw':
                    raw_cache_alloc = self.raw_cache_alloc - step_size
                    if raw_cache_alloc < 0:
                        step_size = step_size - (abs(raw_cache_alloc))
                    self.raw_cache_alloc = self.raw_cache_alloc - step_size

                extra_allocation = 0
                if destination == 'aug':
                    augmented_cache_alloc = self.augmented_cache_alloc + step_size
                    if augmented_cache_alloc > 100:
                        extra_allocation = augmented_cache_alloc - 100
                        step_size = step_size - (augmented_cache_alloc - 100)
                    self.augmented_cache_alloc = self.augmented_cache_alloc + step_size
                    if extra_allocation > 0 and (self.decoded_cache_alloc + extra_allocation) < 100:
                        self.decoded_cache_alloc = self.decoded_cache_alloc + extra_allocation
                    elif extra_allocation > 0:
                        self.raw_cache_alloc = self.raw_cache_alloc + extra_allocation

                elif destination == 'dec':
                    decoded_cache_alloc = self.decoded_cache_alloc + step_size
                    if decoded_cache_alloc > 100:
                        extra_allocation = decoded_cache_alloc - 100
                        step_size = step_size - (decoded_cache_alloc - 100)
                    self.decoded_cache_alloc = self.decoded_cache_alloc + step_size
                    if extra_allocation > 0 and (self.augmented_cache_alloc + extra_allocation) < 100:
                        self.augmented_cache_alloc = self.augmented_cache_alloc + extra_allocation
                    elif extra_allocation > 0:
                        self.raw_cache_alloc = self.raw_cache_alloc + extra_allocation

                self.tune_allocations_smooth(self.cache_alloc,
                                             [self.raw_cache_alloc, self.decoded_cache_alloc,
                                              self.augmented_cache_alloc])
            else:
                print("moving down")
                aug_cache_info = self.tensor_cache.info()
                dec_cache_info = self.decoded_cache.info()
                step_size = abs(
                    aggregate_throughput_current - aggregate_throughput_previous) / aggregate_throughput_current * 100
                # step_size = min(20, int(((aggregate_throughput_current - aggregate_throughput_previous) / aggregate_throughput_previous) * 100))
                if aug_cache_info['used_memory'] > 100000000:
                    source = 'aug'
                    destination = 'dec'
                elif dec_cache_info['used_memory'] > 100000000:
                    source = 'dec'
                    destination = 'raw'
                else:
                    source = 'raw'
                    destination = 'raw'
                    print("[IMPORTANT INFO]: Looks like caching may not be faster than the backing storage")

                self.augmented_cache_alloc_base = self.augmented_cache_alloc
                self.decoded_cache_alloc_base = self.decoded_cache_alloc
                self.raw_cache_alloc_base = self.raw_cache_alloc
                if source == destination:
                    print("Source and destination is the same, will not do anything")
                    return

                if source == 'aug':
                    augmented_cache_alloc = self.augmented_cache_alloc - step_size
                    if augmented_cache_alloc < 0:
                        step_size = step_size - (abs(augmented_cache_alloc))
                    self.augmented_cache_alloc = self.augmented_cache_alloc - step_size
                elif source == 'dec':
                    decoded_cache_alloc = self.decoded_cache_alloc - step_size
                    if decoded_cache_alloc < 0:
                        step_size = step_size - (abs(decoded_cache_alloc))
                    self.decoded_cache_alloc = self.decoded_cache_alloc - step_size

                extra_allocation = 0
                if destination == 'raw':
                    raw_cache_alloc = self.raw_cache_alloc + step_size
                    if raw_cache_alloc > 100:
                        extra_allocation = raw_cache_alloc - 100
                        step_size = step_size - (raw_cache_alloc - 100)
                    self.raw_cache_alloc = self.raw_cache_alloc + step_size
                    if extra_allocation > 0 and (self.augmented_cache_alloc + extra_allocation) < 100:
                        self.augmented_cache_alloc = self.augmented_cache_alloc + extra_allocation
                    elif extra_allocation > 0:
                        self.decoded_cache_alloc = self.decoded_cache_alloc + extra_allocation

                elif destination == 'dec':
                    decoded_cache_alloc = self.decoded_cache_alloc + step_size
                    if decoded_cache_alloc > 100:
                        extra_allocation = decoded_cache_alloc - 100
                        step_size = step_size - (decoded_cache_alloc - 100)
                    self.decoded_cache_alloc = self.decoded_cache_alloc + step_size
                    if extra_allocation > 0 and (self.raw_cache_alloc + extra_allocation) < 100:
                        self.raw_cache_alloc = self.raw_cache_alloc + extra_allocation
                    elif extra_allocation > 0:
                        self.augmented_cache_alloc = self.augmented_cache_alloc + extra_allocation

                self.tune_allocations_storm(self.cache_alloc, [self.raw_cache_alloc,
                                                               self.decoded_cache_alloc,
                                                               self.augmented_cache_alloc])

        elif abs(aggregate_throughput_current - aggregate_throughput_previous) / aggregate_throughput_current * 100 > 7:
            self.direction_up = not (self.direction_up)
            print("flipping direction")
            print("current:", self.raw_cache_alloc, self.decoded_cache_alloc, self.augmented_cache_alloc)
            print("previous:", self.raw_cache_alloc_base, self.decoded_cache_alloc_base,
                  self.augmented_cache_alloc_base)
            self.raw_cache_alloc, self.decoded_cache_alloc, self.augmented_cache_alloc = \
                self.raw_cache_alloc_base, self.decoded_cache_alloc_base, self.augmented_cache_alloc_base
            self.tune_allocations_smooth(self.cache_alloc,
                                         [self.raw_cache_alloc, self.decoded_cache_alloc,
                                          self.augmented_cache_alloc])
        else:
            self.throughput_drop_count += 1
            if self.throughput_drop_count >= 10:
                self.throughput_drop_count = 0

                self.direction_up = not (self.direction_up)
                print("flipping direction")
                print("current:", self.raw_cache_alloc, self.decoded_cache_alloc, self.augmented_cache_alloc)
                print("previous:", self.raw_cache_alloc_base, self.decoded_cache_alloc_base,
                      self.augmented_cache_alloc_base)
                self.raw_cache_alloc, self.decoded_cache_alloc, self.augmented_cache_alloc = \
                    self.raw_cache_alloc_base, self.decoded_cache_alloc_base, self.augmented_cache_alloc_base
                self.tune_allocations_smooth(self.cache_alloc,
                                             [self.raw_cache_alloc, self.decoded_cache_alloc,
                                              self.augmented_cache_alloc])
            print("throughput dropped but within the 5% error margin")

    def autotune_v2(self, datarame_file, dsi_hash, pid, epoch):
        if epoch < 1:
            self.epoch = epoch + 1
            print("no autotuning until epoch >= 1")
            return
        print("Force apply previous tuning states")
        self.tune_allocations_storm(self.cache_alloc,
                                    [self.raw_cache_alloc, self.decoded_cache_alloc,
                                     self.augmented_cache_alloc])
        print("autotune_v2")
        df = pandas.read_csv(datarame_file)
        all_dataset_jobs = df.loc[df['dsi_hash'] == str(dsi_hash)]
        grouped_by_pid = all_dataset_jobs.groupby('pid', group_keys=True)
        aggregate_throughput_current = 0
        aggregate_throughput_previous = 0
        aggregate_throughput_previous_ma = 0
        for name, data in grouped_by_pid:
            data_1 = data.sort_values(by='profiled_timestamp', ascending=False).head(2)
            aggregate_throughput_current += data_1.iloc[0]['throughput']
            if len(data_1) > 1:
                throughput_trend_ma = data['throughput'].rolling(window=2).mean().iloc[-1]
                if not pandas.isna(throughput_trend_ma):
                    aggregate_throughput_previous += throughput_trend_ma  # data_1.iloc[1]['moving_avg']
                else:
                    aggregate_throughput_previous += data_1.iloc[1]['throughput']
                # aggregate_throughput_previous += data_1.iloc[1]['throughput']

        last_3_splits = all_dataset_jobs.tail(3)['cache_split']
        all_same = last_3_splits.nunique() == 1

        if aggregate_throughput_current > aggregate_throughput_previous:
            if self.direction_up == True:
                print("moving up")

        throughput_diff = aggregate_throughput_current - aggregate_throughput_previous
        print(aggregate_throughput_current, aggregate_throughput_previous)
        load_avg = psutil.getloadavg()[1]
        num_cores = os.cpu_count()
        cach_utils = []

        raw_cache_info = self.sample_cache.info()
        cach_utils.append((((raw_cache_info['maxmemory'] - raw_cache_info['used_memory']) / raw_cache_info[
            'maxmemory']) * 100, 'raw')) if raw_cache_info['maxmemory'] > 0 else -1

        dec_cache_info = self.decoded_cache.info()
        cach_utils.append((((dec_cache_info['maxmemory'] - dec_cache_info['used_memory']) / dec_cache_info[
            'maxmemory']) * 100, 'dec')) if dec_cache_info['maxmemory'] > 0 else -1

        aug_cache_info = self.tensor_cache.info()

        if aggregate_throughput_current > aggregate_throughput_previous:
            if self.direction_up == True:
                print("moving up")
                # decision to go to augmented or decoded based on cost benefit

    def autotune_simple(self, datarame_file, dsi_hash, pid, epoch):
        print("Force apply previous tuning states")
        self.tune_allocations_storm(self.cache_alloc,
                                    [self.raw_cache_alloc, self.decoded_cache_alloc,
                                     self.augmented_cache_alloc])
        print("autotune_simple")
        df = pandas.read_csv(datarame_file)
        # df['moving_avg'] = df['throughput'].rolling(window=3).mean()
        all_dataset_jobs = df.loc[df['dsi_hash'] == str(dsi_hash)]
        grouped_by_pid = all_dataset_jobs.groupby('pid', group_keys=True)
        aggregate_throughput_current = 0
        aggregate_throughput_previous = 0
        aggregate_throughput_previous_ma = 0

        for name, data in grouped_by_pid:
            data_1 = data.sort_values(by='profiled_timestamp', ascending=False).head(2)
            aggregate_throughput_current += data_1.iloc[0]['throughput']
            if len(data_1) > 1:
                throughput_trend_ma = data['throughput'].rolling(window=2).mean().iloc[-1]
                # print("-->",throughput_trend_ma)
                if not pandas.isna(throughput_trend_ma):
                    #    print('1')
                    aggregate_throughput_previous_ma += throughput_trend_ma  # data_1.iloc[1]['moving_avg']
                else:
                    #    print('2')
                    aggregate_throughput_previous_ma += data_1.iloc[1]['throughput']
                # if pandas.isna(aggregate_throughput_previous_ma):
                #    print("no moving avg")
                #    aggregate_throughput_previous_ma+= data_1.iloc[1]['throughput']
                aggregate_throughput_previous += data_1.iloc[1]['throughput']
        last_3_splits = all_dataset_jobs.tail(3)['cache_split']
        all_same = last_3_splits.nunique() == 1

        # throughput_diff = aggregate_throughput_current - aggregate_throughput_previous_ma
        change_perc = (
                    (aggregate_throughput_current - aggregate_throughput_previous) / aggregate_throughput_current * 100)
        print("change percentage: ", change_perc, "%")
        print("autotune dir (1=raw to aug; 2=aug to dec; 3=Stable):", self.autotune_simple_dir)
        # print(aggregate_throughput_current, aggregate_throughput_previous, aggregate_throughput_previous_ma)

        if change_perc > 0.5:
            if self.autotune_simple_dir == 1:
                step_size = min(40, 2 * ((
                                                     aggregate_throughput_current - aggregate_throughput_previous) / aggregate_throughput_current * 100))
                source = 'raw'
                destination = 'augmented'
                print("source: ", source, "destination ", destination)

                self.augmented_cache_alloc_base = self.augmented_cache_alloc
                self.decoded_cache_alloc_base = self.decoded_cache_alloc
                self.raw_cache_alloc_base = self.raw_cache_alloc

                raw_cache_alloc = self.raw_cache_alloc - step_size
                if raw_cache_alloc < 0:
                    step_size = step_size - (abs(raw_cache_alloc))
                self.raw_cache_alloc = self.raw_cache_alloc - step_size

                extra_allocation = 0
                augmented_cache_alloc = self.augmented_cache_alloc + step_size
                if augmented_cache_alloc > 100:
                    extra_allocation = augmented_cache_alloc - 100
                    step_size = step_size - (augmented_cache_alloc - 100)
                self.augmented_cache_alloc = self.augmented_cache_alloc + step_size
                self.raw_cache_alloc = self.raw_cache_alloc + extra_allocation
                self.tune_allocations_smooth(self.cache_alloc,
                                             [self.raw_cache_alloc, self.decoded_cache_alloc,
                                              self.augmented_cache_alloc])
            elif self.autotune_simple_dir == 2:
                step_size = min(40, 2 * ((
                                                     aggregate_throughput_current - aggregate_throughput_previous) / aggregate_throughput_current * 100))
                source = 'augmented'
                destination = 'decoded'
                print("source: ", source, "destination ", destination)

                self.augmented_cache_alloc_base = self.augmented_cache_alloc
                self.decoded_cache_alloc_base = self.decoded_cache_alloc
                self.raw_cache_alloc_base = self.raw_cache_alloc

                augmented_cache_alloc = self.augmented_cache_alloc - step_size
                if augmented_cache_alloc < 0:
                    step_size = step_size - (abs(augmented_cache_alloc))
                self.augmented_cache_alloc = self.augmented_cache_alloc - step_size

                extra_allocation = 0

                decoded_cache_alloc = self.decoded_cache_alloc + step_size
                if decoded_cache_alloc > 100:
                    extra_allocation = decoded_cache_alloc - 100
                    step_size = step_size - (decoded_cache_alloc - 100)
                self.decoded_cache_alloc = self.decoded_cache_alloc + step_size
                # if extra_allocation > 0 and (self.raw_cache_alloc + extra_allocation) < 100:
                #    self.raw_cache_alloc = self.raw_cache_alloc + extra_allocation
                # elif extra_allocation > 0:
                #    self.augmented_cache_alloc = self.augmented_cache_alloc + extra_allocation
                if extra_allocation > 0:
                    self.augmented_cache_alloc = self.augmented_cache_alloc + extra_allocation

                self.tune_allocations_storm(self.cache_alloc, [self.raw_cache_alloc,
                                                               self.decoded_cache_alloc,
                                                               self.augmented_cache_alloc])

            else:
                print('split stabilized, throughput should have not increased >4%')

        elif change_perc < -0.5 and self.autotune_simple_dir == 1:
            print("change movement dir becasue throughput reduced and we are in phase 1")
            self.autotune_simple_dir = 2
            self.raw_cache_alloc, self.decoded_cache_alloc, self.augmented_cache_alloc = \
                self.raw_cache_alloc_base, self.decoded_cache_alloc_base, self.augmented_cache_alloc_base
            self.tune_allocations_smooth(self.cache_alloc,
                                         [self.raw_cache_alloc, self.decoded_cache_alloc,
                                          self.augmented_cache_alloc])
        elif change_perc < -0.5 and self.autotune_simple_dir == 2:
            print("stop movement becasue throughput reduced and we are in phase 2. Autotuning done!")
            self.autotune_simple_dir = None
            self.raw_cache_alloc, self.decoded_cache_alloc, self.augmented_cache_alloc = \
                self.raw_cache_alloc_base, self.decoded_cache_alloc_base, self.augmented_cache_alloc_base
            self.tune_allocations_smooth(self.cache_alloc,
                                         [self.raw_cache_alloc, self.decoded_cache_alloc,
                                          self.augmented_cache_alloc])

    def autotune_raw_dec_no_profile(self, datarame_file, dsi_hash, pid):
        print("autotune_no_profile")
        df = pandas.read_csv(datarame_file)
        all_dataset_jobs = df.loc[df['dsi_hash'] == str(dsi_hash)]
        grouped_by_pid = all_dataset_jobs.groupby('pid', group_keys=True)
        aggregate_throughput_current = 0
        aggregate_throughput_previous = 0
        for name, data in grouped_by_pid:
            data_1 = data.sort_values(by='profiled_timestamp', ascending=False).head(2)
            aggregate_throughput_current += data_1.iloc[0]['throughput']
            if len(data_1) > 1:
                aggregate_throughput_previous += data_1.iloc[1]['throughput']

        throughput_diff = aggregate_throughput_current - aggregate_throughput_previous
        print(aggregate_throughput_current, aggregate_throughput_previous)

        if aggregate_throughput_previous == 0:
            proportional_change = 1
        else:
            proportional_change = abs(throughput_diff) / aggregate_throughput_previous

        if throughput_diff < 0:
            if proportional_change > 0.05:
                step = 15  # min(20, (10 * proportional_change))
                self.raw_cache_alloc += step
                self.decoded_cache_alloc -= step
                # self.augmented_cache_alloc -= step

                self.tune_allocations_smooth(self.cache_alloc,
                                             [self.raw_cache_alloc, self.decoded_cache_alloc,
                                              self.augmented_cache_alloc])

                print("go back to prev split")
        else:
            if proportional_change > 0.05:
                step = 15  # min(20, (10 * proportional_change))
                self.raw_cache_alloc -= step
                self.decoded_cache_alloc += step
                # self.augmented_cache_alloc += step
                self.tune_allocations_smooth(self.cache_alloc,
                                             [self.raw_cache_alloc, self.decoded_cache_alloc,
                                              self.augmented_cache_alloc])
                print("go to next split")

        """
        if throughput_diff < 0:
            if (abs(throughput_diff)/aggregate_throughput_previous)>0.05:
                if aggregate_throughput_previous == 0:
                    step=10
                else:
                    step = 10#(abs(throughput_diff)/aggregate_throughput_previous)*100
                self.raw_cache_alloc +=step
                self.decoded_cache_alloc -= step
                self.tune_allocations_storm(self.cache_alloc,
                                            [self.raw_cache_alloc, self.decoded_cache_alloc, self.augmented_cache_alloc])
                print("go back to prev split")

        else:
            if (abs(throughput_diff) / aggregate_throughput_previous) > 0.05:
                if aggregate_throughput_previous == 0:
                    step = 10
                else:
                    step = 10#(abs(throughput_diff) / aggregate_throughput_previous) * 100
                self.raw_cache_alloc -=step
                self.decoded_cache_alloc += step
                self.tune_allocations_storm(self.cache_alloc,
                                            [self.raw_cache_alloc, self.decoded_cache_alloc, self.augmented_cache_alloc])
                print("go to next split")
        """

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolderODS(BRDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            random_transforms: Optional[Callable] = None,
            static_transforms: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            job_sample_tracker_port: Optional[int] = None,
            cache_port: Optional[int] = 6378,
            tensor_port: Optional[int] = 6380,
            decoded_port: Optional[int] = 6379,
            raw_cache_host: Optional[str] = 'localhost',
            tensor_cache_host: Optional[str] = 'localhost',
            decoded_cache_host: Optional[str] = 'localhost',
            xtreme_speed: Optional[bool] = True,
            initial_cache_size: Optional[int] = 100,
            initial_cache_split: Optional[list] = [33, 33, 33]
    ):
        super(ImageFolderODS, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          sample_port = cache_port,
                                          tensor_port = tensor_port,
                                          #####
                                          raw_cache_host = raw_cache_host,
                                          tensor_cache_host=tensor_cache_host,
                                          random_transforms = random_transforms,
                                          static_transforms = static_transforms,
                                          decoded_port = decoded_port,
                                          decoded_cache_host = decoded_cache_host,
                                          xtreme_speed = xtreme_speed,
                                          initial_cache_size=initial_cache_size,
                                          initial_cache_split=initial_cache_split
                                          )
        self.imgs = self.samples
        self.job_sample_tracker_port = job_sample_tracker_port
        self.samples_seen = redis.Redis(port=self.job_sample_tracker_port)
        self.sample_port = cache_port
        self.tensor_port = tensor_port


    def set_seen_samples(self, index):
        #print("set_seen_samples - ", index)
        for i in index:
            self.samples_seen.set(i, 1)



    def get_seen_samples(self):
        return self.samples_seen

class ImageFolderMinIO(MinioDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            random_transforms: Optional[Callable] = None,
            static_transforms: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            job_sample_tracker_port: Optional[int] = None,
            cache_port: Optional[int] = 6378,
            tensor_port: Optional[int] = 6380,
            decoded_port: Optional[int] = 6379,
            raw_cache_host: Optional[str] = 'localhost',
            tensor_cache_host: Optional[str] = 'localhost',
            decoded_cache_host: Optional[str] = 'localhost',
            xtreme_speed: Optional[bool] = True
    ):
        super(ImageFolderMinIO, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          sample_port = cache_port,
                                          tensor_port = tensor_port,
                                          #####
                                          raw_cache_host = raw_cache_host,
                                          tensor_cache_host=tensor_cache_host,
                                          #random_transforms = random_transforms,
                                          #static_transforms = static_transforms,
                                          #decoded_port = decoded_port,
                                          #decoded_cache_host = decoded_cache_host,
                                          #xtreme_speed = xtreme_speed
                                          )
        self.imgs = self.samples
        self.job_sample_tracker_port = job_sample_tracker_port
        self.samples_seen = redis.Redis(port=self.job_sample_tracker_port)
        self.sample_port = cache_port
        self.tensor_port = tensor_port


    def set_seen_samples(self, index):
        #print("set_seen_samples - ", index)
        for i in index:
            self.samples_seen.set(i, 1)



    def get_seen_samples(self):
        return self.samples_seen

class ImageFolderMDP(CachedDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            random_transforms: Optional[Callable] = None,
            static_transforms: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            job_sample_tracker_port: Optional[int] = None,
            cache_port: Optional[int] = 6378,
            tensor_port: Optional[int] = 6380,
            decoded_port: Optional[int] = 6379,
            raw_cache_host: Optional[str] = 'localhost',
            tensor_cache_host: Optional[str] = 'localhost',
            decoded_cache_host: Optional[str] = 'localhost',
            xtreme_speed: Optional[bool] = True,
            initial_cache_size: Optional[int] = 100,
            initial_cache_split: Optional[list] = [33, 33, 33]
    ):
        super(ImageFolderMDP, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          sample_port = cache_port,
                                          tensor_port = tensor_port,
                                          #####
                                          raw_cache_host = raw_cache_host,
                                          tensor_cache_host=tensor_cache_host,
                                          random_transforms = random_transforms,
                                          static_transforms = static_transforms,
                                          decoded_port = decoded_port,
                                          decoded_cache_host = decoded_cache_host,
                                          xtreme_speed = xtreme_speed,
                                          initial_cache_size=initial_cache_size,
                                          initial_cache_split=initial_cache_split
                                          )
        self.imgs = self.samples
        self.job_sample_tracker_port = job_sample_tracker_port
        self.samples_seen = redis.Redis(port=self.job_sample_tracker_port)
        self.sample_port = cache_port
        self.tensor_port = tensor_port


    def set_seen_samples(self, index):
        #print("set_seen_samples - ", index)
        for i in index:
            self.samples_seen.set(i, 1)



    def get_seen_samples(self):
        return self.samples_seen

class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            random_transforms: Optional[Callable] = None,
            static_transforms: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            job_sample_tracker_port: Optional[int] = None,
            cache_port: Optional[int] = 6378,
            tensor_port: Optional[int] = 6380,
            decoded_port: Optional[int] = 6379,
            raw_cache_host: Optional[str] = 'localhost',
            tensor_cache_host: Optional[str] = 'localhost',
            decoded_cache_host: Optional[str] = 'localhost',
            xtreme_speed: Optional[bool] = True
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          sample_port = cache_port,
                                          tensor_port = tensor_port,
                                          #####
                                          #raw_cache_host = raw_cache_host,
                                          #tensor_cache_host=tensor_cache_host,
                                          #random_transforms = random_transforms,
                                          #static_transforms = static_transforms,
                                          #decoded_port = decoded_port,
                                          #decoded_cache_host = decoded_cache_host,
                                          #xtreme_speed = xtreme_speed
                                          )
        self.imgs = self.samples
        self.job_sample_tracker_port = job_sample_tracker_port
        self.samples_seen = redis.Redis(port=self.job_sample_tracker_port)
        self.sample_port = cache_port
        self.tensor_port = tensor_port


    def set_seen_samples(self, index):
        #print("set_seen_samples - ", index)
        for i in index:
            self.samples_seen.set(i, 1)



    def get_seen_samples(self):
        return self.samples_seen

class ImageFolderQuiver(QuiverDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            random_transforms: Optional[Callable] = None,
            static_transforms: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            job_sample_tracker_port: Optional[int] = None,
            cache_port: Optional[int] = 6378,
            tensor_port: Optional[int] = 6380,
            decoded_port: Optional[int] = 6379,
            raw_cache_host: Optional[str] = 'localhost',
            tensor_cache_host: Optional[str] = 'localhost',
            decoded_cache_host: Optional[str] = 'localhost',
            xtreme_speed: Optional[bool] = True
    ):
        super(ImageFolderQuiver, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          sample_port = cache_port,
                                          tensor_port = tensor_port,
                                          #####
                                          #raw_cache_host = raw_cache_host,
                                          #tensor_cache_host=tensor_cache_host,
                                          #random_transforms = random_transforms,
                                          #tatic_transforms = static_transforms,
                                          #decoded_port = decoded_port,
                                          #decoded_cache_host = decoded_cache_host,
                                          #xtreme_speed = xtreme_speed
                                          )
        self.imgs = self.samples
        self.job_sample_tracker_port = job_sample_tracker_port
        self.samples_seen = redis.Redis(port=self.job_sample_tracker_port)
        self.sample_port = cache_port
        self.tensor_port = tensor_port


    def set_seen_samples(self, index):
        #print("set_seen_samples - ", index)
        for i in index:
            self.samples_seen.set(i, 1)



    def get_seen_samples(self):
        return self.samples_seen

class ImageFolderBBModel(BBModelDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            random_transforms: Optional[Callable] = None,
            static_transforms: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            job_sample_tracker_port: Optional[int] = None,
            cache_port: Optional[int] = 6378,
            tensor_port: Optional[int] = 6380,
            decoded_port: Optional[int] = 6379,
            raw_cache_host: Optional[str] = 'localhost',
            tensor_cache_host: Optional[str] = 'localhost',
            decoded_cache_host: Optional[str] = 'localhost',
            xtreme_speed: Optional[bool] = True,
            fetch_time: Optional[list] = None,
            initial_cache_size: Optional[int] = 100,
            initial_cache_split: Optional[list] = [33,33,33]
    ):
        super(ImageFolderBBModel, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          sample_port = cache_port,
                                          tensor_port = tensor_port,
                                          #####
                                          raw_cache_host = raw_cache_host,
                                          tensor_cache_host=tensor_cache_host,
                                          random_transforms = random_transforms,
                                          static_transforms = static_transforms,
                                          decoded_port = decoded_port,
                                          decoded_cache_host = decoded_cache_host,
                                          xtreme_speed = xtreme_speed,
                                          fetch_time = fetch_time,
                                          initial_cache_size = initial_cache_size,
                                          initial_cache_split = initial_cache_split
                                          )
        self.imgs = self.samples
        self.job_sample_tracker_port = job_sample_tracker_port
        self.samples_seen = redis.Redis(port=self.job_sample_tracker_port)
        self.sample_port = cache_port
        self.tensor_port = tensor_port


    def set_seen_samples(self, index):
        #print("set_seen_samples - ", index)
        for i in index:
            self.samples_seen.set(i, 1)



    def get_seen_samples(self):
        return self.samples_seen
