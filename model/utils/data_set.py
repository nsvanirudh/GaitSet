import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import cv2
import xarray as xr
import pdb
import matplotlib.pyplot as plt
import csv

class DataSet(tordata.Dataset):
    def __init__(self, seq_dir, label, seq_type, view, cache, resolution, normalize='max', num_frames_file='/scratch0/snanduri/GaitSet/num_frames.txt'):
        self.seq_dir = seq_dir
        self.view = view
        self.seq_type = seq_type
        self.label = label
        self.cache = cache
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution)/64*10)
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size
        self.normalize = normalize

        self.label_set = set(self.label)
        self.seq_type_set = set(self.seq_type)
        self.view_set = set(self.view)
        _ = np.zeros((len(self.label_set),
                      len(self.seq_type_set),
                      len(self.view_set))).astype('int')
        _ -= 1
        self.index_dict = xr.DataArray(
            _,
            coords={'label': sorted(list(self.label_set)),
                    'seq_type': sorted(list(self.seq_type_set)),
                    'view': sorted(list(self.view_set))},
            dims=['label', 'seq_type', 'view'])

        for i in range(self.data_size):
            _label = self.label[i]
            _seq_type = self.seq_type[i]
            _view = self.view[i]
            self.index_dict.loc[_label, _seq_type, _view] = i
            
        with open(num_frames_file, mode='r') as infile:
            reader = csv.reader(infile, delimiter=' ')
            with open('temp.csv', mode='w') as outfile:
                writer = csv.writer(outfile)
                self.mydict = {rows[0]:rows[1] for rows in reader}

    def load_all_data(self):
        for i in range(self.data_size):
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __loader__(self, path):
        return self.img2xarray(
            path)[:, :, self.cut_padding:-self.cut_padding].astype(
            'float32') / 255.0
    def __loader_potion__(self, path):
        return self.img2xarray_potion(
            path)[:, :, self.cut_padding:-self.cut_padding].astype(
            'float32') / 255.0
            
    def __loader_npy__(self, path):
        return self.img2xarray_npy(
            path).astype(
            'float32')

    def __getitem__(self, index):
        # pose sequence sampling
        if not self.cache:
            # data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            data = [self.__loader_potion__(_path) for _path in self.seq_dir[index]]
            # data = [self.__loader_npy__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
        elif self.data[index] is None:
            # data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            data = [self.__loader_potion__(_path) for _path in self.seq_dir[index]]
            # data = [self.__loader_npy__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            self.data[index] = data
            self.frame_set[index] = frame_set
        else:
            data = self.data[index]
            frame_set = self.frame_set[index]

        return data, frame_set, self.view[
            index], self.seq_type[index], self.label[index],

    def img2xarray(self, flie_path):
        imgs = sorted(list(os.listdir(flie_path)))
        frame_list = [np.reshape(
            cv2.imread(osp.join(flie_path, _img_path)),
            [self.resolution, self.resolution, -1])[:, :, 0]
                      for _img_path in imgs
                      if osp.isfile(osp.join(flie_path, _img_path))]
        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict
        
    def img2xarray_potion(self, flie_path):
        imgs = sorted(list(os.listdir(flie_path)))
        imgs = [img for img in imgs if ('area' in img and 'complete' not in img)]
        frame_list = [np.reshape(
            cv2.imread(osp.join(flie_path, _img_path)),
            [-1, self.resolution, self.resolution])[:, :, :]
                      for _img_path in imgs
                      if osp.isfile(osp.join(flie_path, _img_path))]
        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x', 'channels'],
        )
        return data_dict
        
    def img2xarray_npy(self, flie_path):
        imgs = sorted(list(os.listdir(flie_path)))
        #print(np.shape(np.load(osp.join(flie_path, imgs[0]))))
        # img = np.load(osp.join(flie_path, imgs[0]))
        # img2 = np.load(osp.join(flie_path, imgs[-1]))
        # img_normalized = self.normalize_npy(img)
        # img_normalized2 = self.normalize_npy(img2)
        # img1 = np.reshape(img_normalized[:,10,:,:], [64, -1, 3])[:,10:54,:]
        # img2 = np.reshape(img_normalized2[:,5,:,:], [64, -1, 3])[:,10:54,:]
        # print(img1.shape)
        # plt.imsave('/scratch0/snanduri/temp.jpg', img1/img1.max())
        # plt.imsave('/scratch0/snanduri/temp2.jpg', img2/img2.max())
        
        # frame_list = [np.reshape(
            # np.load(osp.join(flie_path, _img_path)),
            # [3, -1, self.resolution, self.resolution])
                      # for _img_path in imgs
                      # if osp.isfile(osp.join(flie_path, _img_path))]
        if self.normalize == 'max':
            frame_list = self.normalize_npy(np.load(osp.join(flie_path, imgs[0])).astype(
                'float32')).tolist()
        elif self.normalize == 'area':
            frame_list = self.normalize_npy_area(np.load(osp.join(flie_path, imgs[0])).astype(
                'float32'), imgs[0].split('/')[-1]).tolist()
        # pdb.set_trace()
        frame_list = [np.array(a) for a in frame_list]
        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'joints', 'img_y', 'img_x'],
        )
        return data_dict
        
    def normalize_npy(self, arr):
        #assuming the channel dimension is dim 0 and arr is 4d
        channel_joint_max = np.amax(arr,(-1,-2))[:,:,np.newaxis,np.newaxis]
        channel_joint_max[channel_joint_max == 0] = 1.0
        #print(potion_representation.shape,potion_representation.max(),potion_representation.min(),channel_joint_max)
        return np.reshape(arr/channel_joint_max, [3, 19, 64, -1])
        # return np.reshape(arr/(arr.max(2).max(2)[:,:,None,None] + 1e-8), [3, 19, 64, -1])
        
    def normalize_npy_area(self, arr, file_name, channels=3):
        #assuming the channel dimension is dim 0 and arr is 4d
        frames = int(self.mydict[file_name])
        from_utils = self.get_channel_weights(frames, channels)
        summed = np.sum(from_utils,1)
        for ch in range(channels):
            arr[ch]/=(summed[ch]*255.0)
            # arr[ch]/=(summed[ch])
        return np.reshape(arr, [3, 19, 64, -1])
        
    def get_channel_weights(self, frames, channels=3):
        channel_weights = np.zeros((channels,frames))
        correction = 0
        T_by_C_min_1_b = frames/(channels-1)
        T_by_C_min_1 = np.floor(T_by_C_min_1_b).astype(np.int)
        start_points = np.linspace(0,frames,channels).astype(np.int)
        for window in range(channels - 1):
            len_window = start_points[window+1] - start_points[window]
            correction = 0 if window == channels-2 or channels == 2 else 1
            channel_weights[window,start_points[window] : start_points[window+1]+correction] = np.linspace(1,0,len_window+correction)
            channel_weights[window+1,start_points[window] : start_points[window+1]+correction] = np.linspace(0,1,len_window+correction)
        return channel_weights
        
    def __len__(self):
        return len(self.label)
