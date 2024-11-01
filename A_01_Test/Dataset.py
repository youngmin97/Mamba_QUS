import os
import numpy as np
import torch.utils.data as data
import torch
import h5py
import torchvision.transforms.functional as TF
import glob
import random
import cv2
import scipy.ndimage

class create_dataset(data.Dataset):
	def __init__(self, data_dir, data_name, max_scanline=24):
		super(create_dataset, self).__init__()
		self.data_dir = os.path.join(data_dir, data_name)
		self.max_scanline = max_scanline

		self.RF_path = sorted(glob.glob(r'{}/RF_*.png'.format(self.data_dir)))
		self.AT_path = sorted(glob.glob(r'{}/ATT_*.png'.format(self.data_dir)))
		self.SOS_path = sorted(glob.glob(r'{}/SOS_*.png'.format(self.data_dir)))
		self.ESD_path = sorted(glob.glob(r'{}/ESD_*.png'.format(self.data_dir)))
		self.ESC_path = sorted(glob.glob(r'{}/ESC2_*.png'.format(self.data_dir)))

		# print('number of data : {}'.format(len(self.RF_path)))

	def __getitem__(self, index):
		RF_temp = cv2.imread(self.RF_path[index], 0)
		if RF_temp is None:
			print(f"Warning: Failed to read image {self.RF_path[index]}")
			return None  # 또는 continue 등을 사용하여 처리할 수 있습니다.

		RF_temp = RF_temp.astype(np.float32)     # [2048, 2700]
		# print(self.RF_path[index])
		AT = cv2.imread(self.AT_path[index], 0)      # [128, 128], [H, W]
		SOS = cv2.imread(self.SOS_path[index], 0)      # [128, 128], [H, W]
		ESD = cv2.imread(self.ESD_path[index], 0)      # [128, 128], [H, W]
		ESC = cv2.imread(self.ESC_path[index], 0)      # [128, 128], [H, W]

		RF_temp2 = (RF_temp[:,:-2] -127)/255                # -0.5 ~ 0.5
		scale_channel = np.expand_dims(RF_temp[:, -2], -1) /255     # 0 ~ 1
		scale_scanline = np.expand_dims(RF_temp[:, -1], -1) / 255   # 0 ~ 1
		RF = RF_temp2 * scale_channel * scale_scanline + 0.5        # 0 ~ 1

		inputs = np.flip(np.transpose(np.reshape(RF, (self.max_scanline, 48, 2700)), (1, 2, 0)), axis=2)  # [48, 2700, max_scanline]
		attn = AT.transpose().astype(np.float32) /255       # [128, 128], [W, H]
		sos = SOS.transpose().astype(np.float32) /255       # [128, 128], [W, H]
		esd = ESD.transpose().astype(np.float32) /255       # [128, 128], [W, H]
		esc = ESC.transpose().astype(np.float32) /255       # [128, 128], [W, H]

		inputs = inputs[:,:-60,:]
		data = {
			'input': TF.to_tensor(inputs.copy()).contiguous(),
			'attn': TF.to_tensor(attn).contiguous(),
			'sos': TF.to_tensor(sos).contiguous(),
			'esd': TF.to_tensor(esd).contiguous(),
			'esc': TF.to_tensor(esc).contiguous(),
		}

		return data

	def __len__(self):
		return len(self.RF_path)


class create_dataset_alphinion(data.Dataset):
	def __init__(self, data_dir, max_scanline=48):
		super(create_dataset_alphinion, self).__init__()
		self.data_dir, self.max_scanline = data_dir, max_scanline

		self.RF_path = sorted(glob.glob(r'{}/RF_test_*.png'.format(self.data_dir)))
		self.bmode_orgn_path = sorted(glob.glob(r'{}/Bmode_orgn_test_*.png'.format(self.data_dir)))
		self.bmode_resize_path = sorted(glob.glob(r'{}/Bmode_resize_test_*.png'.format(self.data_dir)))
		# self.BMODE_path = sorted(glob.glob(r'{}/BMODE_test_*.png'.format(self.data_dir)))

		print('number of data : {}'.format(len(self.RF_path)))

	def __getitem__(self, index):
		RF_path = self.RF_path[index]
		# BMODE_path = self.BMODE_path[index]
		bmode_orgn_path = self.bmode_orgn_path[index]
		bmode_resize_path = self.bmode_resize_path[index]

		RF_temp = cv2.imread(RF_path, 0).astype(np.float32)  # [2048, 2700]
		# BMODE = cv2.imread(BMODE_path, 0)
		bmode_orgn = cv2.imread(bmode_orgn_path, 0)      # [128, 128], [H, W]
		bmode_resize = cv2.imread(bmode_resize_path, 0)      # [128, 128], [H, W]

		RF_temp2 = (RF_temp[:,:-2] -127)/255                # -0.5 ~ 0.5
		scale_channel = np.expand_dims(RF_temp[:, -2], -1) /255     # 0 ~ 1
		scale_scanline = np.expand_dims(RF_temp[:, -1], -1) / 255   # 0 ~ 1
		RF = RF_temp2 * scale_channel * scale_scanline + 0.5        # 0 ~ 1

		inputs = np.transpose(np.reshape(RF, (self.max_scanline, 48, 2700)), (1, 2, 0))  # [48, 2700, max_scanline]
		# BMODE = BMODE.transpose().astype(np.float32) / 255
		# BMODE_MED = scipy.ndimage.median_filter(BMODE, size=5)
		bmode_orgn = bmode_orgn.transpose().astype(np.float32) /255       # [128, 374], [W, H]
		bmode_resize = bmode_resize.transpose().astype(np.float32) /255       # [128, 374], [W, H]

		inputs = inputs[:, :-60, :]
		# data = {'input': TF.to_tensor(inputs), 'BMODE_MED':TF.to_tensor(BMODE_MED), 'BMODE': TF.to_tensor(BMODE), 'bmode_orgn': TF.to_tensor(bmode_orgn), 'bmode_resize': TF.to_tensor(bmode_resize)}
		data = {'input': TF.to_tensor(inputs), 'bmode_orgn': TF.to_tensor(bmode_orgn), 'bmode_resize': TF.to_tensor(bmode_resize)}

		return data

	def __len__(self):
		return len(self.RF_path)



if __name__ == '__main__':
	x = create_dataset(data_dir = r'E:\k-wave\Data_v5_ConventionalMode\240313_Version3\Data_Refined', data_name = r'Data_v5_ConventionalMode_240923\test')
	y = x.__getitem__(0)
	z = y['input']
	print(z)
	print(z.shape)