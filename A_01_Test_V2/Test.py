################################
# Data V1
# ATT: PSNR 21.083, NMAE 0.076, SSIM 0.876, TIME 0.057.
# SOS: PSNR 16.716, NMAE 0.140, SSIM 0.814, TIME 0.057.
#
# Data V2
# ATT: PSNR 27.978, NMAE 0.018, SSIM 0.949, TIME 0.065.
# SOS: PSNR 27.838, NMAE 0.019, SSIM 0.941, TIME 0.065.
################################

import torch
import argparse
import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import os
import time
from datetime import datetime
import neptune
import math
import imageio
import cv2
import tqdm
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import Dataset, Model


def main():
	parser = argparse.ArgumentParser()
	args = parser.parse_args("")

	args.data_v3_dir = r'E:/k-wave/Data_v5_ConventionalMode/240313_Version3/Data_Refined'
	args.data_v3_name = r'Data_v5_ConventionalMode_240923'
	args.network_name = r'CardNet_max.pt'
	args.savefilename = os.getcwd()

	args.numberOfWorkers = 4
	args.test_batch_size = 1
	args.epoch = 1
	args.lr = 1e-4
	args.save = True
	args.device = 0

	print('Loading Model')
	CardNet = Model.CardNet().cuda(args.device)
	state = torch.load(r'./results/{}'.format(args.network_name))
	# module. 제거
	state_dict = OrderedDict()
	for k, v in state['state_dict'].items():
		name = k[7:] if k.startswith("module.") else k  # 'module.' 접두어 제거
		state_dict[name] = v

	CardNet.load_state_dict(state_dict)
	CardNet_optim = torch.optim.Adam(CardNet.parameters(), lr=args.lr)
	CardNet_optim.load_state_dict(state['optmizer'])

	print('Loading Dataset')
	torch.manual_seed(0)
	testset_v3 = Dataset.create_dataset(args.data_v3_dir, args.data_v3_name+'/test')
	testloader_v3 = DataLoader(dataset=testset_v3, batch_size=args.test_batch_size, shuffle=False,
						   num_workers=args.numberOfWorkers)

	from skimage.metrics import structural_similarity as ssim
	variables = ['ATT', 'SOS', 'ESD', 'ESC', '0BMODE']
	for epoch in range(args.epoch):
		# Test Set
		test_loss = 0.0
		test_psnr = [0.0, 0.0, 0.0, 0.0]
		test_nmae = [0.0, 0.0, 0.0, 0.0]
		test_ssim = [0.0, 0.0, 0.0, 0.0]
		test_time = 0.0
		CardNet.eval()
		with torch.no_grad():
			for i, data in enumerate(tqdm.tqdm(testloader_v3, leave=True)):
				input = data['input'].type(torch.cuda.FloatTensor).cuda(args.device)  # Tensor size [4, 19, 64, 5760]
				target_AT = data['attn'].type(torch.cuda.FloatTensor).cuda(args.device) # Tensor size [4, 1, 128, 128]
				target_SOS = data['sos'].type(torch.cuda.FloatTensor).cuda(args.device) # Tensor size [4, 1, 128, 128]
				target_ESD  = data['esd'].type(torch.cuda.FloatTensor).cuda(args.device) # Tensor size [4, 1, 128, 128]
				target_ESC  = data['esc'].type(torch.cuda.FloatTensor).cuda(args.device) # Tensor size [4, 1, 128, 128]
				target = torch.cat((target_AT, target_SOS), 1)

				x_curr = input
				image_curr = target

				ts = time.time()
				AT_pred, I16, I32, I64 = CardNet(x_curr)
				te = time.time()

				loss =nn.MSELoss()(AT_pred, image_curr) + nn.MSELoss()(I16, image_curr[:,:,::8,::8])\
				         + nn.MSELoss()(I32, image_curr[:,:,::4,::4]) + nn.MSELoss()(I64, image_curr[:,:,::2,::2])

				psnr = []
				nmae = []
				ssim1 = []
				for idx in range(2):
					psnr.append(20 * (1 / (((AT_pred[:, idx, :, :] - image_curr[:, idx, :, :]) ** 2).mean(dim=2).mean(
						dim=1).sqrt())).log10().mean().item())
					nmae.append((AT_pred[:,idx,:,:]-image_curr[:,idx,:,:]).abs().mean().item())
					ssim1.append(ssim(AT_pred[0][idx].data.cpu().numpy().transpose(), image_curr[0][idx].data.cpu().numpy().transpose()))

				test_loss += loss.item()
				test_psnr = [sum(x) for x in zip(test_psnr, psnr)]
				test_nmae = [sum(x) for x in zip(test_nmae, nmae)]
				test_ssim = [sum(x) for x in zip(test_ssim, ssim1)]
				test_time += te-ts

				# print('Test {}/{}, ATT {:2.3f}dB, SOS {:2.3f}dB, ESD {:2.3f}dB, ESC {:2.3f}dB. Took {:2.2f} sec'.format(i+1, len(testloader_v1), psnr[0], psnr[1], psnr[2], psnr[3], te-ts))

				if args.save:
					colormap = [cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_PINK, cv2.COLORMAP_BONE]

					for idx in range(2):
						gt_image = ((image_curr)[0][idx].data.cpu().numpy().transpose() * 255).astype(np.uint8)
						pred_image = ((AT_pred)[0][idx].data.cpu().numpy().transpose() * 255).astype(np.uint8)

						gt_image = cv2.cvtColor(cv2.applyColorMap(gt_image, colormap[idx]), cv2.COLOR_BGR2RGB)
						pred_image = cv2.cvtColor(cv2.applyColorMap(pred_image, colormap[idx]), cv2.COLOR_BGR2RGB)

						imageio.imwrite(r'./results/test_v3/Epoch{}_{}_target.png'.format(i + 1, variables[idx]), gt_image)
						imageio.imwrite(r'./results/test_v3/Epoch{}_{}_predicted.png'.format(i + 1, variables[idx]), pred_image)

		test_loss = test_loss / len(testloader_v3)
		test_psnr = [x/len(testloader_v3) for x in test_psnr]
		test_nmae = [x/len(testloader_v3) for x in test_nmae]
		test_ssim = [x/len(testloader_v3) for x in test_ssim]
		test_time = test_time / len(testloader_v3)

		for icd in range(2):
			print('{}: PSNR {:2.3f}, NMAE {:2.3f}, SSIM {:2.3f}, TIME {:2.3f}.'.format(variables[icd], test_psnr[icd], test_nmae[icd], test_ssim[icd], test_time))


if __name__ == '__main__':
	main()