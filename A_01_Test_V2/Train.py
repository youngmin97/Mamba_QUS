################################
# CardNet01
# U-Net, concatenate by axis1
# Attention within ROI
# Input: Only RF signal
# Output: Attention within ROI
# Loss: MSE Loss
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
import random
import tqdm
from torch.cuda.amp import autocast, GradScaler
import Dataset, Model

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main():
	parser = argparse.ArgumentParser()
	args = parser.parse_args("")

	args.data_v3_dir = r'/mnt/NAS_US1/KYM/Data_v5_ConventionalMode/240313_Version3/Data_Refined'
	args.data_v3_name = r'Data_v5_ConventionalMode_240923'
	args.savefilename = os.getcwd()

	args.numberOfWorkers = 4
	args.train_batch_size = 48
	args.validation_batch_size = 1
	args.epoch = 1000
	args.lr = 1e-4
	args.save = True
	args.neptune = True
	args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

	if args.neptune:
		print('Connecting to Neptune Network')
		args_dict = vars(args)
		run = neptune.init_run(project='youngmin97/Conventional-Mode',
							   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYzRhODA2ODgtNGU0Zi00NzI2LWJkNDItYWYzMTI2OGZjYzRhIn0=',
							   name=args.savefilename, source_files=['Train.py', 'Dataset.py', 'Model.py'])
		run["parameters"]=args_dict

	print('Loading Model')
	CardNet = Model.CardNet().cuda(args.device)
	# DataParallel 적용
	if torch.cuda.device_count() > 1:
		print(f'Using {torch.cuda.device_count()} GPUs!')
		CardNet = torch.nn.DataParallel(CardNet)
	CardNet_optim = torch.optim.Adam(CardNet.parameters(), lr=args.lr)
	scaler = GradScaler()

	print('Loading Dataset')
	torch.manual_seed(0)
	trainset_v3 = Dataset.create_dataset(args.data_v3_dir, args.data_v3_name+'/train')
	valset = Dataset.create_dataset(args.data_v3_dir, args.data_v3_name+'/val')
	trainloader_v3 = DataLoader(dataset=trainset_v3, batch_size=args.train_batch_size, shuffle=True,
								num_workers=args.numberOfWorkers)
	valloader = DataLoader(dataset=valset, batch_size=args.validation_batch_size, shuffle=False,
						   num_workers=args.numberOfWorkers)

	variables = ['ATT', 'SOS', 'ESD', 'ESC', '0BMODE']
	max_psnr = 0
	for epoch in range(args.epoch):

		if args.neptune:
			run["logs/Time Logger"].append('[{}] Epoch {}/{}'
							 .format(datetime.today().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,  args.epoch))
		# Train Set
		train_loss = 0.
		train_psnr = [0., 0., 0., 0.]
		CardNet.train()
		for i, data in enumerate(tqdm.tqdm(trainloader_v3, desc='EPOCH {}'.format(epoch+1), ncols=72, leave=True)):
			input = data['input'].type(torch.cuda.FloatTensor).cuda(args.device)  # Tensor size [4, 64, 32, 3018]
			target_AT = data['attn'].type(torch.cuda.FloatTensor).cuda(args.device) # Tensor size [4, 1, 128, 128]
			target_SOS = data['sos'].type(torch.cuda.FloatTensor).cuda(args.device) # Tensor size [4, 1, 128, 128]
			target_ESD  = data['esd'].type(torch.cuda.FloatTensor).cuda(args.device) # Tensor size [4, 1, 128, 128]
			target_ESC  = data['esc'].type(torch.cuda.FloatTensor).cuda(args.device) # Tensor size [4, 1, 128, 128]
			target = torch.cat((target_AT, target_SOS), 1).contiguous()

			aug_scanline = random.randint(8, 24)
			start_scanline = random.randint(0, 24-aug_scanline)
			end_scanline = start_scanline + aug_scanline
			x_curr = input[:, start_scanline:end_scanline, :, :]
			image_curr = target[:, :, start_scanline*8:end_scanline*8, :]

			with autocast():
				AT_pred, I16, I32, I64 = CardNet(x_curr)
				loss =nn.MSELoss()(AT_pred, image_curr) + nn.MSELoss()(I16, image_curr[:,:,::8,::8])\
						 + nn.MSELoss()(I32, image_curr[:,:,::4,::4]) + nn.MSELoss()(I64, image_curr[:,:,::2,::2])
			psnr = []
			for idx in range(2):
				psnr.append(20 * (1 / (((AT_pred[:,idx,:,:] - image_curr[:,idx,:,:]) ** 2).mean(dim=2).mean(dim=1).sqrt())).log10().mean().item())

			CardNet_optim.zero_grad()
			scaler.scale(loss).backward()
			scaler.step(CardNet_optim)
			scaler.update()

			train_loss += loss.item()
			train_psnr = [sum(x) for x in zip(train_psnr, psnr)]
			if args.neptune:
				run["logs/Batch Loss"].append(loss.item())
				run["logs/Batch PSNR ATT"].append(psnr[0])
				run["logs/Batch PSNR SOS"].append(psnr[1])

		# Validation Set
		val_loss = 0.0
		val_psnr = [0., 0., 0., 0.]
		CardNet.eval()
		with torch.no_grad():
			for i, data in enumerate(valloader):
				input = data['input'].type(torch.cuda.FloatTensor).cuda(args.device)  # Tensor size [4, 64, 32, 3018]
				target_AT = data['attn'].type(torch.cuda.FloatTensor).cuda(args.device) # Tensor size [4, 1, 128, 128]
				target_SOS = data['sos'].type(torch.cuda.FloatTensor).cuda(args.device) # Tensor size [4, 1, 128, 128]
				target_ESD  = data['esd'].type(torch.cuda.FloatTensor).cuda(args.device) # Tensor size [4, 1, 128, 128]
				target_ESC  = data['esc'].type(torch.cuda.FloatTensor).cuda(args.device) # Tensor size [4, 1, 128, 128]
				target = torch.cat((target_AT, target_SOS), 1).contiguous()

				x_curr = input
				image_curr = target

				with autocast():
					AT_pred, I16, I32, I64 = CardNet(x_curr)
					loss = nn.MSELoss()(AT_pred, image_curr) + nn.MSELoss()(I16, image_curr[:, :, ::8, ::8]) \
						   + nn.MSELoss()(I32, image_curr[:, :, ::4, ::4]) + nn.MSELoss()(I64, image_curr[:, :, ::2, ::2])
				psnr = []
				for idx in range(2):
					psnr.append(20 * (1 / (((AT_pred[:,idx,:,:] - image_curr[:,idx,:,:]) ** 2).mean(dim=2).mean(dim=1).sqrt())).log10().mean().item())

				val_loss += loss.item()
				val_psnr = [sum(x) for x in zip(val_psnr, psnr)]

		train_loss_epoch = train_loss / len(trainloader_v3)
		train_psnr_epoch = [x/len(trainloader_v3) for x in train_psnr]
		val_loss_epoch = val_loss / len(valloader)
		val_psnr_epoch = [x / len(valloader) for x in val_psnr]
		if args.neptune:
			run["logs/Train Loss"].append(train_loss_epoch)
			run["logs/Train PSNR ATT"].append(train_psnr_epoch[0])
			run["logs/Train PSNR SOS"].append(train_psnr_epoch[1])
			run["logs/Validation Loss"].append(val_loss_epoch)
			run["logs/Validation PSNR ATT"].append(val_psnr_epoch[0])
			run["logs/Validation PSNR SOS"].append(val_psnr_epoch[1])

		if args.save:
			state = {'epoch': epoch + 1, 'state_dict': CardNet.state_dict(), 'optmizer': CardNet_optim.state_dict()}
			if (epoch + 1) % 10 == 0:
				torch.save(state, './results/CardNet_Epoch{}.pt'.format(epoch + 1))
			if sum(val_psnr_epoch) > max_psnr:
				max_psnr = sum(val_psnr_epoch)
				torch.save(state, './results/CardNet_max.pt')

			for idx in range(2):
				gt_image = ((image_curr)[0][idx].data.cpu().numpy().transpose() * 255).astype(np.uint8)
				pred_image = ((AT_pred)[0][idx].data.cpu().numpy().transpose() * 255).astype(np.uint8)
				imageio.imwrite(r'./results/Epoch{}_val_{}_target.png'.format(epoch + 1, variables[idx]), gt_image)
				imageio.imwrite(r'./results/Epoch{}_val_{}_predicted.png'.format(epoch + 1, variables[idx]), pred_image)


if __name__ == '__main__':
	main()