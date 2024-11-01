import torch
from torch import nn, einsum
from einops import rearrange
from math import sqrt
import time
import torch.nn.functional as F

from mamba_ssm import Mamba


def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
	layers = []
	layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
						 kernel_size=kernel_size, stride=stride, padding=padding,
						 bias=bias)]
	layers += [nn.BatchNorm2d(num_features=out_channels)]
	layers += [nn.ReLU()]

	cbr = nn.Sequential(*layers)

	return cbr

def dn_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
	layers = []
	layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
						 kernel_size=2, stride=2, padding=0,
						 bias=bias)]
	layers += [nn.BatchNorm2d(num_features=out_channels)]
	layers += [nn.ReLU()]
	cbr = nn.Sequential(*layers)

	return cbr

def up_layer(in_channels, out_channels, kernel_size=3,  padding=1, bias=True):
	layers = []
	layers += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
						 kernel_size=2, stride=2, padding=0,
						 bias=bias)]
	layers += [nn.BatchNorm2d(num_features=out_channels)]
	layers += [nn.ReLU()]
	cbr = nn.Sequential(*layers)

	return cbr

def Sin_blk(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
	layers = []
	layers += [ResidualBlock(in_channels=in_channels, out_channels=out_channels)]
	layers += [ResidualBlock(in_channels=out_channels, out_channels=out_channels)]
	layers += [ResidualBlock(in_channels=out_channels, out_channels=out_channels)]
	layers += [ResidualBlock(in_channels=out_channels, out_channels=out_channels)]

	cbr = nn.Sequential(*layers)

	return cbr

class ResidualBlock(nn.Module):
	expansion = 1

	def __init__(self, in_channels, out_channels, stride=1):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(
			in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
							   stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_channels != self.expansion*out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, self.expansion*out_channels,
					kernel_size=1, stride=stride, bias=False),
					nn.BatchNorm2d(self.expansion*out_channels)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out

class Decoder(nn.Module):    # [B, 256*3, 4, 16] --> [B, 1, 31, 128]
	def __init__(self, in_channels):
		super().__init__()
		#################   HR NET   ############################
		self.dec1_16 = Sin_blk(in_channels, 512)

		self.mn1_16_16 = CBR2d(512, 512)
		self.up1_16_32 = up_layer(in_channels=512, out_channels=256,
								  kernel_size=2, padding=0, bias=True)

		#################   Second   ###################
		self.dec2A = CBR2d(in_channels=512, out_channels=512)
		self.dec2B = CBR2d(in_channels=256+128, out_channels=256)

		self.dec2_16 = Sin_blk(512, 512)
		self.dec2_32 = Sin_blk(256, 256)

		self.mn2_16_16A = CBR2d(512, 512)
		self.mn2_32_32B = CBR2d(256, 256)
		self.up2_16_32_1A = up_layer(in_channels=512, out_channels=256,
									 kernel_size=2, padding=0, bias=True)
		self.up2_16_32_2A = up_layer(in_channels=512, out_channels=256,
									 kernel_size=2, padding=0, bias=True)
		self.up2_32_64_2A = up_layer(in_channels=256, out_channels=128,
									 kernel_size=2, padding=0, bias=True)
		self.up2_32_64_1B = up_layer(in_channels=256, out_channels=128,
									 kernel_size=2, padding=0, bias=True)
		self.dn2_32_16_1B = dn_layer(in_channels=256, out_channels=512, stride=2)

		#################   Thrid   ###################
		self.dec3A = CBR2d(in_channels=512 * 2, out_channels=512)
		self.dec3B = CBR2d(in_channels=256 * 2, out_channels=256)
		self.dec3C = CBR2d(in_channels=128 * 2+64, out_channels=128)
		self.dec3_16 = Sin_blk(512, 512)
		self.dec3_32 = Sin_blk(256, 256)
		self.dec3_64 = Sin_blk(128, 128)

		self.mn3_16_16A = CBR2d(512, 512)
		self.up3_16_32_1A = up_layer(in_channels=512, out_channels=256,
									 kernel_size=2, padding=0, bias=True)
		self.up3_16_32_2A = up_layer(in_channels=512, out_channels=256,
									 kernel_size=2, padding=0, bias=True)
		self.up3_32_64_2A = up_layer(in_channels=256, out_channels=128,
									 kernel_size=2, padding=0, bias=True)
		self.up3_16_32_3A = up_layer(in_channels=512, out_channels=256,
									 kernel_size=2, padding=0, bias=True)
		self.up3_32_64_3A = up_layer(in_channels=256, out_channels=128,
									 kernel_size=2, padding=0, bias=True)
		self.up3_64_128_3A = up_layer(in_channels=128, out_channels=64,
									  kernel_size=2, padding=0, bias=True)

		self.dn3_32_16_1B = dn_layer(in_channels=256, out_channels=512, stride=2)
		self.mn3_32_32B = CBR2d(256, 256)
		self.up3_32_64_1B = up_layer(in_channels=256, out_channels=128,
									 kernel_size=2, padding=0, bias=True)
		self.up3_32_64_2B = up_layer(in_channels=256, out_channels=128,
									 kernel_size=2, padding=0, bias=True)
		self.up3_64_128_2B = up_layer(in_channels=128, out_channels=64,
									  kernel_size=2, padding=0, bias=True)

		self.dn3_64_32_2C = dn_layer(in_channels=128, out_channels=256, stride=2)
		self.dn3_32_16_2C = dn_layer(in_channels=256, out_channels=512, stride=2)
		self.dn3_64_32_1C = dn_layer(in_channels=128, out_channels=256, stride=2)
		self.mn3_64_64C = CBR2d(128, 128)
		self.up3_64_128_1C = up_layer(in_channels=128, out_channels=64,
									  kernel_size=2, padding=0, bias=True)

		#################   Fourth   ###################
		self.dec4A = CBR2d(in_channels=512 * 3, out_channels=512)
		self.dec4B = CBR2d(in_channels=256 * 3, out_channels=256)
		self.dec4C = CBR2d(in_channels=128 * 3, out_channels=128)
		self.dec4D = CBR2d(in_channels=64 * 3, out_channels=64)

		self.dec4_16 = Sin_blk(512, 512)
		self.dec4_32 = Sin_blk(256, 256)
		self.dec4_64 = Sin_blk(128, 128)
		self.dec4_128 = Sin_blk(64, 64)

		self.up4_16_32_3A = up_layer(in_channels=512, out_channels=256,
									 kernel_size=2, padding=0, bias=True)
		self.up4_32_64_3A = up_layer(in_channels=256, out_channels=128,
									 kernel_size=2, padding=0, bias=True)
		self.up4_64_128_3A = up_layer(in_channels=128, out_channels=64,
									  kernel_size=2, padding=0, bias=True)

		self.up4_32_64_2B = up_layer(in_channels=256, out_channels=128,
									 kernel_size=2, padding=0, bias=True)
		self.up4_64_128_2B = up_layer(in_channels=128, out_channels=64,
									  kernel_size=2, padding=0, bias=True)

		self.up4_64_128_1C = up_layer(in_channels=128, out_channels=64,
									  kernel_size=2, padding=0, bias=True)
		self.mn4_128_128D = CBR2d(64, 64)

		##############Fifth#######
		self.dec5D = CBR2d(in_channels=64 * 4, out_channels=64)

		self.dec5_128 = Sin_blk(64, 64)

		self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)

		self.to_img16 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1, bias=True)
		self.to_img32 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, bias=True)
		self.to_img64 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1, bias=True)

		self.drop_layer = nn.Dropout(p=0.2)

	def forward(self, x16, x32, x64):           # [B, 512, 4, 16]
		#################   First   ###################
		dec_16_1 = self.dec1_16(x16)
		I16 = self.to_img16(dec_16_1)

		dec_16_2 = self.mn1_16_16(dec_16_1)
		dec_32_2 = self.up1_16_32(dec_16_1)
		dec_32_2 = torch.cat((dec_32_2, x32), dim=1)

		#################   Second   ###################
		dec_16_2 = self.dec2A(dec_16_2)
		dec_32_2 = self.dec2B(dec_32_2)
		dec_16_2 = self.dec2_16(dec_16_2)
		dec_32_2 = self.dec2_32(dec_32_2)
		I32 = self.to_img32(dec_32_2)

		dec_16_3_1 = self.mn2_16_16A(dec_16_2)
		dec_16_3_2 = self.dn2_32_16_1B(dec_32_2)
		dec_16_3 = torch.cat((dec_16_3_1, dec_16_3_2), dim=1)

		dec_32_3_1 = self.mn2_32_32B(dec_32_2)
		dec_32_3_2 = self.up2_16_32_1A(dec_16_2)
		dec_32_3 = torch.cat((dec_32_3_1, dec_32_3_2), dim=1)

		dec_64_3_1 = self.up2_32_64_2A(self.up2_16_32_2A(dec_16_2))
		dec_64_3_2 =self.up2_32_64_1B(dec_32_2)
		dec_64_3 = torch.cat((dec_64_3_1, dec_64_3_2, x64), dim=1)

		#################   Third   ###################
		dec_16_3 = self.dec3A(dec_16_3)
		dec_32_3 = self.dec3B(dec_32_3)
		dec_64_3 = self.dec3C(dec_64_3)
		dec_16_3 = self.dec3_16(dec_16_3)
		dec_32_3 = self.dec3_32(dec_32_3)
		dec_64_3 = self.dec3_64(dec_64_3)
		I64 = self.to_img64(dec_64_3)

		dec_16_4_1 =  self.mn3_16_16A(dec_16_3)
		dec_16_4_2 = self.dn3_32_16_1B(dec_32_3)
		dec_16_4_3 = self.dn3_32_16_2C (self.dn3_64_32_2C(dec_64_3))
		dec_16_4 = torch.cat((dec_16_4_1, dec_16_4_2,dec_16_4_3), dim=1)

		dec_32_4_1 = self.up3_16_32_1A(dec_16_3)
		dec_32_4_2 = self.mn3_32_32B (dec_32_3)
		dec_32_4_3 = self.dn3_64_32_1C(dec_64_3)
		dec_32_4 = torch.cat((dec_32_4_1, dec_32_4_2,dec_32_4_3), dim=1)

		dec_64_4_1 = self.up3_32_64_2A (self.up3_16_32_2A (dec_16_3))
		dec_64_4_2 = self.up3_32_64_1B(dec_32_3)
		dec_64_4_3 = self.mn3_64_64C(dec_64_3)
		dec_64_4 = torch.cat((dec_64_4_1, dec_64_4_2,dec_64_4_3), dim=1)

		dec_128_4_1 =  self.up3_64_128_3A(self.up3_32_64_3A (self.up3_16_32_3A(dec_16_3)))
		dec_128_4_2 =  self.up3_64_128_2B(self.up3_32_64_2B(dec_32_3))
		dec_128_4_3 = self.up3_64_128_1C(dec_64_3)
		dec_128_4 = torch.cat((dec_128_4_1, dec_128_4_2,dec_128_4_3), dim=1)

		#################   Fourth   ###################
		dec_16_4 = self.dec4A(dec_16_4)
		dec_32_4 = self.dec4B(dec_32_4)
		dec_64_4 = self.dec4C(dec_64_4)
		dec_128_4 = self.dec4D(dec_128_4)

		dec_16_4 = self.dec4_16(dec_16_4)
		dec_32_4 = self.dec4_32(dec_32_4)
		dec_64_4 = self.dec4_64(dec_64_4)
		dec_128_4 = self.dec4_128(dec_128_4)

		dec_128_5_1 = self.up4_64_128_3A(self.up4_32_64_3A(self.up4_16_32_3A(dec_16_4)))
		dec_128_5_2 = self.up4_64_128_2B(self.up4_32_64_2B(dec_32_4))
		dec_128_5_3 = self.up4_64_128_1C(dec_64_4)
		dec_128_5_4 = self.mn4_128_128D(dec_128_4)
		dec_128_5 = torch.cat((dec_128_5_1, dec_128_5_2,dec_128_5_3,dec_128_5_4), dim=1)

		dec_128_5 = self.dec5D(dec_128_5)
		dec_128_5 = self.dec5_128(dec_128_5)
		x =self.out(dec_128_5)

		return x, I16, I32, I64


class LayerNorm(nn.Module):
	r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
	The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
	shape (batch_size, height, width, channels) while channels_first corresponds to inputs
	with shape (batch_size, channels, height, width).
	"""

	def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(normalized_shape))
		self.bias = nn.Parameter(torch.zeros(normalized_shape))
		self.eps = eps
		self.data_format = data_format
		if self.data_format not in ["channels_last", "channels_first"]:
			raise NotImplementedError
		self.normalized_shape = (normalized_shape,)

	def forward(self, x):
		if self.data_format == "channels_last":
			return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
		elif self.data_format == "channels_first":
			u = x.mean(1, keepdim=True)
			s = (x - u).pow(2).mean(1, keepdim=True)
			x = (x - u) / torch.sqrt(s + self.eps)
			x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

			return x


class MambaLayer(nn.Module):
	def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
		super().__init__()
		self.dim = dim
		self.norm = nn.LayerNorm(dim)
		self.mamba = Mamba(
			d_model=dim,  # Model dimension d_model
			d_state=d_state,  # SSM state expansion factor
			d_conv=d_conv,  # Local convolution width
			expand=expand,  # Block expansion factor
			bimamba_type="v3",
			nslices=num_slices,
		)

	def forward(self, x):
		B, C = x.shape[:2]
		x_skip = x
		assert C == self.dim
		n_tokens = x.shape[2:].numel()
		img_dims = x.shape[2:]
		x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
		x_norm = self.norm(x_flat)
		x_mamba = self.mamba(x_norm)

		out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
		out = out + x_skip

		return out


class MlpChannel(nn.Module):
	def __init__(self, hidden_size, mlp_dim, ):
		super().__init__()
		self.fc1 = nn.Conv2d(hidden_size, mlp_dim, 1)
		self.act = nn.GELU()
		self.fc2 = nn.Conv2d(mlp_dim, hidden_size, 1)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.fc2(x)
		return x


class GSC(nn.Module):
	def __init__(self, in_channles) -> None:
		super().__init__()

		self.proj = nn.Conv2d(in_channles, in_channles, 3, 1, 1)
		self.norm = nn.InstanceNorm2d(in_channles)
		self.nonliner = nn.ReLU()

		self.proj2 = nn.Conv2d(in_channles, in_channles, 3, 1, 1)
		self.norm2 = nn.InstanceNorm2d(in_channles)
		self.nonliner2 = nn.ReLU()

		self.proj3 = nn.Conv2d(in_channles, in_channles, 1, 1, 0)
		self.norm3 = nn.InstanceNorm2d(in_channles)
		self.nonliner3 = nn.ReLU()

		self.proj4 = nn.Conv2d(in_channles, in_channles, 1, 1, 0)
		self.norm4 = nn.InstanceNorm2d(in_channles)
		self.nonliner4 = nn.ReLU()

	def forward(self, x):
		x_residual = x

		x1 = self.proj(x)
		x1 = self.norm(x1)
		x1 = self.nonliner(x1)

		x1 = self.proj2(x1)
		x1 = self.norm2(x1)
		x1 = self.nonliner2(x1)

		x2 = self.proj3(x)
		x2 = self.norm3(x2)
		x2 = self.nonliner3(x2)

		x = x1 + x2
		x = self.proj4(x)
		x = self.norm4(x)
		x = self.nonliner4(x)

		return x + x_residual


class MambaEncoder(nn.Module):
	def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
				 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
		super().__init__()

		self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
		stem = nn.Sequential(
			  nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
			  )
		self.downsample_layers.append(stem)
		for i in range(2):
			downsample_layer = nn.Sequential(
				# LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
				nn.InstanceNorm2d(dims[i]),
				nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
			)
			self.downsample_layers.append(downsample_layer)

		self.stages = nn.ModuleList()
		self.gscs = nn.ModuleList()
		num_slices_list = [64, 32, 16, 8]
		cur = 0
		for i in range(3):
			gsc = GSC(dims[i])

			stage = nn.Sequential(
				*[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
			)

			self.stages.append(stage)
			self.gscs.append(gsc)
			cur += depths[i]

		self.out_indices = out_indices

		self.mlps = nn.ModuleList()
		for i_layer in range(3):
			layer = nn.InstanceNorm2d(dims[i_layer])
			layer_name = f'norm{i_layer}'
			self.add_module(layer_name, layer)
			self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

	def forward_features(self, x):
		outs = []
		for i in range(3):
			x = self.downsample_layers[i](x)
			x = self.gscs[i](x)
			x = self.stages[i](x)

			if i in self.out_indices:
				norm_layer = getattr(self, f'norm{i}')
				x_out = norm_layer(x)
				x_out = self.mlps[i](x_out)
				outs.append(x_out)

		return tuple(outs)

	def forward(self, x):
		x = self.forward_features(x)
		return x

class CardNet(nn.Module):
	'''
	Default values from Mix Transformer B0
	'''
	def __init__(self):
		super(CardNet, self).__init__()

		in_chans = 64
		depths = [2, 2, 2]
		feat_size = [64, 128, 256]
		drop_path_rate = 0
		layer_scale_init_value = 1e-6

		def enc_channel():
			layers = []
			layers += [CBR2d(in_channels=1, out_channels=4, stride=(3, 7), padding=(1, 3), kernel_size=(5, 9))]
			layers += [CBR2d(in_channels=4, out_channels=16, stride=(2, 2), padding=(1, 2))]
			layers += [CBR2d(in_channels=16, out_channels=32, stride=(1, 1), padding=(1, 2))]
			layers += [CBR2d(in_channels=32, out_channels=64, stride=(1, 1), padding=(1, 1))]
			layers += [CBR2d(in_channels=64, out_channels=64, stride=(1, 1), padding=(1, 1))]
			cbr = nn.Sequential(*layers)
			return cbr

		self.enc1_1 = enc_channel()

		self.vit = MambaEncoder(in_chans,
								depths=depths,
								dims=feat_size,
								drop_path_rate=drop_path_rate,
								layer_scale_init_value=layer_scale_init_value,
							  )

		self.Decoder = Decoder(256)

	def forward(self, x):                               # [B, 24, 48, 2640]
		enc1 = []
		for scanline in range(x.shape[1]):
			enc1.append(self.enc1_1(x[:,scanline,:,:].unsqueeze(dim=1)))
		enc1 = torch.cat(enc1, dim=2).contiguous()  	# [B, 64, 128, 128]
		# print(enc1.shape)
		layer_outputs = self.vit(enc1)  # [B, 64, 64, 64], [B, 128, 32,32], [B, 256, 16, 16]
		x64, x32, x16 = layer_outputs
		# print(x64.shape, x32.shape, x16.shape)

		x, I16, I32, I64 = self.Decoder(x16, x32, x64)

		image = nn.Sigmoid()(x)     # [B, 1, 192, 192]
		out = image, I16, I32, I64

		return out


def count_parameters(model):
	return sum(p.numel() for p in model.parameters())

if __name__ == '__main__':
	model = CardNet().cuda()
	print(count_parameters(model))
	sl = 24
	img = torch.rand((4, sl, 48, 2640)).cuda()
	bmode = torch.rand((4, 1, 128, 128)).cuda()
	for i in range(1):
		ts = time.time()
		output = model(img)
		print(f'OUTPUT Shape: {output[0].shape}')
		print('{:2.3f} sec.'.format(time.time()-ts))
		del output