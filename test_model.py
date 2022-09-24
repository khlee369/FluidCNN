import torch
from models.cnn1d import FluidCNN


# in features means time dimension (e.g. 12 hours)
# in ch means number of regions (e.g. seocho, yongsan, namhyun)
in_f = 12
in_ch = 3
model = FluidCNN(in_features=in_f , in_ch=in_ch )

batch_size = 8
x = torch.ones(batch_size , in_ch, in_f)
print(x.shape)
out = model(x)
print(out.shape)