from torchvision.utils import make_grid,save_image
import torch
from matplotlib import pyplot as plt

img = torch.randn([3, 1, 224, 224])
no_normalize = make_grid(img)
normalized = make_grid(img, normalize=True)
plt.imshow(no_normalize.permute(1, 2, 0))
plt.show()
plt.imshow(normalized.permute(1, 2, 0))
plt.imsave('1.png', normalized.permute(1, 2, 0).numpy()) #自动映射到[0,255]的范围内
save_image(normalized,'2.png')
plt.show()