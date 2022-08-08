import numpy as np
# from scipy.misc import imread,imresize 已廢棄
import imageio.v2 as iio # imread / imsave
from PIL import Image # imresize
import matplotlib.pyplot as plt   

img = iio.imread('asset/iWantBoth.jpeg')
print(img.dtype, img.shape)
print(img)
# img_tinted = img * [1, 0.95, 0.9] # 調色 rgb 比例 
# 參考 https://linuxconfig.net/programming/how-to-process-images-with-python-pil-library.html#C6

# Resize the tinted image to be 300 by 300 pixels.
img_resized = np.array(Image.fromarray(img).resize((300,300)))

# Write the tinted image back to disk
iio.imsave('asset/iWantBoth_after.jpg', img_resized)

