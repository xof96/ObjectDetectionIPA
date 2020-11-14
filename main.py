# import argparse
import matplotlib.pyplot as plt
import numpy as np
import utils.utils as utils
import skimage.morphology as morph

# if __name__ == '__main__':
# parser = argparse.ArgumentParser()
# parser.add_argument("-image_path", type=str)
# args = parser.parse_args()
# if not args.image_path:
#     parser.error('-image_path is required.')
# img_path = args.image_path

# %%
img_path = 'assets/web_im3.jpg'
image = utils.imread(filename=img_path, as_gray=False)

# %%
print(image.shape)
im_red = image[:, :, 0]
im_green = image[:, :, 1]
im_blue = image[:, :, 2]
# Yellow
im_yellow = im_red.astype(np.float64) + im_green.astype(np.float64) - im_blue.astype(np.float64)
im_yellow = utils.to_uint8(im_yellow / (255.0 + 255.0))

# %%
img_op = morph.erosion(im_yellow, morph.disk(radius=0))
img_proc = im_yellow - img_op

# %%
# Playing with binary representation of the picture
th = 150
# th = utils.get_threshold_otsu(gray_im=im_yellow)
print('Getting binary image with threshold =', th)
bin_image = utils.threshold(im_yellow, th)
bin_image = morph.binary_erosion(image=bin_image, selem=morph.disk(radius=0))

# %%
ig, ax = plt.subplots(1, 4)
ax[0].imshow(im_yellow, cmap='gray')
ax[1].imshow(img_op, cmap='gray', vmin=0, vmax=255)
ax[2].imshow(img_proc, cmap='gray', vmin=0, vmax=255)
ax[3].imshow(bin_image*255, cmap='gray', vmin=0, vmax=255)
plt.show()
