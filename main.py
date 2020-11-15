import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import utils.utils as utils
import skimage.morphology as morph
import skimage.restoration as restoration
import scipy.ndimage.morphology as sp_morph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-image_path", type=str)
    parser.add_argument("-d_lambda", type=int, default=1)
    parser.add_argument("-bth_type", type=str, default="otsu")  # or 'custom'
    parser.add_argument("-bin_th", type=int, default=150)

    parser.add_argument("-erosion", action="store_true")
    parser.add_argument("-re", type=int, default=5)

    parser.add_argument("-dilation", action="store_true")
    parser.add_argument("-rd", type=int, default=5)

    parser.add_argument("-opening", action="store_true")
    parser.add_argument("-ro", type=int, default=5)

    parser.add_argument("-closing", action="store_true")
    parser.add_argument("-rc", type=int, default=5)

    parser.add_argument("-rm_ts", type=int, default=100)

    parser.add_argument("-r_error_th", type=int, default=3)
    parser.add_argument("-output_file", type=str, default='output.txt')
    args = parser.parse_args()
    if not args.image_path:
        parser.error('-image_path is required.')
    img_path = args.image_path

    # Opening the image
    image = utils.imread(filename=img_path, as_gray=False)

    # Separating layers
    red_image = image[:, :, 0]
    green_image = image[:, :, 1]
    blue_image = image[:, :, 2]
    # Yellow layer
    yellow_image = red_image.astype(np.float64) + green_image.astype(np.float64) - blue_image.astype(np.float64)
    yellow_image = utils.to_uint8(yellow_image / (255.0 + 255.0))

    # Denoising
    d_lambda = args.d_lambda
    if d_lambda > 0:
        den_image = (restoration.denoise_tv_chambolle(image=yellow_image, weight=d_lambda) * 255).astype(np.uint8)
    else:
        den_image = yellow_image

    # Binary representation of the picture
    bth_type = args.bth_type
    bin_th = args.bin_th
    if bth_type == 'otsu':
        bth = utils.get_threshold_otsu(gray_im=den_image)
        print('Getting binary image with otsu threshold =', bth)
    else:
        # bth = 150 is a good threshold for some images with white background
        bth = bin_th

    bin_image = utils.threshold(den_image, bth)

    # Applying morphology processing
    erosion = args.erosion
    re = args.re
    dilation = args.dilation
    rd = args.rd
    opening = args.opening
    ro = args.ro
    closing = args.closing
    rc = args.rc

    if erosion or dilation or opening or closing:
        if erosion:
            proc_image = morph.binary_erosion(image=bin_image, selem=morph.disk(radius=re))
        if dilation:
            proc_image = morph.binary_dilation(image=bin_image, selem=morph.disk(radius=rd))
        if opening:  # radius = 5 for web_im2
            proc_image = morph.binary_opening(image=bin_image, selem=morph.disk(radius=ro))
        if closing:
            proc_image = morph.binary_closing(image=bin_image, selem=morph.disk(radius=rc))
    else:
        proc_image = bin_image
    # Getting connected components
    cc_list = utils.get_ccomponents(bw_image=proc_image)
    rm_ts = args.rm_ts

    # Removing small components
    if rm_ts > 0:
        cc_list = utils.remove_small_components(cc_list=cc_list, target_size=rm_ts)

    # Creating binary image out of components
    f_bin_image = utils.cc2image(cc_list=cc_list, image_shape=yellow_image.shape)
    # Filling holes
    f_bin_image = sp_morph.binary_fill_holes(input=f_bin_image)

    # Filtering circle shaped components
    r_error_th = args.r_error_th
    f_cc_list = utils.filtering_circles(cc_list=cc_list, error_th=r_error_th)

    # Plotting
    plt_rows = 1
    plt_cols = 2
    fig, ax = plt.subplots(nrows=plt_rows, ncols=plt_cols)
    ax[0].set_axis_off()
    ax[0].set_title('Original Image')
    ax[1].set_axis_off()
    ax[1].set_title('Yellow Components')
    ax[0].imshow(image)
    ax[1].imshow(f_bin_image, cmap='plasma')
    for cc in f_cc_list:
        bbox = cc['bbox']
        rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3], bbox[2], edgecolor='g', linewidth='2', facecolor='none')
        ax[0].add_patch(rect)
    plt.show()

    # Writing out the components file
    output_file = args.output_file
    with open(output_file, 'w') as output:
        for x in cc_list:
            output.write(utils.cc_to_str(x))
