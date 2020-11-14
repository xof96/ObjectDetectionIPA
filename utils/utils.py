"""
A set of basic functions to operate on gray-scale images
@author: jsaavedr
"""

import numpy as np
import skimage.io as sk_io
import skimage.measure as measure
import scipy.ndimage.filters as nd_filters


# to read
def imread(filename, as_gray=False):
    image = sk_io.imread(filename, as_gray=as_gray)
    if image.dtype == np.float64:
        image = to_uint8(image)
    return image


# to uint8
def to_uint8(image):
    if image.dtype == np.float64:
        image = image * 255
    image[image < 0] = 0
    image[image > 255] = 255
    image = image.astype(np.uint8, copy=False)
    return image


# get histogram of an image
def get_histogram(gray_im):
    h = np.zeros(256, dtype=np.float32)
    for i in range(gray_im.shape[0]):
        for j in range(gray_im.shape[1]):
            h[gray_im[i, j]] += 1.0
    return h


# ----------------------------------------------
def threshold(gray_im, th):
    bin_im = np.zeros(gray_im.shape, np.uint8)
    bin_im[gray_im >= th] = 1
    return bin_im


# ----------------------------------------------
# compute accum
def get_accum(histogram, length=256):
    accum = np.zeros(length)
    accum[0] = histogram[0]
    for i in range(1, length):
        accum[i] = accum[i - 1] + histogram[i]
    return accum


# det otsu
def get_threshold_otsu(gray_im):
    h = get_histogram(gray_im)
    p = h / np.sum(h)
    accum = get_accum(p)
    mu_t = np.zeros(256, np.float32)
    # compute media for each t in [0..255]
    for i in range(0, 256):
        mu_t[i] = mu_t[i - 1] + i * p[i]
    mu = mu_t[255]
    best_t = 0
    best_val = 0
    eps = 0.0001
    for t in range(1, 256):
        w0 = accum[t]
        w1 = 1.0 - w0
        mu_0 = mu_t[t] / (w0 + eps)
        mu_1 = (mu - mu_t[t]) / (w1 + eps)
        val = w0 * (mu_0 - mu) * (mu_0 - mu) + w1 * (mu_1 - mu) * (mu_1 - mu)
        if val > best_val:
            best_val = val
            best_t = t
    return best_t


# equalization
def equalize_image(gray_im):
    h = get_histogram(gray_im)
    p = h / np.sum(h)
    accum = get_accum(p)
    imeq = np.zeros(gray_im.shape, np.float32)
    for i in range(imeq.shape[0]):
        for j in range(imeq.shape[1]):
            imeq[i, j] = 255.0 * accum[gray_im[i, j]]
    return to_uint8(imeq)


# gaussian  2D
def get_gaussian2d(sigma, radius):
    # radius= 3xsigma
    s = np.int(2 * radius + 1)
    mask = np.zeros([s, s])
    variance = sigma * sigma
    k = 1.0 / (2.0 * np.pi * variance)
    for u in range(-radius, radius + 1, 1):
        for v in range(-radius, radius + 1, 1):
            mask[u + radius, v + radius] = k * np.exp(-(u * u + v * v) / (2 * variance))
    return mask


# escalamiento lineal
def constrast_stretching(gray_im):
    a = gray_im.min()
    b = gray_im.max()
    im_t = np.zeros(gray_im.shape)
    for i in range(gray_im.shape[0]):
        for j in range(gray_im.shape[1]):
            im_t[i, j] = 255 * (gray_im[i, j] - a) / (b - a)
    return to_uint8(im_t)


def add_gaussian_noise(image, std):
    noise = np.random.normal(loc=0, scale=std, size=image.shape)
    noisy_image = image + noise
    noisy_image[image < 0] = 0
    noisy_image[image > 255] = 255
    return noisy_image.astype(np.uint8);


def get_borde(image, gx_kernel):
    gy_kernel = np.transpose(gx_kernel)
    gx = nd_filters.convolve(image.astype(np.float32), gx_kernel, mode='constant', cval=0)
    gy = nd_filters.convolve(image.astype(np.float32), gy_kernel, mode='constant', cval=0)
    borde = np.sqrt(gx ** 2 + gy ** 2)
    return borde


def get_image_from_lsb(gray_image):
    """
    lsb = least significant bits
    """
    bin_image = gray_image - ((gray_image >> 1) << 1)
    return bin_image


def set_image_on_lsb(gray_image, bin_image):
    gray_image[bin_image == 0] = (gray_image[bin_image == 0] >> 1) << 1
    gray_image[bin_image == 1] = ((gray_image[bin_image == 1] >> 1) << 1) + 1
    return gray_image


# Binary images processing methods
def is_valid(shape, i, j):
    if (i >= 0) and (j >= 0) and (i < shape[0]) and (j < shape[1]):
        return True
    else:
        return False


# getCC
def get_ccomponents(bw_image):
    print("--labeling ")
    bw_sets, n_cc = measure.label(bw_image, return_num=True)
    print("--labeling OK")
    cc_list = []
    for i_cc in range(1, n_cc + 1):
        inds = np.where(bw_sets == i_cc)
        points = list(zip(inds[0].tolist(), inds[1].tolist()))
        min_y = inds[0].min()
        max_y = inds[0].max()
        min_x = inds[1].min()
        max_x = inds[1].max()
        bbox = (min_y, min_x, max_y - min_y + 1, max_x - min_x + 1)
        boundary = get_boundary(points)
        cc_list.append({'id': i_cc,
                        'points': points,
                        'size': len(points),
                        'bbox': bbox,
                        'boundary': boundary})
    return cc_list


# remove components with size < target_size
def remove_small_components(cc_list, target_size):
    to_keep = []
    for i, cc in enumerate(cc_list):
        if (cc['size'] >= target_size):
            to_keep.append(i)
    new_cc_list = [cc_list[index] for index in to_keep]
    return new_cc_list


def cc2image(cc_list, image_shape, type='points'):
    '''
    cc_list : the list of connected components
    image_shape: the target shape
    type : 'points' or 'boundary'
    '''
    image = np.zeros(image_shape)
    for cc in cc_list:
        rows, cols = zip(*cc[type])
        image[rows, cols] = 1
    return image


# digital topology: getBoundary
def get_boundary(cc_points):
    # print (cc_points)
    if len(cc_points) == 1:
        return cc_points
    rows = [p[0] for p in cc_points]
    cols = [p[1] for p in cc_points]

    min_x = np.min(cols)
    min_y = np.min(rows)
    max_x = np.max(cols)
    max_y = np.max(rows)

    height = max_y - min_y + 1 + 2
    width = max_x - min_x + 1 + 2

    # creating a simple representation of the component
    cc_array = np.zeros([height, width], np.float32)
    # cc_rows and cc_cols with respect to the cc's size
    cc_rows = rows - min_y + 1
    cc_cols = cols - min_x + 1
    cc_array[cc_rows, cc_cols] = 1
    # print(cc_array)
    # neighbors
    l_r = [0, -1, -1, -1, 0, 1, 1, 1]
    l_c = [-1, -1, 0, 1, 1, 1, 0, -1]
    i = cc_rows[0]
    j = cc_cols[0]
    end = False
    p1_set = False
    P = (i, j)
    contour = [P]
    # first point is P
    # point at  right is Q
    idx_q = 0
    P0 = P
    P1 = (-1, -1)
    # Q0 = (i + l_r[0], j + l_c[0])
    while not end:
        Pant = P
        i = P[0]
        j = P[1]
        idx = idx_q
        # print("{} {} {} ".format(i,j,idx))
        # -------------------------------------------------------
        # moving  Q  P until Q=0 and P=1
        P = (i + l_r[idx], j + l_c[idx])
        Q = (i + l_r[(idx - 1 + 8) % 8], j + l_c[(idx - 1 + 8) % 8])
        while cc_array[P] != 1 or cc_array[Q] != 0:
            idx = (idx + 1) % 8
            Q = P
            P = (i + l_r[idx], j + l_c[idx])
        # -------------------------------------------------------
        # redefining the position of Q with respect to P
        if P[0] - Q[0] > 0:
            idx_q = 2
        elif P[0] - Q[0] < 0:
            idx_q = 6
        elif P[1] - Q[1] > 0:
            idx_q = 0
        elif P[1] - Q[1] < 0:
            idx_q = 4
        else:
            raise ValueError("something wrong")
            # stop condition
        if P == P1 and Pant == P0:
            end = True
        else:
            contour.append(P)
        if not p1_set:
            P1 = P
            p1_set = True

            # getting back to the real coordinates
    rows_p = [p[0] for p in contour]
    cols_p = [p[1] for p in contour]
    rows_p = rows_p + min_y - 1
    cols_p = cols_p + min_x - 1
    contour = list(zip(rows_p, cols_p))
    return contour
