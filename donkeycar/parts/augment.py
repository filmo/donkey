'''
    File: augment.py
    Author : Phil Glau (numpy/cv2 implementation), Tawn Kramer (original PIL implementation.)
    Date : July 2018, July 2017
'''
import random
from PIL import Image
from PIL import ImageEnhance
import glob
import numpy as np
import math
import cv2

'''
    find_coeffs and persp_transform borrowed from:
    https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
'''
def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def rand_persp_transform(img):
    width, height = img.size
    new_width = math.floor(float(width) * random.uniform(0.9, 1.1))
    xshift = math.floor(float(width) * random.uniform(-0.2, 0.2))
    coeffs = find_coeffs(
        [(0, 0), (256, 0), (256, 256), (0, 256)],
        [(0, 0), (256, 0), (new_width, height), (xshift, height)])

    return img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

def load_shadow_images(path_mask):
    shadow_images = []
    filenames = glob.glob(path_mask)
    for filename in filenames:
        shadow = Image.open(filename)
        shadow.thumbnail((256, 256))
        channels = shadow.split()
        if len(channels) != 4:
            continue
        r, g, b, a = channels
        top = Image.merge("RGB", (r, g, b))
        mask = Image.merge("L", (a,))
        shadow_images.append((top, mask))
    return shadow_images

''' Refactored Augmentations in pure numpy '''

def rand_color_balance_np(img_np,cb_idx=False,renormalize=False):
    '''
    Largely yoinked from
    https://stackoverflow.com/questions/11884544/setting-color-temperature-for-a-given-image-like-in-photoshop

    Black Body Numbers in HEX:
    http://www.vendian.org/mncharity/dir3/blackbody/

    Assumes that the image was mostly white-balanced correctly to begin with. Recording in certain artifical lighting
    situations will violate that assumption. Particularly sodium vapor lights which are not full spectrum.

    This is a Numpy only solution and avoids a call to PIL which can involve costly Python GILs

    :param img_np: image converted to numpy format. (H,W,3)
    :param cb_idx: choose a particular index to set white balance to. Otherwise choose one at random.
    :param renormalize: Clip and return as uint8

    :return: a white balance perturbation as numpy array of (H,W,3) (float64 unless renormalize=True)
    '''

    kelvin_table = {
            1000: (255, 56, 0),
            1500: (255, 109, 0),
            2000: (255, 137, 18),
            2500: (255, 161, 72),
            3000: (255, 180, 107),
            3500: (255, 196, 137),
            4000: (255, 209, 163),
            4500: (255, 219, 186),
            5000: (255, 228, 206),
            5500: (255, 236, 224),
            6000: (255, 243, 239),
            6500: (255, 249, 253),
            7000: (245, 243, 255),
            7500: (235, 238, 255),
            8000: (227, 233, 255),
            8500: (220, 229, 255),
            9000: (214, 225, 255),
            9500: (208, 222, 255),
            10000: (204, 219, 255),
            11000: (196, 215, 255),
            12000: (191, 211, 255)}

    if not cb_idx:
        # pick a color balance at random if not passed in as a parameter
        kelvin_table_idx = np.array([2000, 2500, 3000, 3500, 4000, 4500, 5000,
                                     5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000,
                                     9500, 10000, 11000, 12000])

        cb_idx = np.random.choice(kelvin_table_idx)

    # get the black body adjustment
    r, g, b = kelvin_table[cb_idx]

    # transform matrix for adjusment
    matrix = np.array([[r / 255.,   0,          0       ],
                       [0,          g / 255.,   0       ],
                       [0,          0,          b / 255.]])

    # perform the black body adjustment.
    new_color_balance = img_np.dot(matrix)

    if renormalize:
        new_color_balance = np.clip(img_np.dot(matrix).round(),0,255).astype(np.uint8)

    return new_color_balance

def white_noise_np(np_img,var=0.05,lum_only=True,normal_noise=False,renormalize=False):
    '''
    Take image stored as a (H,W,C) numpy array and add noise. This avoids
    any call to the PIL package. random.uniform is much less expensive
    than random.normal.

    :param np_img: numpy array in (H,W,C) format
    :param var: variance (sigma) around 0 to multiply the pixel values by
    :param lum_only: TRUE = brightness noise only, FALSE = Color Noise
    :param normal_noise: Use the more expensive random.normal noise.
    :param renormalize: Clip and return as uint8
    :return: numpy array of (H,W,C)
    '''
    h,w, channels  = np_img.shape

    if lum_only:
        # generate one noise mask and apply to all channels
        # The noise will only affect the relative brightness, but not color
        if not normal_noise:
            # uniform noise is much quicker to generate
            # var = ~0.10 to roughly approximate normal noise. (opinion)
            noise_ary = np.random.uniform(size=(h, w), low=-var, high=var)
        else:
            # var = ~0.05
            noise_ary = np.random.normal(size=(h, w), scale=var)

        # vectorized application of same 2D noise to all three channels
        img_np = noise_ary[:,:,np.newaxis] * np_img + np_img
    else:
        # apply noise so that it effects color as well.
        # apply different noise values to each channel
        if not normal_noise:
            noise_ary = np.random.uniform(size=(h, w,channels), low=-var, high=var)
        else:
            noise_ary = np.random.normal(size=(h, w,channels), scale=var)

        img_np = noise_ary * np_img + np_img

    if renormalize:
        img_np = np.clip(img_np, a_min=0, a_max=255).astype(np.uint8)

    return img_np

def contrast_np(np_img,factor=False,low=0.75,high=1.2,renormalize=False):
    '''
    Pure numpy contrast to avoid GIL when runing augmentation via Python multithreading/multiprocessing

    This is not equivalent to what PIL is doing. With Factor set to 0, this will return 127 for all pixel
    values. PIL is returning 99 for some reason.

    :param np_img: numpy image array (H,W,3)
    :param factor: Use passed in factor or random between low and high if False.
    :param low: low factor (less contrast)
    :param high: high factor (more contrast)
    :param renormalize: Clip and return as uint8
    :return: numpy ary with contrast adjusted (H,W,3) as float64 unless renormalize=True
    '''
    if factor == 0:
        factor = random.uniform(low,high)
    if renormalize:
        return np.clip(128 + factor * np_img - factor * 128, 0, 255).astype(np.uint8)

    return 128 + factor * np_img - factor * 128

def brightness_np(np_img,factor=False,low=0.75, high=1.5,renormalize=False):
    '''
    Replacement for PIL brightness adjustment. Avoid expensive Python GIL when doing
    multithreaded/multi-processing Augmentation.

    :param np_img: numpy image array (H,W,3)
    :param factor: Use passed in factor or random between low and high if False.
    :param low: low factor (darker)
    :param high: high factor (brighter)
    :param renormalize: Clip and return as uint8
    :return: numpy ary with brightness adjusted (H,W,3) as float64 unless renormalize=True
    '''
    if not factor:
        factor = random.uniform(low,high)

    ajusted_brightness = np_img * factor

    if renormalize:
        ajusted_brightness = np.clip(ajusted_brightness, 0, 255).astype(np.uint8)

    return ajusted_brightness

def sharpen_np(np_img,factor=False,low=0.50, high=1.5,renormalize=False):
    '''
    OpenCV2 implementation of PIL ImageSharpen filter. This is an approximation
    (3,3) sigma=2 seems the closest to the PIL implementation.
    :param np_img: (H,W,3)
    :param factor: 1 = same, <1 = blurry >1 <2 = sharpen
    :param low: minimum random pertubation
    :param high: maximum random pertubation
    :param renormalize: Clip and return as uint8
    :return: numpy array (H,W,3)
    '''
    if not factor:
        factor = random.uniform(low,high)

    alpha = factor
    beta = (1 - factor)

    # (3,3) sigma = 2.0 seem closest to ImageInhance version.
    # (5,5) sigma = 5.0 gives a bit more range.
    g3 = cv2.GaussianBlur(np_img, (5, 5), 5)
    unsharp_mask = cv2.addWeighted(np_img, alpha, g3, beta, 0)

    if renormalize:
        return np.clip(unsharp_mask,0,255).astype(np.uint8)

    return unsharp_mask

def color_np(np_img,factor=False,low=0.50,high=1.0,renormalize=False):
    '''
    This is actually slower than the built in PIL based on inital test 2.4x slower
    The cv2.cvtColor is expensive. HLS is marginally cheaper than HSV
    :param np_img: (H,W,3) numpy image
    :param factor: <1 = desaturation, >1 = more saturation
    :param low: minimum change
    :param high: maximum change
    :param renormalize: convert back to unit8
    :return: numpy array (H,W,3)
    '''
    if not factor:
        factor = random.uniform(low,high)

    img_hsv = cv2.cvtColor(np_img.astype(np.uint8),cv2.COLOR_RGB2HLS)

    # if the factor is < 1.0, we can safely remove the expensive astype conversions.
    # Any value less than 1.0 is a desaturation operation
    if factor >= 1.0:
        # in order to protect against overflow of uint8, move the image to float32
        img_hsv = img_hsv.astype(np.float32)

    img_hsv[:, :, 2] = img_hsv[:,:,2]*factor

    if factor >= 1.0:
        # image needs to be clipped back to uint8 prior to conversion back from HSV to RGB
        # or you'll get weird overflow artifacts
        return cv2.cvtColor(np.clip(img_hsv,0,255).astype(np.uint8),cv2.COLOR_HLS2RGB)
    else:
        img_RGB = cv2.cvtColor(img_hsv,cv2.COLOR_HLS2RGB)

    if renormalize:
        img_RGB =  np.clip(img_RGB,0,255).astype(np.uint8)

    return img_RGB

def augment_image(np_img, **kwargs):
    '''
    Takes an numpy image array, typically (H,W,3) and performs augmentations based on
    dictionary of values in kwargs. Most operations are quicker using pure Numpy/OpenCV
    transformations rather than using PIL which suffers from poor performance, particularly
    with regards to multiple workers.

    np_img starts typically starts out as a uint8 array. Some of these operations end up
    converting it to float64. Rather than clip and return it to uint8 after each operation,
    the np_img.float64 array is simply passed to the next operation.

    :param np_img: (H,W,n) image where n is number of channels (3 = RGB, 1 = Gray)
    :param kwargs: dictionary of transforms to perform.
    :return: augmented numpy array
    '''

    # Do Numpy Native Adjustments
    if kwargs.get('vary_sat'):
        # # even though by itself the PIL implementation is faster, when run
        # # in multithreaded training, PIL GIL blocks end up being more expensive
        # # than the compute saving.
        # if not 'img' in locals():
        #     img = Image.fromarray(np_img)
        # factor = random.uniform(0.7, 1.0)
        # img = ImageEnhance.Color(img).enhance(factor)
        # np_img = np.asarray(img)

        factor = random.uniform(0.50, 1.5)
        np_img = color_np(np_img,factor)

    if kwargs.get('vary_sharpness'):
        factor = random.uniform(0.50, 1.5)
        np_img = sharpen_np(np_img,factor)

    if kwargs.get('vary_bright'):
        factor = random.uniform(0.75, 1.5)
        np_img = brightness_np(np_img,factor)

    if kwargs.get('vary_contrast'):
        factor = random.uniform(0.75, 1.2)
        np_img = contrast_np(np_img,factor)

    if kwargs.get('vary_color_balance'):
        # Optional color balance shift
        np_img = rand_color_balance_np(np_img)

    if kwargs.get('add_noise'):
        # White noise to either just the luminance or to color as well
        var = 0.07
        lum = True
        use_norm = False
        if kwargs.get('noise_params'):
            # Lum = False = Color Noise, Lum = True = Luminance noise only.
            var,lum, use_norm = kwargs['noise_params']
        np_img = white_noise_np(np_img=np_img,var=var,lum_only=lum,normal_noise=use_norm)

    # functions that rely on PIL
    if kwargs.get('shadow_images'):
        #optionaly composite a shadow, perpared from load_shadow_images
        iShad = random.randrange(0, len(kwargs['shadow_images']))
        top, mask = kwargs['shadow_images'][iShad]
        theta = random.randrange(-35, 35)
        mask.rotate(theta)
        top.rotate(theta)
        mask = ImageEnhance.Brightness(mask).enhance(random.uniform(0.3, 1.0))
        offset = (random.randrange(-128, 128), random.randrange(-128, 128))
        if not 'img' in locals():
            img = Image.fromarray(np_img)
        img.paste(top, offset, mask)

    if kwargs.get('do_warp_persp'):
        #optionaly warp perspective
        if not 'img' in locals():
            img = Image.fromarray(np_img)
        img = rand_persp_transform(img)

    return np.clip(np_img,0,255)
