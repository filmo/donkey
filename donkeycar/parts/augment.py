'''
    File: augment.py
    Author : Tawn Kramer
    Date : July 2017
'''
import random
from PIL import Image
from PIL import ImageEnhance
import glob
import numpy as np
import math

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

def rand_cb_idx():
    '''
    Helper function for rand_color_balance so that we can use a repeatable color balance if desired
    :return: int index for use with rand_color_balance
    '''

    # values below 2000 seem highly unrealistic.
    # this index corresponds with kelvin_table dictionary in rand_color_balance
    kelvin_table_idx = np.array([2000,2500,3000,3500,4000,4500,5000,
                                 5500,6000,6500,7000,7500,8000,8500,9000,
                                 9500,10000,11000,12000])

    return np.random.choice(kelvin_table_idx)

def rand_color_balance_np(img_np,cb_idx=False):
    '''
    Largely younked from
    https://stackoverflow.com/questions/11884544/setting-color-temperature-for-a-given-image-like-in-photoshop

    Black Body Numbers in HEX:
    http://www.vendian.org/mncharity/dir3/blackbody/

    Assumes that the image was mostly white-balanced correctly to begin with. Recording in certain artifical lighting
    situations will violate that assumption. Particularly sodium vapor lights which are not full spectrum.

    This is a Numpy only solution and avoids a call to PIL which can involve costly Python GILs

    See also: rand_cb_idx()

    :param img_np: image converted to numpy format. (H,W,3)
    :return: a white balance perturbation as numpy array of (H,W,3)
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
        cb_idx = rand_cb_idx()
    r, g, b = kelvin_table[cb_idx]

    # apply the black body adjustment
    matrix = np.array([[r / 255.,   0,          0       ],
                       [0,          g / 255.,   0       ],
                       [0,          0,          b / 255.]])

    new_color_balance = img_np.dot(matrix).round().astype(np.uint8)

    return new_color_balance

def rand_color_balance(img,cb_idx=False):
    '''
    Largely younked from 
    https://stackoverflow.com/questions/11884544/setting-color-temperature-for-a-given-image-like-in-photoshop

    Black Body Numbers in HEX:
    http://www.vendian.org/mncharity/dir3/blackbody/

    Assumes that the image was mostly white-balanced correctly to begin with. Recording in certain artifical lighting 
    situations will violate that assumption. Particularly sodium vapor lights which are not full spectrum.

    See also: rand_cb_idx()

    :param img: image from Pi Camera in Image PIL format.
    :return: a white balance perturbation.
    '''
    img_converted = img.copy()

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
        cb_idx = rand_cb_idx()
    r, g, b = kelvin_table[cb_idx]

    #print('Picked:',cb_idx)

    # apply the black body adjustment
    matrix = (r/255.0,    0.0,      0.0,      0.0,
              0.0,        g/255.0,  0.0,      0.0,
              0.0,        0.0,      b/255.0,  0.0)

    return img_converted.convert('RGB', matrix)

def white_noise(img,var=0.05,lum_only=True):
    '''
    Take a PIL image and apply noise.
    :param img: PIL image object
    :param var: variance around 0 to multiply the pixel values by
    :param lum_only: TRUE = brightness noise only, FALSE = Color Noise
    :return: PIL image object.
    '''
    h = img.height
    w = img.width
    channels    = img.split()
    img_np      = np.array(img)

    if lum_only:
        # generate one noise mask and apply to all channels
        # The noise will only affect the relative brightness, but not color
        noise_ary = np.random.normal(size=(h, w), scale=var)

        for i in range(len(channels)):
            # apply the noise elementwise as a percentage of the existing value
            subtotal = np.multiply(noise_ary,img_np[:,:,i]) + img_np[:,:,i]
            # clip the values to acceptable uint8 range
            np.clip(a=subtotal,a_max=255,a_min=0,out=subtotal)
            # convert the floating values to nearest unit8s
            img_np[:,:,i] = np.array(subtotal,dtype=np.uint8)
    else:
        # apply noise so that it effects color as well.
        for i in range(len(channels)):
            # apply different noise values to each channel
            noise_ary = np.random.normal(size=(h, w), scale=var)

            # apply the noise elementwise as a percentage of the existing value
            subtotal = np.multiply(noise_ary,img_np[:,:,i]) + img_np[:,:,i]
            # clip the values to acceptable uint8 range
            np.clip(a=subtotal,a_max=255,a_min=0,out=subtotal)
            # convert the floating values to nearest unit8s
            img_np[:,:,i] = np.array(subtotal,dtype=np.uint8)

    return Image.fromarray(img_np)

def contrast_np(np_img,factor=False,low=0.75,high=1.5):
    '''
    Pure numpy contrast to avoid GIL when runing augmentation via Python multithreading/multiprocessing
    :param np_img: numpy image array (H,W,3)
    :param factor: Use passed in factor or random between low and high if False.
    :param low: low factor (less contrast)
    :param high: high factor (more contrast)
    :return: numpy ary with contrast adjusted (H,W,3)
    '''
    if not factor:
        factor = random.uniform(low,high)
    return np.clip(128 + factor * np_img - factor * 128, 0, 255).astype(np.uint8)

def brightness_np(np_img,factor=False,low=0.75, high=1.5):
    '''
    Replacement for PIL brightness adjustment. Avoid expensive Python GIL when doing
    multithreaded/multi-processing Augmentation.

    :param np_img: numpy image array (H,W,3)
    :param factor: Use passed in factor or random between low and high if False.
    :param low: low factor (darker)
    :param high: high factor (brighter)
    :return: numpy ary with brightness adjusted (H,W,3)
    '''
    if not factor:
        factor = random.uniform(low,high)

    ajusted_brightness = np_img * factor
    ajusted_brightness = np.clip(ajusted_brightness, 0, 255)
    ajusted_brightness = ajusted_brightness.astype(np.uint8)
    return ajusted_brightness

def augment_image(np_img, shadow_images=None, do_warp_persp=False, do_cb=False, do_noise=False):

    #change the coloration, sharpness, and composite a shadow
    # factor = random.uniform(0.75, 1.5)
    # img = ImageEnhance.Brightness(img).enhance(factor)

    np_img = brightness_np(np_img)

    img = Image.fromarray(np_img)

    factor = random.uniform(0.75, 1.25)
    img = ImageEnhance.Contrast(img).enhance(factor)

    factor = random.uniform(0.5, 1.5)
    img = ImageEnhance.Sharpness(img).enhance(factor)

    factor = random.uniform(0.7, 1.0)
    img = ImageEnhance.Color(img).enhance(factor)

    if shadow_images is not None:
        '''
        optionaly composite a shadow, perpared from load_shadow_images
        '''
        iShad = random.randrange(0, len(shadow_images))
        top, mask = shadow_images[iShad]
        theta = random.randrange(-35, 35)
        mask.rotate(theta)
        top.rotate(theta)
        mask = ImageEnhance.Brightness(mask).enhance(random.uniform(0.3, 1.0))
        offset = (random.randrange(-128, 128), random.randrange(-128, 128))
        img.paste(top, offset, mask)
    
    if do_warp_persp:
        '''
        optionaly warp perspective
        '''
        img = rand_persp_transform(img)

    if do_cb:
        '''
        Optional color balance shift
        '''
        img = rand_color_balance(img=img)

    if do_noise:
        '''
        White noise to either just the luminance or to color as well
        '''
        img = white_noise(img=img,var=0.05,lum_only=True)

    return np.array(img)

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


