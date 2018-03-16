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

def rand_color_balance(img):
    '''
    Largely younked from 
    https://stackoverflow.com/questions/11884544/setting-color-temperature-for-a-given-image-like-in-photoshop
    
    Assumes that the image was mostly white-balanced correctly to begin with. Recording in certain artifical lighting 
    situations will violate that assumption. Particularly sodium vapor lights which are not full spectrum.
    :param img: image from Pi Camera
    :return: a random white balance perturbation.
    '''
    img_converted = img.copy()
    kelvin_table_idx = np.array([1000,1500,2000,2500,3000,3500,4000,4500,5000,
                                5500,6000,6500,7000,7500,8000,8500,9000,9500,10000])
    kelvin_table_idx = np.array([2000,2500,3000,3500,4000,4500,5000,
                                5500,6000,6500,7000,7500,8000,8500,9000,9500,10000])

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
            10000: (204, 219, 255)}

    # pick a color balance at random.
    rand_idx = np.random.choice(kelvin_table_idx)
    r, g, b = kelvin_table[rand_idx]

    print('Picked:',rand_idx)
    matrix = (r/255.0,    0.0,      0.0,      0.0,
              0.0,        g/255.0,  0.0,      0.0,
              0.0,        0.0,      b/255.0,  0.0)

    return img_converted.convert('RGB', matrix)


def augment_image(np_img, shadow_images=None, do_warp_persp=False):
    img = Image.fromarray(np_img)
    #change the coloration, sharpness, and composite a shadow
    factor = random.uniform(0.75, 1.5)
    img = ImageEnhance.Brightness(img).enhance(factor)

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


