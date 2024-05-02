#!/usr/bin/env python3

import os
import argparse
import cv2 as cv
import numpy as np

def main():
    args = parse_args()
    image = cv.imread(args.input)

    dark_to_black(image)

    # colour contrast + brightness boost?
    contrast_brightness(image, contrast=args.contrast)

    image = smooth_bilateral(image)

    mean_shift_segment(image)

    image = kmeans(image, max_colours=args.colours)
#    posterise(image)

    light_to_white(image)

    cv.imwrite(args.output, image)


def contrast_brightness(image, contrast:float=1.4, brightness:int=0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    brightness += int(round(255 * (1 - contrast) / 2))
    return cv.addWeighted(image, contrast, image, 0, brightness, image)


def light_to_white(image):
    light_lo=np.array([215, 215, 215])
    light_hi=np.array([255, 255, 255])

    # Mask image to only select lights, force white
    mask = cv.inRange(image, light_lo, light_hi)
    image[mask > 0] = (255, 255, 255)


def dark_to_black(image):
    dark_lo=np.array([0,0,0])
    dark_hi=np.array([50,50,50])

    # Mask image to only select darks, force black
    mask = cv.inRange(image, dark_lo, dark_hi)
    image[mask > 0] = (0, 0, 0)


def smooth_bilateral(image):
    # like a blur, but preserves edges better
    return cv.bilateralFilter(image, 15, 30, 30)


def kmeans(image, max_colours=6):
    colours = max_colours
    rounds = 1
    h, w = image.shape[:2]
    samples = np.zeros([h * w, 3], dtype=np.float32)
    count = 0
    
    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv.kmeans(samples,
                colours,
                None,
                (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
                rounds,
                cv.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))


def mean_shift_segment(image, spatial_grouping=30, colour_grouping=10):
    # Higher: more accurate shapes, finer detail.  Slower.
    # This one is quite subjective, so start low,
    # it's faster.
    spatial_distance = 45

    # lower: larger number of more detailed groups, slower.
    # Too low is "bitty" / grainy,
    # too high and groups tend to merge together.
    colour_distance = 20
    cv.pyrMeanShiftFiltering(image, spatial_distance, colour_distance, image)


def posterise(image):
    n = 3
    for i in range(n):
        image[(image >= i * 255 / n)
              & (image < (i + 1) * 255 / n)] = i * 255 / (n - 1)


def count_colours(image):
    b, g, r = cv.split(image)
    combined_channels = b \
                        + 1000 * (g + 1) \
                        + 1000 * 1000 * (r + 1)
    unique = np.unique(combined_channels)
    return len(unique)


def parse_args():
    description = '''
    Smart posterise a supplied image.
    '''

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("input",
                        help="image file")
    parser.add_argument("output",
                        help="posterised image file to create")
    parser.add_argument("--colours", "-c",
                        default=6,
                        type=int,
                        help="Max colours in the posterised output")
    parser.add_argument("--contrast",
                        default=1.4,
                        type=float,
                        help="Enhance or reduce contrast, default 1.4, an increase")

    args = parser.parse_args()
    if not os.path.isfile(args.input):
        print("input file didn't exist: '%s'" % args.input)
        exit()

    return args


if __name__ == "__main__":
    main()