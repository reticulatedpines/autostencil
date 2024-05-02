#!/usr/bin/env python3

import os
import argparse
import cv2 as cv
import numpy as np

def main():
    args = parse_args()
    image = cv.imread(args.input)
    image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)

    # retrieve all unique colours
    colours = get_colours(image)
    
    # create one layer per colour
    for i, c in enumerate(colours):
        mask = cv.inRange(image, c, c)
        layer = np.copy(image)
        layer[mask == 0] = (0, 0, 0, 0)
        image_path = os.path.join(args.output, "layer_%02d.png" % i)
        cv.imwrite(image_path, layer)


def get_colours(image):
    b, g, r, a = cv.split(image)
    b = b.astype(np.uint32)
    g = g.astype(np.uint32)
    r = r.astype(np.uint32)
    a = a.astype(np.uint32)
    combined_channels = b + (g << 8) + (r << 16) + (a << 24)
    uniques = np.unique(combined_channels)
    # unmunge and return in a sensible format
    colours = []
    for c in uniques:
        colours.append([c & 0xff,
                        (c >> 8) & 0xff,
                        (c >> 16) & 0xff,
                        (c >> 24) & 0xff])
    return np.array(colours)


def parse_args():
    description = '''
    Split an image into colour based layers, perhaps made using st_posterise.py.
    '''

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("input",
                        help="image file")
    parser.add_argument("output",
                        help="dir (folder) name for output layers")

    args = parser.parse_args()
    if not os.path.isfile(args.input):
        print("input file didn't exist: '%s'" % args.input)
        exit()
    if not os.path.isdir(args.output):
        print("output dir didn't exist: '%s'" % args.output)
        exit()

    return args


if __name__ == "__main__":
    main()
