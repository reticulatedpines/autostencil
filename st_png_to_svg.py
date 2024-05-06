#!/usr/bin/env python3

import os
import argparse
import cv2 as cv
import numpy as np

def main():
    args = parse_args()
    image = cv.imread(args.input, cv.IMREAD_UNCHANGED)
    channels = np.shape(image)[2]
    if channels != 4:
        print("Expected image in RGBA format, but this image "
              "has %d channels, not 4" % channels)
        exit(-1)

    # if image contains only one visible colour,
    # use that for lines and fills
    colours = get_colours(image)
    colours = [c for c in colours if c[3] == 255]
    if len(colours) == 1:
        c = colours[0]
        colour_code = "#" + "%02x" % c[2]
        colour_code += "%02x" % c[1]
        colour_code += "%02x" % c[0]
    else:
        colour_code = "#00ff00"

    # add border, so we can place registration markers
    b_size = 120
    image = cv.copyMakeBorder(image, b_size, b_size, b_size, b_size,
                              cv.BORDER_CONSTANT, value=[0, 0, 0, 0])

    # create a mask for all full alpha pixels; we expect these
    # to be the region of interest, everything else gets nuked
    mask = cv.inRange(image, (0, 0, 0, 255), (255, 255, 255, 255))

    image[mask > 0] = (255, 255, 255, 255)
    image[mask == 0] = (0, 0, 0, 255)

    # find regions / create contours
    binary_image = cv.extractChannel(image, 1)
    ret, binary_image = cv.threshold(binary_image, 1, 255, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(binary_image,
                                          cv.RETR_TREE,
                                          cv.CHAIN_APPROX_SIMPLE)

    if args.debug_output:
        # draw contours for debugging
        colour = (0, 255, 0, 255)
        cv.drawContours(image, contours, -1, colour, 1)
        mask = cv.inRange(image, colour, colour)
        image[mask == 0] = (0, 0, 0, 255)
        cv.imwrite(args.debug_output, image)

    # all regions defined as contours, convert to SVG
    height, width = np.shape(binary_image)
    save_contours_to_svg(contours, args.output, width, height, fill_colour=colour_code)


def save_contours_to_svg(contours, filename, width, height, fill_colour="none"):
    with open(filename, "w+") as f:
        f.write('<svg fill="%s" ' % fill_colour
                + 'width="' + str(width) + '" height="' + str(height)
                + '" xmlns="http://www.w3.org/2000/svg">')

        for c in contours:
            f.write('<path d="M')
            for i in range(len(c)):
                x, y = c[i][0]
                f.write("%d %d " % (x, y))
            f.write('Z" style="stroke:green"/>')

        # add registration markers
        radius = 25
        offset = radius + 20
        for x in [offset, width - offset]:
            for y in [offset, height - offset]:
                f.write('<circle cx="%d" cy="%d" r="%d" style="stroke:green"/>' % (x, y, radius))
        f.write("</svg>")


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
    Takes a PNG, outputs an SVG.  Can turn the output from st_split_layers.py
    into files useful for laser cutting, etc.  Input PNG must have an alpha channel;
    the SVG is created by finding borders between transparent and opaque regions.
    '''

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("input",
                        help="image file")
    parser.add_argument("--output", "-o",
                        default="",
                        help="output image file, defaults to input file with extension changed to .svg")
    parser.add_argument("--debug-output", "-d",
                        default="",
                        help="output debug image file")

    args = parser.parse_args()

    input_abs = os.path.abspath(args.input)
    input_start = input_abs.split(".")[0]
    output = input_start + ".svg"

    if not args.output:
        args.output = output

    if not os.path.isfile(args.input):
        print("input file didn't exist: '%s'" % args.input)
        exit()

    return args


if __name__ == "__main__":
    main()
