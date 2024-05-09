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

    if args.no_curves:
        curves = False
    else:
        curves = True

    contours, width, height, fill_colour = convert_rgba_to_contours(image, args.size_filter)
    svg_data = contours_to_svg_string(contours, width, height, fill_colour=fill_colour,
                                      curved_lines=curves)
    with open(args.output, "w") as f:
        f.write(svg_data)


def convert_rgba_to_contours(image, size_filter=0.0001, remove_nested=True):
    """
    Takes an RGBA image, returns a tuple containing:
    (list of contours, image_width, image_height, fill_colour).

    This is filtered to exclude anything below size_filter in area.

    It may also be filtered to remove nested regions; that is, regions that
    would, if cut from a sheet of material, not be attached after the cut.
    For some uses, this prevents wasteful cutting.
    """
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

    if remove_nested:
        contour_hierarchy = cv.RETR_EXTERNAL
    else:
        contour_hierarchy = cv.RETR_TREE
        
    contours, hierarchy = cv.findContours(binary_image,
                                          contour_hierarchy,
                                          cv.CHAIN_APPROX_SIMPLE)
    # OpenCV hierarchy is an array of
    # [next, prev, first_child, parent] arrays.  A value of -1
    # means the element doesn't exist (e.g., -1 in next means no
    # prior sibling, -1 in parent is no parent, etc).
    # Other values are the ID of the contour.
    #print(hierarchy)

    contours = [c for c in contours if cv.contourArea(c) > size_filter]

    height, width = np.shape(binary_image)
    return (contours, width, height, colour_code)


def contours_to_svg_string(contours, width, height, fill_colour="none",
                           curved_lines=True):
    svg = ""
    svg += '<svg fill="%s" ' % fill_colour \
               + 'fill-opacity="0.5" ' \
               + 'width="' + str(width) + '" height="' + str(height) \
               + '" xmlns="http://www.w3.org/2000/svg">'

    if curved_lines:
        line_curve_style = "Q" # quadratic beziers
    else:
        line_curve_style = "M" # plain move

    for c in contours:
        # Lightburn seems to require first point to be a move,
        # e.g. pure curves don't get displayed
#        svg += '<path stroke-dasharray="5,5" stroke-width="4" d="M'
        svg += '<path stroke-width="1" d="M'
        x, y = c[0][0]
        svg += "%d %d " % (x, y)
        if len(c) > 1:
            svg += line_curve_style
            for i in range(len(c) - 1):
                x, y = c[i][0]
                svg += "%d %d " % (x, y)
        svg += 'Z" style="stroke:%s"/>' % fill_colour

    # add registration markers
    radius = 25
    offset = radius + 20
    for x in [offset, width - offset]:
        for y in [offset, height - offset]:
            svg += '<circle cx="%d" cy="%d" r="%d" style="stroke:%s"/>' % (x, y, radius, fill_colour)

    # add border line
    offset = 5
    svg += '<polyline points="'
    svg += "%d, %d " % (offset, offset)
    svg += "%d, %d " % (offset, height - offset)
    svg += "%d, %d " % (width - offset, height - offset)
    svg += "%d, %d " % (width - offset, offset)
    svg += "%d, %d " % (offset, offset)
    svg += '" style="fill:none;stroke:%s"/>' % fill_colour

    svg += "</svg>"
    return svg


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
    parser.add_argument("--size-filter",
                        default=0.001,
                        type=float,
                        help="if specified, remove regions below this area, e.g. 1.23.  "
                             "Default is 0.001; only point-like regions are removed")
    parser.add_argument("--no-curves",
                        default=False,
                        action="store_true",
                        help="Default behaviour is to create regions with smooth lines."
                             "This option disables that, leading to regions made from "
                             "many short straight line.")
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
