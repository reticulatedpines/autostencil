#!/usr/bin/env python3

import os
import argparse
import cv2 as cv
import numpy as np

def main():
    args = parse_args()
    image = cv.imread(args.input)
    image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)

    layers = get_layers(image, split_more=args.split_more)
    for i, layer in enumerate(layers):
        image_path = os.path.join(args.output, "layer_%02d.png" % i)
        cv.imwrite(image_path, layer)


def get_layers(image, split_more=False):
    """
    Return a list of layers, each having a single colour.

    If split_more is True, when nested regions of the same
    colour are found, these are split into new layers, such
    that all regions of that colour are connected to the base layer;
    this prevents disconnected areas of material being cut,
    when they would fall away from the main sheet.

    If split_more is False, each layer is a unique colour.
    If True, there may be more than one layer per colour,
    due to possible subdivisions.
    """
    # retrieve all unique colours
    colours = get_colours(image)
    
    # create layers per colour
    layers = []
    for i, c in enumerate(colours):
        mask = cv.inRange(image, c, c)
        layer = np.copy(image)
        layer[mask == 0] = (0, 0, 0, 0)
        layers.append(layer)

        if split_more:
            binary_image = cv.extractChannel(layer, 1)
            ret, binary_image = cv.threshold(binary_image, 1, 255, cv.THRESH_BINARY)
            contours, hierarchy = cv.findContours(binary_image,
                                                  cv.RETR_TREE,
                                                  cv.CHAIN_APPROX_NONE)

            # Because we're finding contours in a binary image, and we
            # previously set the wrong-colour regions all (0, 0, 0, 0),
            # we know two things:
            #
            # The top object in the hierarchy is coloured, and
            # all its grandchildren are the same colour (children are
            # transparent, or the inner boundary if you prefer).

            # Contour hierarchy is given as a flat list.  Each item is:
            # (next_sibling, prev_sibling, first_child, parent)
            # Parents always occur before their children.
            top_level = dict()
            children = dict()
            grandchildren = dict()
            if (hierarchy is not None) and len(hierarchy > 0):
                for i, h in enumerate(hierarchy[0]):
                    if h[3] == -1: # no parent, top-level
                        top_level[i] = contours[i]
                    elif h[3] in top_level: # child
                        children[i] = contours[i]
                    elif h[3] in children: # grandchild
                        grandchildren[i] = contours[i]

            # make a new layer for all grandchildren
            gc_layer = np.zeros_like(layer)
            gc = [g for g in grandchildren.values()]
            if len(gc):
                cv.drawContours(gc_layer, gc, -1, color=255, thickness=-1)
                # convert grayscale non-alpha to coloured alpha
                mask = cv.inRange(gc_layer, 255, 255)
                gc_layer[mask > 0] = (c[0], c[1], c[2], c[3])
                layers.append(gc_layer)

    return layers


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
    parser.add_argument("--split-more",
                        action="store_true",
                        help="After separating into one colour per layer, "
                             "should we create new layers for areas that are "
                             "nested and disconnected?  I.e., areas that would fall "
                             "off if you were cutting the stencils from paper.")

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
