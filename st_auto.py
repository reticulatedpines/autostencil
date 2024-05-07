#!/usr/bin/env python3

import os
import argparse
import tempfile
import io
import zipfile

import cv2 as cv

import st_posterise
import st_split_layers
import st_png_to_svg

def main():
    args = parse_args()
    image = cv.imread(args.input)

    # get output zip name
    in_name = os.path.split(args.input)[1] # filename only
    in_prefix = in_name.rsplit(".", maxsplit=1)[0]
    out_name = in_prefix + ".zip"

    # st_posterise
    image = posterise(image, max_colours=6)

    # st_split_layers, prior output -> set of single-colour layers
    image_a = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
    layers = st_split_layers.get_layers(image_a)

    # st_png_to_svg, each layer -> svg
    con_w_h_colour = [st_png_to_svg.convert_rgba_to_contours(layer) for layer in layers]
    SVGs = [st_png_to_svg.contours_to_svg_string(c[0], c[1], c[2], c[3]) for c in con_w_h_colour]

    # zip the in-memory svg "files"
    mf = io.BytesIO()
    with zipfile.ZipFile(mf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, s in enumerate(SVGs):
            zf.writestr("layer_%02d.svg" % i, s)

    # and save!
    with open(out_name, "wb") as f:
        f.write(mf.getvalue())


def parse_args():
    description = '''
    Turn an image into a set of stencils ready to use for laser cutting.
    Automatically converts image into a small number of regions by
    colour and shape.
    '''

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("input",
                        help="image file")

    args = parser.parse_args()
    if not os.path.isfile(args.input):
        print("input file didn't exist: '%s'" % args.input)
        exit()

    return args


def posterise(image, max_colours):
    """
    Using some often useful default pre-processing,
    posterise the input image and return the result
    """
    st_posterise.dark_to_black(image)

    st_posterise.contrast_brightness(image, contrast=1.4)

    image = st_posterise.smooth_bilateral(image)

    st_posterise.mean_shift_segment(image)

    image = st_posterise.kmeans(image, max_colours=max_colours)

    st_posterise.light_to_white(image)
    return image


if __name__ == "__main__":
    main()
