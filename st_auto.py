#!/usr/bin/env python3

import os
import argparse
import tempfile
import io
import zipfile
import math

import cv2 as cv

import st_posterise
import st_split_layers
import st_png_to_svg

def main():
    args = parse_args()
    image = cv.imread(args.input)

    # st_posterise
    image = st_posterise.default_posterise(image, max_colours=args.max_colours)

    # st_split_layers, prior output -> set of single-colour layers
    image_a = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
    layers = st_split_layers.get_layers(image_a, split_more=True)

    # st_png_to_svg, each layer -> svg
    image_size = image.shape[0] * image.shape[1]
    size_filter = image_size * 0.00002 # low aggression trim of small regions
    con_w_h_colour = [st_png_to_svg.convert_rgba_to_contours(layer, size_filter=size_filter)
                            for layer in layers]
    SVGs = [st_png_to_svg.contours_to_svg_string(c[0], c[1], c[2], c[3])
                for c in con_w_h_colour if len(c[0]) > 0]

    # get output names for zip
    in_name = os.path.basename(args.input) # filename only
    in_prefix, in_suffix = in_name.rsplit(".", maxsplit=1)
    preview_suffix = ".jpg"
    preview_name = in_prefix + "_preview" + preview_suffix
    out_name = in_prefix + ".zip"

    # convert the posterised version to something nice
    # to use as a preview
    res, poster_image = cv.imencode(preview_suffix, image)

    # zip the in-memory svg "files"
    mf = io.BytesIO()
    with zipfile.ZipFile(mf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, s in enumerate(SVGs):
            zf.writestr("layer_%02d.svg" % i, s)
        zf.writestr(preview_name, poster_image)
        # copy in original
        zf.write(args.input, os.path.basename(args.input))

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
    parser.add_argument("--max-colours", "-c",
                        default=6,
                        type=int,
                        help="Max colours (and therefore layers).  Default 6")

    args = parser.parse_args()
    if not os.path.isfile(args.input):
        print("input file didn't exist: '%s'" % args.input)
        exit()

    return args


if __name__ == "__main__":
    main()
