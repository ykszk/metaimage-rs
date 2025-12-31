import argparse
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write a .mhd file.")
    parser.add_argument(
        "mhd_file", type=str, nargs="+", help="Path to the .mhd file(s) to read."
    )
    parser.add_argument("--shape", type=int, nargs="+", help="Shape of the image.")
    parser.add_argument(
        "--value", type=float, required=True, help="Pixel value of the image."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        required=True,
        help="Data type of the image (e.g., 'float32', 'uint8').",
    )
    parser.add_argument(
        "--compress", action="store_true", help="Whether to compress the .mhd file."
    )

    args = parser.parse_args()

    # lazy import
    import numpy as np
    import SimpleITK as sitk

    for mhd_file_path in args.mhd_file:
        image_shape = tuple(args.shape)
        print(
            f"Creating image of shape {image_shape} with value {args.value} and dtype {args.dtype}"
        )
        arr = np.full(image_shape, args.value, dtype=args.dtype)

        image = sitk.GetImageFromArray(arr)
        sitk.WriteImage(image, mhd_file_path, useCompression=args.compress)

    sys.exit(0)
