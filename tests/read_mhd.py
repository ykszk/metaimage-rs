import SimpleITK as sitk
import argparse
import os
import sys
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read and display metadata of a .mhd file."
    )
    parser.add_argument(
        "mhd_file", type=str, nargs="+", help="Path to the .mhd file(s) to read."
    )
    # test shape
    parser.add_argument(
        "--shape", type=int, nargs="*", help="Expected shape of the image."
    )
    # test value
    parser.add_argument(
        "--value", type=float, default=0.0, help="Expected value to check in the image."
    )
    args = parser.parse_args()

    for mhd_file_path in args.mhd_file:

        if not os.path.isfile(mhd_file_path):
            print(f"Error: The file {mhd_file_path} does not exist.")
            exit(1)

        # Read the .mhd file using SimpleITK

        image = sitk.ReadImage(mhd_file_path)

        # Display some metadata
        print("Image Size:", image.GetSize())
        print("Image Spacing:", image.GetSpacing())
        print("Image Origin:", image.GetOrigin())
        print("Image Direction:", image.GetDirection())
        print("Number of Components per Pixel:", image.GetNumberOfComponentsPerPixel())

        if args.shape:
            expected_shape = tuple(args.shape)
            actual_shape = image.GetSize()
            if actual_shape != expected_shape:
                print(
                    f"Error: Expected shape {expected_shape}, but got {actual_shape}."
                )
                sys.exit(1)
            else:
                print(f"Shape check passed: {actual_shape}")

        arr = sitk.GetArrayFromImage(image)
        assert np.all(
            arr == args.value
        ), f"Error: Not all values in the image are {args.value}."

    sys.exit(0)
