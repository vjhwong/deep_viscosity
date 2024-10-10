import os
import numpy as np
from scipy import interpolate


def find_interpolated_viscosities(desired_percentages: list[float]) -> tuple[list[int]]:
    """Finds the viscosity of glycerol solutions of different percentages based on known viscosities

    Args:
        desired_percentages: A list with the weight percentages of glycerol for which one seeks the corresponding viscosity

    Returns:
        tuple[list[int]]: A tuple with a list of percentage values and a list of corresponding viscosities
    """
    known_percentages = [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        65,
        67,
        70,
        75,
        80,
        85,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
    ]
    known_viscosities = [
        1.005,
        1.31,
        1.76,
        2.60,
        3.72,
        6.00,
        10.8,
        15.2,
        17.7,
        22.5,
        35.5,
        60.1,
        109,
        219,
        259,
        310,
        367,
        437,
        523,
        624,
        765,
        939,
        1150,
        1412,
    ]

    desired_percentages = list(map(lambda x: x * 100, desired_percentages))

    x = np.array(known_percentages)
    y = np.array(known_viscosities)

    f = interpolate.interp1d(x, y, kind="cubic")
    interpolated_viscosities = f(desired_percentages)

    desired_percentages.sort()
    interpolated_viscosities.sort()

    return (desired_percentages, interpolated_viscosities)

def rename_videos(data_path: str, percentages: list[float]):
    """Renames video files so they contain the viscosity instead of the percentage glycerol.

    Args:
        data_path: Path to video folders.
        percentages: A list with the weight percentages of glycerol for all videos
    """
    (percentages, viscosities) = find_interpolated_viscosities(percentages)
    old_name_to_new_name = {}
    for percent, viscosity in zip(percentages, viscosities):
        old_name_to_new_name[str(percent).rstrip('0').rstrip('.')] = str(round(viscosity, 2))
        
    for file in os.listdir(data_path):
        if "avi" in file:
            old_label = file.split("_", 1)
            old_file_name = os.path.join(data_path, file)
            new_file_name = os.path.join(
                data_path, old_name_to_new_name[old_label[0]] + old_label[1]
            )

            os.rename(old_file_name, new_file_name)


def main():
    data_path = "data/raw_modified"
    percentages = [
        0.98,
        0.9775,
        0.975,
        0.9725,
        0.97,
        0.9675,
        0.965,
        0.9625,
        0.96,
        0.9575,
        0.955,
        0.9525,
        0.95,
        0.9475,
        0.945,
        0.9425,
        0.94,
        0.9375,
        0.935,
        0.9325,
        0.93,
        0.9275,
        0.925,
        0.9225,
        0.92,
        0.9175,
        0.915,
        0.9125,
        0.91,
        0.9075,
        0.905,
        0.9025,
        0.9,
        0.89,
        0.88,
        0.87,
        0.86,
        0.85,
        0.825,
        0.8,
        0.75,
        0.7,
        0.65,
        0.6,
        0.5,
        0.4,
        0.3,
        0.2,
        0.1,
        0.0,
    ]
    rename_videos(data_path, percentages)


if __name__ == "__main__":
    main()
