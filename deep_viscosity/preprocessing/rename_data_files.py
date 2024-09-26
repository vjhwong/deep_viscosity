import os
import numpy as np
from scipy import interpolate


def find_interpolated_viscosities():
    percentages = [
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
    viscosities = [
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

    interpolated_percentages = [
        82.5,
        86,
        87,
        88,
        89,
        90.5,
        91.5,
        92.5,
        93.5,
        94.5,
        95.5,
        96.5,
        97.5,
    ]

    x = np.array(percentages)
    y = np.array(viscosities)

    f = interpolate.interp1d(x, y, kind="cubic")
    interpolated_viscosities = f(interpolated_percentages)

    percentages_all = percentages + interpolated_percentages
    percentages_all.sort()

    viscosities_all = viscosities + interpolated_viscosities.tolist()
    viscosities_all.sort()

    return percentages_all, viscosities_all

def rename_videos(data_path):
    percentages, viscosities = find_interpolated_viscosities()
    old_name_to_new_name = {}
    for percent, viscosity in zip(percentages, viscosities):
        old_name_to_new_name[str(percent)] = str(round(viscosity, 2))

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
    rename_videos(data_path)


if __name__ == "__main__":
    main()
