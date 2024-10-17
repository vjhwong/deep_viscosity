import numpy as np

def get_border_pixels(frame: np.ndarray) -> list[int, int, int, int]:
    borders = []
    for i, row in enumerate(frame):
        if row.sum() > 0:
            borders.append(i)
            break
    for i, col in enumerate(frame.T):
        if col.sum() > 0:
            borders.append(i)
            break
    for i, row in enumerate(frame[::-1]):
        if row.sum() > 0:
            borders.append(len(frame) - i)
            break
    for i, col in enumerate(frame.T[::-1]):
        if col.sum() > 0:
            borders.append(len(frame.T) - i)
            break
    return borders

def get_window_size(frames: np.ndarray, padding=10) -> tuple[int, int, int, int]:
    top_border, left_border, bottom_border, right_border = (
        np.inf,
        np.inf,
        -np.inf,
        -np.inf,
    )

    for frame in frames:
        top, left, bottom, right = get_border_pixels(frame)
        if top_border > top:
            top_border = top
        if left_border > left:
            left_border = left
        if bottom_border < bottom:
            bottom_border = bottom
        if right_border < right:
            right_border = right
    return (
        top_border - padding,
        left_border - padding,
        bottom_border + padding,
        right_border + padding,
    )