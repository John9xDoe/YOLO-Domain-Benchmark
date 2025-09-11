import numpy as np
import cv2
import random

def generate_background(height=640, width=640, vis=True, color=None):
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    if color:
        bg[:] = color
    if vis:
        cv2.imshow("Background", bg)
        cv2.waitKey(0)
    return bg

def generate_object(fig_type, h_bg=640, w_bg=640, color_fig=None, color_bg=None, base_size=100):
    bg = generate_background(h_bg, w_bg, vis=False, color=color_bg)

    if color_fig is None:
        color_fig = (
            random.randint(0,255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

    if color_bg is None:
        bg[:] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

    center = (
        random.randint(base_size,h_bg - base_size),
        random.randint(base_size, w_bg - base_size)
    )

    if fig_type == 'circle':
        _generate_circle(bg, center, base_size, color_fig)

    return bg

def _generate_circle(bg, center, radius, color_circle):
    cv2.circle(bg, center, radius, color_circle, thickness=-1)
    return bg

def visualize(bg, time=0, name='Figure'):
    cv2.imshow(name, bg)
    cv2.waitKey(time)


visualize(generate_object('circle',640, 640), time=3000)