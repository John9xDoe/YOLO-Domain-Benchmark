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

def generate_object(h_bg=640, w_bg=640, color_fig=None, color_bg=None, radius=100):
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

    center = (random.randint(radius,h_bg - radius), random.randint(radius, w_bg - radius))
    cv2.circle(bg, center, radius, color_fig,None, None, None)
    cv2.imshow("Figure", bg)
    cv2.waitKey(0)

generate_object(640, 640)