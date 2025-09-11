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
    elif fig_type == 'rectangle':
        _generate_square(bg, center, base_size, color_fig)
    elif fig_type == 'triangle':
        center = (
            random.randint(int(np.sqrt(3) * base_size), int(w_bg - np.sqrt(3) * base_size)),
            random.randint(base_size * 2, h_bg - base_size)
        )
        _generate_triangle(bg, center, base_size, color_fig)

    return bg

def _generate_circle(bg, center, radius, color_circle):
    cv2.circle(bg, center, radius, color_circle, thickness=-1)
    return bg

def _generate_square(bg, center, apothem, color_square):
    cx, cy = center
    cv2.rectangle(
        bg,
        (cx - apothem, cy - apothem),
        (cx + apothem, cy + apothem),
        color_square,
        thickness=-1
    )
    return bg

def _generate_triangle(bg, center, apothem, color_triangle):
    cx, cy = center
    pts = [
        (cx - int(np.sqrt(3) * apothem), cy + apothem),
        (cx, cy - 2 * apothem),
        (cx + int(np.sqrt(3) * apothem), cy + apothem)
    ]

    pts = np.array([pts], dtype=np.int32)

    cv2.fillPoly(bg, [pts], color_triangle)
    return bg

def visualize(bg, time=0, name='Figure'):
    cv2.imshow(name, bg)
    cv2.waitKey(time)
    cv2.destroyAllWindows()


visualize(generate_object('triangle',640, 640), time=1000)