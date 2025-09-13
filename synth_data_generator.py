import numpy as np
import cv2
import random
import json

import albumentations as A

def generate_background(height=640, width=640, vis=True, color=None):
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    if color:
        bg[:] = color
    if vis:
        cv2.imshow("Background", bg)
        cv2.waitKey(0)
    return bg

def generate_object(fig_type, h_bg=640, w_bg=640, color_fig=None, color_bg=None, base_size=100, save=True):
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
        img, label = _generate_circle(bg, center, base_size, color_fig)
    elif fig_type == 'square':
        img, label = _generate_square(bg, center, base_size, color_fig)
    elif fig_type == 'triangle':
        center = (
            random.randint(int(np.sqrt(3) * base_size), int(w_bg - np.sqrt(3) * base_size)),
            random.randint(base_size * 2, h_bg - base_size)
        )
        img, label = _generate_triangle(bg, center, base_size, color_fig)

    classes = {
        'circle': 0,
        'triangle': 1,
        'square': 2
    }

    img, label['rel'] = transform_image(clean_image=img, bboxes=[label['rel']], strong=True)

    if save:
        _save_data_sample(classes[fig_type], img, label)

    return img, label

def _generate_circle(bg, center, radius, color_circle, show_bbox=True):
    cv2.circle(bg, center, radius, color_circle, thickness=-1)
    cx, cy = center

    bbox = {
        'type': 'circle',
        'abs': [
        cx - radius, cy - radius,
        cx + radius, cy + radius,
    ],
        'rel': [
        cx / bg.shape[0],
        cy / bg.shape[1],
        (2 * radius) / bg.shape[0],
        (2 * radius) / bg.shape[1]
    ],
        'base_size': radius,
        'center': center
    }

    if show_bbox:
        bg_copy = bg.copy()
        cv2.rectangle(
            bg_copy,
            (bbox['abs'][0], bbox['abs'][1]),
            (bbox['abs'][2], bbox['abs'][3]),
            (0,255,0),
            thickness=4
        )
        print(bg_copy.shape)
        visualize_image(bg_copy, time=1000)

    return bg, bbox

def _generate_square(bg, center, apothem, color_square, show_bbox=True):
    cx, cy = center
    cv2.rectangle(
        bg,
        (cx - apothem, cy - apothem),
        (cx + apothem, cy + apothem),
        color_square,
        thickness=-1
    )

    bbox = {
        'type': 'circle',
        'abs': [
        cx - apothem, cy - apothem,
        cx + apothem, cy + apothem,
    ],
        'rel': [
        cx / bg.shape[0],
        cy / bg.shape[1],
        (2 * apothem) / bg.shape[0],
        (2 * apothem) / bg.shape[1]
    ],
        'base_size': apothem,
        'center': center
    }

    if show_bbox:
        bg_copy = bg.copy()
        cv2.rectangle(
            bg_copy,
            (bbox['abs'][0], bbox['abs'][1]),
            (bbox['abs'][2], bbox['abs'][3]),
            (0,255,0),
            thickness=4
        )
        visualize_image(bg_copy, time=1000)

    return bg, bbox

def _calculate_bbox():
    pass

def _generate_triangle(bg, center, apothem, color_triangle, show_bbox=True):
    cx, cy = center
    pts = [
        (cx - int(np.sqrt(3) * apothem), cy + apothem),
        (cx, cy - 2 * apothem),
        (cx + int(np.sqrt(3) * apothem), cy + apothem)
    ]

    pts = np.array([pts], dtype=np.int32)

    cv2.fillPoly(bg, [pts], color_triangle)

    bbox = {
        'type': 'circle',
        'abs': [
            int(cx - np.sqrt(3) * apothem), int(cy - 2 * apothem),
            int(cx + np.sqrt(3) * apothem), int(cy + apothem),
        ],
        'rel': [
            cx / bg.shape[0],
            cy / bg.shape[1],
            (2 * apothem) / bg.shape[0],
            (2 * apothem) / bg.shape[1]
        ],
        'base_size': apothem,
        'center': center
    }

    if show_bbox:
        bg_copy = bg.copy()
        cv2.rectangle(
            bg_copy,
            (bbox['abs'][0], bbox['abs'][1]),
            (bbox['abs'][2], bbox['abs'][3]),
            (0, 255, 0),
            thickness=4
        )
        visualize_image(bg_copy, time=1000)

    return bg, bbox

def _save_data_sample(class_id, img, label, path='data', filename='datasample', meta=False):
    cv2.imwrite(f'{path}/images/{filename}.png', img)

    with open(f'{path}/bbox/{filename}.txt', 'w') as f:
        bbox = label['rel'][0]
        f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")

    if meta:
        with open(f'{path}/meta/{filename}.json', 'w') as f:
            json.dump(label, f, indent=2)

def visualize_image(img, time=0, name='Figure'):
    cv2.imshow(name, img)
    cv2.waitKey(time)
    cv2.destroyAllWindows()

def transform_image(clean_image, bboxes, strong=False):
    weak_aug = A.Compose([
        A.Blur(blur_limit=(3,5), p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(std_range=(0.05, 0.15), p=0.3),
    ])

    strong_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.3),  # Масштабирование
        A.Blur(blur_limit=(5, 9), p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(std_range=(0.3, 0.7), p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.8))

    if random.random() < 0.2 or strong:
        ds = strong_aug(image=clean_image, bboxes=bboxes)
    else:
        ds = weak_aug(image=clean_image, bboxes=bboxes)

    return ds['image'], ds['bboxes']

visualize_image(generate_object('triangle',640, 640)[0], time=1000)