import cv2, json, random, os
import numpy as np


def load_from_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
        loc_list = data['customCoordinates']
        return loc_list
    

# streetview images are saved with weird focals, this fixes to normal image views
def equirect_to_perspective(img_bgr, fov_deg, yaw_deg, pitch_deg, out_size):    
    w_out, h_out = out_size
    fov = np.deg2rad(fov_deg)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    # focal length from FOV
    f = 0.5 * w_out / np.tan(fov / 2)

    # pixel grid in image (screen) coordinates
    x = np.linspace(-w_out / 2, w_out / 2, w_out)
    y = np.linspace(-h_out / 2, h_out / 2, h_out)
    xx, yy = np.meshgrid(x, y)
    zz = np.full_like(xx, f)

    # normalize direction vectors
    norm = np.sqrt(xx**2 + yy**2 + zz**2)

    xx = xx / norm
    yy = -yy / norm
    zz = zz / norm

    # rotation matrices (yaw then pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)

    Ry = np.array([[ cos_y, 0, sin_y],
                   [     0, 1,     0],
                   [-sin_y, 0, cos_y]])

    Rx = np.array([[1,     0,      0],
                   [0, cos_p, -sin_p],
                   [0, sin_p,  cos_p]])

    R = Ry @ Rx

    dirs = np.stack([xx, yy, zz], axis=-1)    # (h, w, 3)
    dirs = dirs @ R.T                         # rotate

    # convert directions to spherical coordinates
    lon = np.arctan2(dirs[..., 0], dirs[..., 2])   # [-pi, pi]
    lat = np.arcsin(dirs[..., 1])                  # [-pi/2, pi/2]

    # map to equirectangular pixel coords
    h_eq, w_eq, _ = img_bgr.shape
    x_map = (lon + np.pi) / (2 * np.pi) * w_eq
    y_map = (np.pi / 2 - lat) / np.pi * h_eq

    x_map = x_map.astype(np.float32)
    y_map = y_map.astype(np.float32)

    out = cv2.remap(img_bgr, x_map, y_map,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_WRAP)
    return out


def save_four_views_from_pano(pano_img, out_dir, base_name='view', size=512):
    img_bgr = cv2.cvtColor(np.array(pano_img), cv2.COLOR_RGB2BGR)

    yaws = [
        (180, f'{out_dir}/{base_name}_1.jpg'),  # back
        (-90, f'{out_dir}/{base_name}_2.jpg'),  # left
        (0,   f'{out_dir}/{base_name}_3.jpg'),  # front
        (90,  f'{out_dir}/{base_name}_4.jpg'),  # right
    ]

    for yaw, fname in yaws:
        persp = equirect_to_perspective(
            img_bgr,
            fov_deg=90,
            yaw_deg=yaw,
            pitch_deg=0,
            out_size=(size, size),
        )
        cv2.imwrite(fname, persp)


def save_random_view_from_pano(pano_img, out_dir, base_name, size=512):
    # saves 1 random of 4 views from the panorama
    img_bgr = cv2.cvtColor(np.array(pano_img), cv2.COLOR_RGB2BGR)

    options = [
        (180, '_1.jpg'),
        (-90, '_2.jpg'),
        (0,   '_3.jpg'),
        (90,  '_4.jpg'),
    ]
    
    # one random tuple from list
    yaw, suffix = random.choice(options)
    
    filename = f'{base_name}{suffix}'
    full_path = os.path.join(out_dir, filename)

    persp = equirect_to_perspective(
        img_bgr,
        fov_deg=90,
        yaw_deg=yaw,
        pitch_deg=0,
        out_size=(size, size),
    )
    cv2.imwrite(full_path, persp)
    
    return filename
