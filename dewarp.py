import cv2
import numpy as np

def equirect_to_perspective(img, fov, pitch, yaw, out_size):
    # Convert degrees to radians
    fov = np.deg2rad(fov)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    # Output image size
    w, h = out_size

    # Focal length
    f = 0.5 * w / np.tan(fov / 2)

    # Build a grid of pixel coordinates
    x = np.linspace(-w/2, w/2, w)
    y = np.linspace(-h/2, h/2, h)
    xx, yy = np.meshgrid(x, y)

    # Normalize pixel positions
    z = f
    norm = np.sqrt(xx**2 + yy**2 + z**2)
    xx, yy, z = xx / norm, yy / norm, z / norm

    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])

    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])

    R = Ry @ Rx

    dirs = np.stack([xx, yy, z], axis=-1)
    dirs = dirs @ R.T

    # Convert to spherical coords
    lon = np.arctan2(dirs[...,0], dirs[...,2])
    lat = np.arcsin(dirs[...,1])

    # Map to pixel coords of the equirectangular image
    eq_h, eq_w, _ = img.shape

    x_map = (lon + np.pi) / (2*np.pi) * eq_w
    y_map = (np.pi/2 - lat) / np.pi * eq_h

    # Remap
    perspective = cv2.remap(
        img, x_map.astype(np.float32), y_map.astype(np.float32),
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
    )

    return perspective

img = cv2.imread("panorama.jpeg")
output = equirect_to_perspective(img, fov=90, pitch=0, yaw=45, out_size=(800, 600))
cv2.imwrite("normal_view.jpeg", output)
