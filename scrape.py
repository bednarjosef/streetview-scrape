import os, json, time, asyncio
from streetlevel import streetview
import numpy as np
import cv2
from PIL import Image
import aiohttp
from asyncio import Semaphore


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


def save_four_views_from_pano(pano_img, out_dir, base_name="view", size=512):
    img_bgr = cv2.cvtColor(np.array(pano_img), cv2.COLOR_RGB2BGR)

    yaws = [
        (180, f"{out_dir}/{base_name}_1.jpg"),
        (-90, f"{out_dir}/{base_name}_2.jpg"),
        (0,   f"{out_dir}/{base_name}_3.jpg"),
        (90,  f"{out_dir}/{base_name}_4.jpg"),
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


def load_from_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
        loc_list = data['customCoordinates']
        return loc_list


# panoid = 'gjPHUx9jDrEh5EnXjGrmvw'
# pano = streetview.find_panorama_by_id(panoid)
# pano_img = streetview.get_panorama(pano, zoom=2)
# save_four_views_from_pano(pano_img)

async def scrape_og(locations):
    total = len(locations)
    ts = time.time()
    for idx, location in enumerate(locations):
        print(f'Scraping location {idx+1}/{total}...')
        panoid = location['panoId']
        pano = await streetview.find_panorama_by_id_async(panoid)
        # pano.country_code
        # pano.date
        # pano.elevation
        # pano.lat
        # pano.lon
        pano_img = await streetview.get_panorama_async(pano, zoom=2)
        save_four_views_from_pano(pano_img, out_dir=out_dir, base_name=panoid)

    te = time.time()
    td = round(te-ts, 2)
    lps = round(total / td, 2)
    spl = round(1 / lps, 2)
    print(f'Finished scraping {total} locations in {td} seconds - {lps} locations/second - {spl} seconds/location')


async def scrape_one(location, session, sem, out_dir):
    panoid = location['panoId']
    async with sem:  # limit concurrency
        try:
            pano = await streetview.find_panorama_by_id_async(panoid, session=session)
            pano_img = await streetview.get_panorama_async(pano, zoom=2, session=session)
            # save_four_views_from_pano(pano_img, out_dir=out_dir, base_name=panoid)
            return True
        except Exception as e:
            print(f"Error for {panoid}: {e}")
            return False
        

async def scrape(locations, out_dir, max_concurrency=8):
    total = len(locations)
    ts = time.time()
    sem = Semaphore(max_concurrency)

    connector = aiohttp.TCPConnector(limit=max_concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(scrape_one(loc, session, sem, out_dir))
            for loc in locations
        ]
        done = 0
        success = 0
        for coro in asyncio.as_completed(tasks):
            ok = await coro
            done += 1
            if ok:
                success += 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - ts
                lps = done / elapsed
                print(f"[{done}/{total}] {lps:.2f} locations/s ({success} ok)")

    te = time.time()
    td = round(te - ts, 2)
    lps = round(total / td, 2)
    spl = round(1 / lps, 2)
    print(f"Finished scraping {total} locations in {td} s "
          f"- {lps} locations/s - {spl} s/location")

if __name__ == '__main__':
    out_dir = 'panoramas_2'
    os.makedirs(out_dir, exist_ok=True)
    locations = load_from_json('Czechia (100 locations).json')
    asyncio.run(scrape(locations, out_dir, max_concurrency=16))


# if __name__ == '__main__':
#     out_dir = 'panoramas_2'
#     os.makedirs(out_dir, exist_ok=True)
#     locations = load_from_json('Czechia (100 locations).json')
#     asyncio.run(scrape(locations))
