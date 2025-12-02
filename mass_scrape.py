import os, json, time, asyncio
from streetlevel import streetview
import numpy as np
import cv2
from PIL import Image
import aiohttp
from asyncio import Semaphore

PANOS_PER_SHARD = 10_000  # 10k panos per shard/folder


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


async def scrape_og(locations):
    # your old sequential version (kept for reference)
    total = len(locations)
    ts = time.time()
    for idx, location in enumerate(locations):
        print(f'Scraping location {idx+1}/{total}...')
        panoid = location['panoId']
        pano = await streetview.find_panorama_by_id_async(panoid)
        pano_img = await streetview.get_panorama_async(pano, zoom=2)
        save_four_views_from_pano(pano_img, out_dir=out_dir, base_name=panoid)

    te = time.time()
    td = round(te-ts, 2)
    lps = round(total / td, 2)
    spl = round(1 / lps, 2)
    print(f'Finished scraping {total} locations in {td} seconds - {lps} locations/second - {spl} seconds/location')


async def scrape_one(idx, location, session, sem, root_dir):
    """
    Scrape a single panorama, save its 4 views into the correct shard folder,
    and return a metadata dict (or None on error).
    """
    panoid = location['panoId']

    # determine shard/folder for this pano
    shard_idx = idx // PANOS_PER_SHARD
    shard_dir = f"{shard_idx:06d}"

    images_root = os.path.join(root_dir, "images")
    img_dir = os.path.join(images_root, shard_dir)
    os.makedirs(img_dir, exist_ok=True)

    async with sem:
        try:
            pano = await streetview.find_panorama_by_id_async(panoid, session=session)
            pano_img = await streetview.get_panorama_async(pano, zoom=2, session=session)

            # save the 4 views into this shard folder
            save_four_views_from_pano(pano_img, out_dir=img_dir, base_name=panoid)

            # relative paths for metadata
            rel_views = [
                f"images/{shard_dir}/{panoid}_1.jpg",
                f"images/{shard_dir}/{panoid}_2.jpg",
                f"images/{shard_dir}/{panoid}_3.jpg",
                f"images/{shard_dir}/{panoid}_4.jpg",
            ]

            # pano.date is a CaptureDate -> make it JSON-serializable
            date_value = str(pano.date)

            meta = {
                "shard_idx": shard_idx,      # internal, for choosing jsonl file
                "panoid": panoid,
                "views": rel_views,
                "country_code": pano.country_code,
                "date": date_value,
                "elevation": pano.elevation,
                "lat": pano.lat,
                "lon": pano.lon,
            }
            return meta

        except Exception as e:
            print(f"Error for {panoid}: {e}")
            return None
        

async def scrape(locations, out_dir, max_concurrency=8):
    total = len(locations)
    ts = time.time()
    sem = Semaphore(max_concurrency)

    images_root = os.path.join(out_dir, "images")
    meta_root = os.path.join(out_dir, "metadata")
    os.makedirs(images_root, exist_ok=True)
    os.makedirs(meta_root, exist_ok=True)

    # shard_idx -> open jsonl file handle
    meta_files = {}

    def get_meta_file(shard_idx: int):
        """Open (or reuse) a JSONL file for this shard."""
        if shard_idx not in meta_files:
            path = os.path.join(meta_root, f"panos-{shard_idx:06d}.jsonl")
            # line-buffered so each line is flushed quickly
            f = open(path, "a", encoding="utf-8", buffering=1)
            meta_files[shard_idx] = f
        return meta_files[shard_idx]

    connector = aiohttp.TCPConnector(limit=max_concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(scrape_one(idx, loc, session, sem, out_dir))
            for idx, loc in enumerate(locations)
        ]

        done = 0
        success = 0

        for coro in asyncio.as_completed(tasks):
            meta = await coro
            done += 1

            if meta is not None:
                success += 1
                shard_idx = meta.pop("shard_idx")
                f = get_meta_file(shard_idx)
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            if done % 100 == 0 or done == total:
                elapsed = time.time() - ts
                lps = done / elapsed
                print(f"[{done}/{total}] {lps:.2f} locations/s ({success} ok)")

    # close metadata files
    for f in meta_files.values():
        f.close()

    te = time.time()
    td = round(te - ts, 2)
    lps = round(total / td, 2)
    spl = round(1 / lps, 2)
    print(f"Finished scraping {total} locations in {td} s "
          f"- {lps} locations/s - {spl} seconds/location")


if __name__ == '__main__':
    out_dir = 'streetview_cz'
    os.makedirs(out_dir, exist_ok=True)
    locations = load_from_json('Custom polygon 1 (100000 locations).json')
    asyncio.run(scrape(locations, out_dir, max_concurrency=16))
