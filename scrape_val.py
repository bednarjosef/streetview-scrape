import os, json, time, asyncio, random
from streetlevel import streetview
import numpy as np
import cv2
from PIL import Image
import aiohttp
from asyncio import Semaphore

PANOS_PER_SHARD = 10_000 

def equirect_to_perspective(img_bgr, fov_deg, yaw_deg, pitch_deg, out_size):
    w_out, h_out = out_size
    fov = np.deg2rad(fov_deg)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    f = 0.5 * w_out / np.tan(fov / 2)

    x = np.linspace(-w_out / 2, w_out / 2, w_out)
    y = np.linspace(-h_out / 2, h_out / 2, h_out)
    xx, yy = np.meshgrid(x, y)
    zz = np.full_like(xx, f)

    norm = np.sqrt(xx**2 + yy**2 + zz**2)
    xx = xx / norm
    yy = -yy / norm
    zz = zz / norm

    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)

    Ry = np.array([[ cos_y, 0, sin_y],
                   [     0, 1,     0],
                   [-sin_y, 0, cos_y]])

    Rx = np.array([[1,     0,      0],
                   [0, cos_p, -sin_p],
                   [0, sin_p,  cos_p]])

    R = Ry @ Rx

    dirs = np.stack([xx, yy, zz], axis=-1)
    dirs = dirs @ R.T

    lon = np.arctan2(dirs[..., 0], dirs[..., 2])
    lat = np.arcsin(dirs[..., 1])

    h_eq, w_eq, _ = img_bgr.shape
    x_map = (lon + np.pi) / (2 * np.pi) * w_eq
    y_map = (np.pi / 2 - lat) / np.pi * h_eq

    x_map = x_map.astype(np.float32)
    y_map = y_map.astype(np.float32)

    out = cv2.remap(img_bgr, x_map, y_map,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_WRAP)
    return out


def save_random_view_from_pano(pano_img, out_dir, base_name, size=512):
    """
    Picks ONE random view (0, 90, 180, or -90) and saves it.
    Returns the relative filename used.
    """
    img_bgr = cv2.cvtColor(np.array(pano_img), cv2.COLOR_RGB2BGR)

    # Definitions: (Yaw, Suffix)
    # We keep the suffix consistent so we know which direction it is later
    options = [
        (180, "_1.jpg"), # Back
        (-90, "_2.jpg"), # Left
        (0,   "_3.jpg"), # Front
        (90,  "_4.jpg"), # Right
    ]
    
    # Pick one random tuple from the list
    yaw, suffix = random.choice(options)
    
    filename = f"{base_name}{suffix}"
    full_path = os.path.join(out_dir, filename)

    persp = equirect_to_perspective(
        img_bgr,
        fov_deg=90,
        yaw_deg=yaw,
        pitch_deg=0,
        out_size=(size, size),
    )
    cv2.imwrite(full_path, persp)
    
    # Return just the filename so we can log it
    return filename


def load_from_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
        loc_list = data['customCoordinates']
        return loc_list


async def scrape_one(idx, location, session, sem, root_dir):
    panoid = location['panoId']

    shard_idx = idx // PANOS_PER_SHARD
    shard_dir = f"{shard_idx:06d}"

    images_root = os.path.join(root_dir, "images")
    img_dir = os.path.join(images_root, shard_dir)
    os.makedirs(img_dir, exist_ok=True)

    try:
        # 1) NETWORK-BOUND
        async with sem:
            pano = await streetview.find_panorama_by_id_async(panoid, session=session)
            pano_img = await streetview.get_panorama_async(pano, zoom=2, session=session)

        # 2) CPU + DISK-BOUND (Save ONLY ONE random view)
        loop = asyncio.get_running_loop()
        saved_filename = await loop.run_in_executor(
            None,
            save_random_view_from_pano,
            pano_img,
            img_dir,
            panoid,
        )

        # Build the single relative path
        rel_view = f"images/{shard_dir}/{saved_filename}"

        date_value = str(pano.date)

        meta = {
            "shard_idx": shard_idx,
            "panoid": panoid,
            "views": [rel_view], # List containing just the one view
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

    meta_files = {}

    def get_meta_file(shard_idx: int):
        if shard_idx not in meta_files:
            path = os.path.join(meta_root, f"panos-{shard_idx:06d}.jsonl")
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

    for f in meta_files.values():
        f.close()

    te = time.time()
    td = round(te - ts, 2)
    print(f"Finished scraping {total} locations in {td} s")


if __name__ == '__main__':
    # --- CHANGED CONFIG ---
    out_dir = 'streetview_world_val'
    os.makedirs(out_dir, exist_ok=True)
    locations = load_from_json('world_3k_val.json')
    # ----------------------
    asyncio.run(scrape(locations, out_dir, max_concurrency=32))