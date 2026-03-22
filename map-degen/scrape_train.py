import os, json, time, asyncio, aiohttp

from streetlevel import streetview
from asyncio import Semaphore

from utils import *

# THIS SCRIPT WORKS ON LOCATIONS EXPORTED TO JSON USING https://map-degen.vercel.app/

PANOS_PER_SHARD = 10_000  # 10k panos per shard

# scrape a single panorama and save 4 views
async def scrape_one(idx, location, session, sem, root_dir):
    panoid = location['panoId']

    shard_idx = idx // PANOS_PER_SHARD
    shard_dir = f'{shard_idx:06d}'

    images_root = os.path.join(root_dir, 'images')
    img_dir = os.path.join(images_root, shard_dir)
    os.makedirs(img_dir, exist_ok=True)

    try:
        async with sem:
            pano = await streetview.find_panorama_by_id_async(panoid, session=session)
            pano_img = await streetview.get_panorama_async(pano, zoom=2, session=session)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,  # default ThreadPoolExecutor
            save_four_views_from_pano,
            pano_img,
            img_dir,
            panoid,
        )

        # relative paths for metadata
        rel_views = [
            f'images/{shard_dir}/{panoid}_1.jpg',
            f'images/{shard_dir}/{panoid}_2.jpg',
            f'images/{shard_dir}/{panoid}_3.jpg',
            f'images/{shard_dir}/{panoid}_4.jpg',
        ]

        date_value = str(pano.date)  # CaptureDate -> str

        meta = {
            'shard_idx': shard_idx,
            'panoid': panoid,
            'views': rel_views,
            'country_code': pano.country_code,
            'date': date_value,
            'elevation': pano.elevation,
            'lat': pano.lat,
            'lon': pano.lon,
        }
        return meta

    except Exception as e:
        print(f'Error for {panoid}: {e}')
        return None
        

async def scrape(locations, out_dir, max_concurrency=8):
    total = len(locations)
    ts = time.time()
    sem = Semaphore(max_concurrency)

    images_root = os.path.join(out_dir, 'images')
    meta_root = os.path.join(out_dir, 'metadata')
    os.makedirs(images_root, exist_ok=True)
    os.makedirs(meta_root, exist_ok=True)

    meta_files = {}

    def get_meta_file(shard_idx: int):
        if shard_idx not in meta_files:
            path = os.path.join(meta_root, f'panos-{shard_idx:06d}.jsonl')
            f = open(path, 'a', encoding='utf-8', buffering=1)
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
                shard_idx = meta.pop('shard_idx')
                f = get_meta_file(shard_idx)
                f.write(json.dumps(meta, ensure_ascii=False) + '\n')

            if done % 500 == 0 or done == total:
                elapsed = time.time() - ts
                lps = done / elapsed
                print(f'[{done}/{total}] {lps:.2f} locations/s ({success} ok)')

    for f in meta_files.values():
        f.close()

    te = time.time()
    td = round(te - ts, 2)
    lps = round(total / td, 2)
    spl = round(1 / lps, 2)
    print(f'Finished scraping {total} locations in {td} s - {lps} locations/s - {spl} seconds/location')


if __name__ == '__main__':
    out_dir = 'streetview-prague'
    os.makedirs(out_dir, exist_ok=True)
    locations = load_from_json('locations/prague/prague_50k.json')
    asyncio.run(scrape(locations, out_dir, max_concurrency=32))
