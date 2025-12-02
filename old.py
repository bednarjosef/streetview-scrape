import time
import requests, json
from io import BytesIO
from PIL import Image


def get_gmaps_url(pano_id, heading=None):
    if heading is not None:
        return f'https://www.google.com/maps/@?api=1&map_action=pano&pano={pano_id}&heading={heading}'
    else:
        return f'https://www.google.com/maps/@?api=1&map_action=pano&pano={pano_id}'


def get_pano_urls(pano_id):
    base = "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={pano_id}&x={x}&y={y}&zoom=3&nbt=1&fover=2"

    coords = [
        (6, 1), (7, 1), (0, 1), (1, 1),  # top row
        (6, 2), (7, 2), (0, 2), (1, 2),  # bottom row
    ]

    urls = [base.format(pano_id=pano_id, x=x, y=y) for (x, y) in coords]
    return urls


def stitch_8_tiles(urls, out_path="stitched_pano.jpg"):
    """
    urls: list/tuple of 8 URLs in this order:
        [url1, url2, url3, url4, url5, url6, url7, url8]
    Layout:
        top row:    url1 url2 url3 url4
        bottom row: url5 url6 url7 url8
    Saves a single stitched image to `out_path` and returns the PIL Image.
    """
    if len(urls) != 8:
        raise ValueError("Expected exactly 8 URLs")

    # 1) download all tiles
    tiles = []
    for i, u in enumerate(urls):
        resp = requests.get(u)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        tiles.append(img)

    # assume all tiles same size
    tile_w, tile_h = tiles[0].size

    cols = 4
    rows = 2
    out_w = cols * tile_w
    out_h = rows * tile_h

    # 2) create output canvas
    stitched = Image.new("RGB", (out_w, out_h))

    # 3) paste tiles
    for idx, tile in enumerate(tiles):
        row = idx // cols     # 0 or 1
        col = idx % cols      # 0..3
        x = col * tile_w
        y = row * tile_h
        stitched.paste(tile, (x, y))

    # 4) save
    stitched.save(out_path, quality=95)
    return stitched


def load_from_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
        loc_list = data['customCoordinates']
        return loc_list


def main():
    pid = 'mqg8OZCf3gmSby02RIvdMw'
    url = get_gmaps_url(pid)
    print(url)
    urls = get_pano_urls(pid)
    stitched = stitch_8_tiles(urls, "stitched_pano.jpg")
    # locations = load_from_json('Czechia (100 locations).json')
    # total = len(locations)
    # ts = time.time()
    # for idx, location in enumerate(locations):
    #     print(f'Scraping location {idx+1}/{total}...')
    #     pid = location['panoId']
    #     urls = get_pano_urls(pid)
    #     stitched = stitch_8_tiles(urls, f"location_{idx}.jpg")
    # te = time.time()
    # td = round(te-ts, 2)
    # lps = round(total / td, 2)
    # spl = round(1 / lps, 2)
    # print(f'Finished scraping {total} locations in {td} seconds - {lps} locations/second - {spl} seconds/location')


if __name__ == '__main__':
    main()
