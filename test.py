from streetlevel import streetview

# pano = streetview.find_panorama(46.883958, 12.169002)
panoid = 'ZYd4couAYLM5dOgrM_i3Bw'
panoid = 'ycNAMisvDHAUKyi3SnxbXQ'
panoid = 'gjPHUx9jDrEh5EnXjGrmvw'
pano = streetview.find_panorama_by_id(panoid)
pano_img = streetview.get_panorama(panoid)
streetview.download_panorama(pano, f"{pano.id}.jpg", zoom=5)
print(pano.image_sizes[2])
print(pano.tile_size)
