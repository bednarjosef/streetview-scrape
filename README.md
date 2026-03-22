# streetview scrape
This tool was written to simplify and speed up scraping large quantities of streetview images. Thanks to parallelization it can download 100s of thousands of images from pre-defined locations in a few hours (under datacenter-tier download speed conditions).
It was used to generate the dataset that [Neuroguessr](https://github.com/bednarjosef/neuroguessr) was trained on.

# How to use
1. Simply create a list of locations automatically using [map-degen](https://map-degen.vercel.app/) or [Vali](https://github.com/slashP/Vali). Vali is recommemded as it allows for more even distributions but is not as easy to use as map-degen.
2. Configure and run the scrape files from either the vali/ or map-degen/ directory. This will start the scraping.
3. If you desire upload your dataset to HuggingFace for fast downloading to any machine and cheap storage.
