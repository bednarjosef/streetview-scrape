import os
import json
import glob
from datasets import Dataset, Features, Value, Image
from huggingface_hub import create_repo

# --- CONFIGURATION ---
# List your root folders here
DATA_DIRS = ["streetview_world_1", "streetview_world_2"]

# CHANGE THIS to your Hugging Face username and dataset name
HF_REPO_ID = "josefbednar/world-streetview-500k" 

# Set to False if you want everyone to see it
PRIVATE_DATASET = True 

def generate_examples():
    """
    Generator that reads metadata files line-by-line and yields 
    individual image rows to avoid RAM spikes.
    """
    total_processed = 0
    
    # We map the 4 images in the list to suffixes 0, 1, 2, 3
    # Your scraper saved them in this order: 180 (Back), -90 (Left), 0 (Front), 90 (Right)
    # We will just map them strictly by index as requested.
    suffixes = ["0", "1", "2", "3"]

    for root_dir in DATA_DIRS:
        print(f"Scanning directory: {root_dir}...")
        
        meta_dir = os.path.join(root_dir, "metadata")
        
        # Get all JSONL files in the metadata folder
        jsonl_files = sorted(glob.glob(os.path.join(meta_dir, "*.jsonl")))
        
        if not jsonl_files:
            print(f"Warning: No metadata files found in {meta_dir}")
            continue

        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        panoid = record['panoid']
                        
                        # The 'views' list in your JSON contains relative paths
                        # e.g., "images/000005/panoid_1.jpg"
                        stored_views = record['views']

                        # Validate we have exactly 4 images
                        if not stored_views or len(stored_views) != 4:
                            continue
                            
                        # Loop through the 4 views and yield 4 separate rows
                        for i, rel_path in enumerate(stored_views):
                            
                            # Construct absolute path: root_dir + relative_path
                            full_path = os.path.join(root_dir, rel_path)
                            
                            if not os.path.exists(full_path):
                                # Skip if image file is corrupted/missing
                                continue
                                
                            # Create the specific ID: panoid_0, panoid_1, etc.
                            unique_id = f"{panoid}_{suffixes[i]}"
                            
                            yield {
                                "image_id": unique_id,
                                "panoid": panoid,
                                "image": full_path, # HF library handles opening this path
                                # Metadata (duplicated for each view)
                                "country_code": record.get("country_code"),
                                "date": record.get("date"),
                                "latitude": record.get("lat"),
                                "longitude": record.get("lon"),
                                "elevation": record.get("elevation"),
                            }
                        
                        total_processed += 1
                        if total_processed % 10_000 == 0:
                            print(f"Processed {total_processed} panoramas ({total_processed*4} images)...")

                    except Exception as e:
                        print(f"Error parsing line in {jsonl_file}: {e}")

def main():
    # 1. Define the Schema
    features = Features({
        "image_id": Value("string"),     # e.g., "g8s7d6f_0"
        "panoid": Value("string"),       # e.g., "g8s7d6f"
        "image": Image(),                # The actual image data
        "country_code": Value("string"),
        "date": Value("string"),
        "latitude": Value("float64"),
        "longitude": Value("float64"),
        "elevation": Value("float64"),
    })

    print(f"Initializing dataset stream from: {DATA_DIRS}")

    # 2. Create Dataset Object (Lazy Loading)
    # keep_in_memory=False ensures we stream from disk -> upload
    ds = Dataset.from_generator(
        generate_examples, 
        features=features,
        keep_in_memory=False
    )

    print("Generator ready. Starting upload to Hugging Face...")
    print("This will take time. Images are being embedded into Parquet files.")

    # 3. Upload
    # max_shard_size="500MB" is good for image datasets to avoid massive single files
    ds.push_to_hub(
        HF_REPO_ID, 
        private=PRIVATE_DATASET, 
        max_shard_size="500MB",
        embed_external_files=True 
    )
    
    print(f"Done! Dataset available at https://huggingface.co/datasets/{HF_REPO_ID}")

if __name__ == "__main__":
    main()
    