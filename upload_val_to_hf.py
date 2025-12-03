import os
import json
import glob
from datasets import Dataset, Features, Value, Image

# --- CONFIGURATION ---
# Point this to your NEW validation folder
DATA_DIRS = ["streetview_world_val"] 

HF_REPO_ID = "josefbednar/world-streetview-500k" 
PRIVATE_DATASET = True 

def generate_examples():
    """
    Generator that handles the validation data (1 view per pano).
    """
    total_processed = 0
    
    # Mapping filename suffix to ID suffix to match training data
    # Training data logic was:
    # Index 0 -> _1.jpg (Back) -> ID suffix "0"
    # Index 1 -> _2.jpg (Left) -> ID suffix "1"
    # Index 2 -> _3.jpg (Front) -> ID suffix "2"
    # Index 3 -> _4.jpg (Right) -> ID suffix "3"
    
    suffix_map = {
        "1": "0",
        "2": "1",
        "3": "2",
        "4": "3"
    }

    for root_dir in DATA_DIRS:
        print(f"Scanning directory: {root_dir}...")
        
        meta_dir = os.path.join(root_dir, "metadata")
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
                        stored_views = record['views'] # This is now a list of 1 string

                        if not stored_views:
                            continue
                            
                        # Loop through views (only 1 now, but loop handles it fine)
                        for rel_path in stored_views:
                            
                            full_path = os.path.join(root_dir, rel_path)
                            
                            if not os.path.exists(full_path):
                                continue
                            
                            # EXTRACT SUFFIX FROM FILENAME
                            # filename is like "panoid_3.jpg"
                            filename = os.path.basename(rel_path)
                            # get the character between '_' and '.jpg'
                            file_suffix = filename.split('_')[-1].split('.')[0]
                            
                            # Get the ID suffix (0, 1, 2, or 3)
                            id_suffix = suffix_map.get(file_suffix, "0")
                            
                            unique_id = f"{panoid}_{id_suffix}"
                            
                            yield {
                                "image_id": unique_id,
                                "panoid": panoid,
                                "image": full_path,
                                "country_code": record.get("country_code"),
                                "date": record.get("date"),
                                "latitude": record.get("lat"),
                                "longitude": record.get("lon"),
                                "elevation": record.get("elevation"),
                            }
                        
                        total_processed += 1
                        if total_processed % 500 == 0:
                            print(f"Processed {total_processed} panoramas...")

                    except Exception as e:
                        print(f"Error parsing line in {jsonl_file}: {e}")

def main():
    features = Features({
        "image_id": Value("string"),
        "panoid": Value("string"),
        "image": Image(),
        "country_code": Value("string"),
        "date": Value("string"),
        "latitude": Value("float64"),
        "longitude": Value("float64"),
        "elevation": Value("float64"),
    })

    print(f"Initializing VALIDATION dataset stream from: {DATA_DIRS}")

    ds = Dataset.from_generator(
        generate_examples, 
        features=features,
        keep_in_memory=False
    )

    print("Generator ready. Starting upload to Hugging Face (VALIDATION SPLIT)...")

    ds.push_to_hub(
        HF_REPO_ID, 
        private=PRIVATE_DATASET, 
        split="validation",  # <--- THIS IS THE KEY CHANGE
        max_shard_size="500MB",
        embed_external_files=True 
    )
    
    print(f"Done! Validation split available at https://huggingface.co/datasets/{HF_REPO_ID}")

if __name__ == "__main__":
    main()