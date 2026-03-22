import os, json, glob

from datasets import Dataset, Features, Value, Image

# THIS SCRIPT WORKS ON IMAGES SAVED USING map-degen/scrape_train.py

DATA_DIRS = ['streetview-prague']

# path to existing dataset
HF_REPO_ID = 'josefbednar/prague-streetview-50k' 
PRIVATE_DATASET = True 

def generate_examples():
    total_processed = 0
    suffixes = ['0', '1', '2', '3']

    for root_dir in DATA_DIRS:
        print(f'Scanning directory: {root_dir}...')
        
        meta_dir = os.path.join(root_dir, 'metadata')
        
        # get all JSONL files in the metadata folder
        jsonl_files = sorted(glob.glob(os.path.join(meta_dir, '*.jsonl')))
        
        if not jsonl_files:
            print(f'No metadata files found in {meta_dir}')
            continue

        for jsonl_file in jsonl_files:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        panoid = record['panoid']
                        
                        stored_views = record['views']

                        if not stored_views or len(stored_views) != 4:
                            continue
                            
                        for i, rel_path in enumerate(stored_views):
                            full_path = os.path.join(root_dir, rel_path)
                            
                            if not os.path.exists(full_path):
                                continue
                                
                            unique_id = f'{panoid}_{suffixes[i]}'
                            
                            yield {
                                'image_id': unique_id,
                                'panoid': panoid,
                                'image': full_path,
                                'country_code': record.get('country_code'),
                                'date': record.get('date'),
                                'latitude': record.get('lat'),
                                'longitude': record.get('lon'),
                                'elevation': record.get('elevation'),
                            }
                        
                        total_processed += 1
                        if total_processed % 10_000 == 0:
                            print(f'Processed {total_processed} panoramas ({total_processed*4} images)...')

                    except Exception as e:
                        print(f'Error parsing line in {jsonl_file}: {e}')

def main():
    features = Features({
        'image_id': Value('string'),
        'panoid': Value('string'),
        'image': Image(),
        'country_code': Value('string'),
        'date': Value('string'),
        'latitude': Value('float64'),
        'longitude': Value('float64'),
        'elevation': Value('float64'),
    })

    print(f'Initializing dataset stream from: {DATA_DIRS}')

    ds = Dataset.from_generator(
        generate_examples, 
        features=features,
        keep_in_memory=False
    )

    print('Generator ready. Starting upload to Hugging Face...')

    ds.push_to_hub(
        HF_REPO_ID, 
        private=PRIVATE_DATASET, 
        max_shard_size='500MB',
        embed_external_files=True 
    )
    
    print(f'Done! Dataset available at https://huggingface.co/datasets/{HF_REPO_ID}')

if __name__ == '__main__':
    main()
    