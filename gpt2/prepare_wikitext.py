import datasets

datasets.load_dataset(
        "wikitext", 
        "wikitext-103-v1",
        cache_dir="/mnt/disk2/wikitext_cache",
        )

