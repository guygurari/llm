def load_c4(split="train"):
    """Load a split of the C4 dataset.

    Args:
        split: "train" or "validation" 

    Returns:
        The Dataset.
    """
    data_files = {
            "train": "en/*train*",
            "validation": "en/*validation*",
            }

    return datasets.load_dataset(
            "allenai/c4", 
            cache_dir="/mnt/disk2/c4_cache",
            data_files=data_files,
            split=split,
            )
