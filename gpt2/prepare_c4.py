import datasets

#data_files = {"validation": "en/*validation*"}

data_files = {
        "train": "en/*train*",
        "validation": "en/*validation*",
        }

ds = datasets.load_dataset(
        "allenai/c4", 
        cache_dir="/mnt/disk2/c4_cache",
        data_files=data_files,
        #split="validation",
        )

print(ds)

