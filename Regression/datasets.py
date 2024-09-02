import tensorflow_datasets as tfds

# List of datasets to download
datasets = ['imdb_reviews/subwords8k']

for dataset in datasets:
    print(f"Downloading {dataset}...")
    dataset, info = tfds.load(dataset, with_info=True, as_supervised=True)
