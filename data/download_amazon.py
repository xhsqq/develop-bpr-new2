"""
Download Amazon Review datasets
Amazon Review Data (2018): https://nijianmo.github.io/amazon/index.html
"""

import os
import gzip
import json
import ast
import argparse
from pathlib import Path
from tqdm import tqdm
import urllib.request


# Amazon 2018 dataset URLs
AMAZON_URLS = {
    'beauty': {
        'reviews': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz',
        'metadata': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz'
    },
    'games': {
        'reviews': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz',
        'metadata': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Video_Games.json.gz'
    },
    'sports': {
        'reviews': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz',
        'metadata': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Sports_and_Outdoors.json.gz'
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str):
    """Download a file with progress bar"""
    print(f"Downloading {url} to {output_path}...")

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

    print(f"✓ Downloaded to {output_path}")


def parse_json_gz(file_path: str, output_path: str):
    """Parse and convert gzipped JSON to plain JSON
    
    Note: Amazon metadata files use Python dict format (single quotes),
    while review files use standard JSON format (double quotes).
    This function handles both formats.
    """
    print(f"Parsing {file_path}...")

    data = []
    parse_errors = 0
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading lines"):
            line = line.strip()
            if not line:
                continue
            
            try:
                # 首先尝试标准 JSON 格式
                data.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    # 如果失败，尝试 Python 字典格式（使用 ast.literal_eval）
                    data.append(ast.literal_eval(line))
                except (ValueError, SyntaxError):
                    parse_errors += 1
                    continue

    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"✓ Parsed {len(data)} records to {output_path}")
    if parse_errors > 0:
        print(f"  ⚠ Skipped {parse_errors} lines due to parse errors")
    return len(data)


def download_amazon_dataset(category: str, data_dir: str = 'data/raw'):
    """
    Download Amazon dataset for a specific category

    Args:
        category: One of 'beauty', 'games', 'sports'
        data_dir: Directory to save raw data
    """
    if category not in AMAZON_URLS:
        raise ValueError(f"Unknown category: {category}. Choose from {list(AMAZON_URLS.keys())}")

    os.makedirs(data_dir, exist_ok=True)

    urls = AMAZON_URLS[category]

    # Download reviews
    reviews_gz = os.path.join(data_dir, f'{category}_reviews.json.gz')
    reviews_json = os.path.join(data_dir, f'{category}_reviews.json')

    if not os.path.exists(reviews_json):
        if not os.path.exists(reviews_gz):
            download_file(urls['reviews'], reviews_gz)
        parse_json_gz(reviews_gz, reviews_json)
        os.remove(reviews_gz)  # Clean up gz file
    else:
        print(f"✓ Reviews already exist at {reviews_json}")

    # Download metadata
    meta_gz = os.path.join(data_dir, f'{category}_meta.json.gz')
    meta_json = os.path.join(data_dir, f'{category}_meta.json')

    if not os.path.exists(meta_json):
        if not os.path.exists(meta_gz):
            download_file(urls['metadata'], meta_gz)
        parse_json_gz(meta_gz, meta_json)
        os.remove(meta_gz)  # Clean up gz file
    else:
        print(f"✓ Metadata already exist at {meta_json}")

    print(f"\n✓ Successfully downloaded {category} dataset!")


def download_all_datasets(data_dir: str = 'data/raw'):
    """Download all three datasets"""
    categories = ['beauty', 'games', 'sports']

    print("=" * 80)
    print("Downloading Amazon Datasets")
    print("=" * 80)

    for i, category in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] Processing {category.upper()} dataset...")
        download_amazon_dataset(category, data_dir)

    print("\n" + "=" * 80)
    print("✓ All datasets downloaded successfully!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Download Amazon Review datasets')
    parser.add_argument(
        '--category',
        type=str,
        choices=['beauty', 'games', 'sports', 'all'],
        default='all',
        help='Dataset category to download'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/raw',
        help='Directory to save raw data'
    )

    args = parser.parse_args()

    if args.category == 'all':
        download_all_datasets(args.data_dir)
    else:
        download_amazon_dataset(args.category, args.data_dir)


if __name__ == '__main__':
    main()
