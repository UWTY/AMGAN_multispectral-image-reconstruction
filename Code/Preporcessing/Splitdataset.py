# -*- coding: utf-8 -*-
"""
Dataset Split Script by Region with GRIDMET Support
Author: Tong Yu (Updated by ChatGPT)
"""

import os
import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import math
import re
from collections import defaultdict

def find_region_folders(base_dir):
    region_folders = []
    base_path = Path(base_dir)
    if not base_path.is_dir():
        print(f"Error: Base directory '{base_dir}' not found.")
        return []

    # pattern = re.compile(r"^.+_Exports_\d{4}_masked$")
    pattern = re.compile(r"^.+_\d{4}_cliped$")
    for item in base_path.iterdir():
        if item.is_dir() and pattern.match(item.name):
            region_folders.append(item)

    print(f"Found {len(region_folders)} valid region folders in '{base_dir}'.")
    return region_folders

def parse_foldername(folder_name):
    # pattern = re.compile(r"^(.+)_Exports_(\d{4})_masked$")
    pattern = re.compile(r"^(.+)_(\d{4})_cliped$")
    match = pattern.match(folder_name)
    if match:
        return match.group(1), int(match.group(2))
    else:
        return None, None

def gather_all_correspondences(region_folders):
    region_dict = defaultdict(list)
    print("Gathering patch correspondence data...")
    for region_path in tqdm(region_folders, desc="Reading JSON files"):
        folder_name = region_path.name
        json_path = region_path / "patch_correspondences_gridmet.json"

        study_area, year = parse_foldername(folder_name)
        if study_area is None:
            print(f"Skipping folder {folder_name} due to parsing failure.")
            continue

        if not json_path.is_file():
            print(f"Warning: JSON file not found in {region_path}. Skipping region.")
            continue

        try:
            with open(json_path, 'r') as f:
                region_data = json.load(f)
            for entry in region_data:
                entry['region_name'] = folder_name
                entry['study_area'] = study_area
                entry['year'] = year
                entry['region_path'] = str(region_path.resolve())
                region_dict[folder_name].append(entry)
        except Exception as e:
            print(f"Error reading {json_path}: {e}")

    return region_dict

def split_indices(n, ratios, seed=None):
    if not math.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    indices = list(range(n))
    if seed is not None:
        random.seed(seed)
    random.shuffle(indices)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]

def copy_files(correspondence_list, set_name, output_base_dir):
    set_output_dir = Path(output_base_dir) / set_name
    copied_manifest = []

    patch_types = {
        's30_path': 'patches_s30',
        'l30_path': 'patches_l30',
        's1_path':  'patches_s1',
        'pl_path':  'patches_planet'
    }

    for patch_dir in patch_types.values():
        os.makedirs(set_output_dir / patch_dir, exist_ok=True)

    print(f"\nCopying {len(correspondence_list)} files to '{set_name}'...")
    for entry in tqdm(correspondence_list, desc=f"Copying {set_name}"):
        region_name = entry['region_name']
        region_path = Path(entry['region_path'])
        output_entry = entry.copy()

        for key, patch_subdir in patch_types.items():
            relative_path_str = entry.get(key)
            if relative_path_str:
                source_path = region_path / relative_path_str
                original_filename = source_path.name
                unique_filename = f"{region_name}_{original_filename}"
                destination_path = set_output_dir / patch_subdir / unique_filename

                output_entry[key] = str(Path(set_name) / patch_subdir / unique_filename)

                if source_path.is_file():
                    shutil.copy2(source_path, destination_path)
                else:
                    print(f"Warning: Source file not found: {source_path}")
                    output_entry[key] = None
            else:
                output_entry[key] = None

        copied_manifest.append(output_entry)

    manifest_path = set_output_dir / f"{set_name}_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(copied_manifest, f, indent=4)
    print(f"Saved manifest to {manifest_path}")

# --- Configuration ---
INPUT_BASE_DIR = "D:/DigAgLab/Alfalfa/satellite/data/FusionDataset/FullDataset"
OUTPUT_BASE_DIR = "D:/DigAgLab/Alfalfa/satellite/data/FusionDataset/Split2"
SPLIT_RATIOS = [0.8, 0.1, 0.1]
RANDOM_SEED = 42

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting region-wise dataset split process...")

    region_paths = find_region_folders(INPUT_BASE_DIR)
    region_patch_dict = gather_all_correspondences(region_paths)

    train_set, val_set, test_set = [], [], []

    for region, patches in region_patch_dict.items():
        # if len(patches) < 10:
        #     print(f"Skipping region {region} (too few patches: {len(patches)})")
        #     continue
        train_idx, val_idx, test_idx = split_indices(len(patches), SPLIT_RATIOS, seed=RANDOM_SEED)
        train_set.extend([patches[i] for i in train_idx])
        val_set.extend([patches[i] for i in val_idx])
        test_set.extend([patches[i] for i in test_idx])

    copy_files(train_set, "train", OUTPUT_BASE_DIR)
    copy_files(val_set, "val", OUTPUT_BASE_DIR)
    copy_files(test_set, "test", OUTPUT_BASE_DIR)

    print("\nRegion-wise dataset splitting complete.")
