# -*- coding: utf-8 -*-
"""
Created on Mon May 26 12:12:30 2025

@author: Tong Yu
"""
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Optimized patch generation for satellite images with quality control
- Generates patches with overlap
- Applies unified quality control (Fmask, NDVI)
- Maintains correspondence between different sensors
- Processes GRIDMET environmental data based on temporal gaps
"""

import os
import pandas as pd
import rasterio
from rasterio.windows import Window
from datetime import datetime, timedelta
import numpy as np
import json
from tqdm import tqdm
from pyproj import Transformer
import warnings
import glob

# Band indices for quality checks (0-based)
BAND_INDICES = {
    'S30': {
        'fmask': 13,  # 14th band
        'red': 3,     # 4th band
        'nir': 7,     # 8th band
    },
    'L30': {
        'fmask': 10,  # 11th band
    },
    'S1': {},  # No quality checks for S1
    'Planet': {}  # No quality checks for Planet
}

# GRIDMET variables configuration for vegetation growth
GRIDMET_VARS = {
    'tmmx': {'description': 'Maximum temperature', 'unit': 'K', 'convert_to_celsius': True},
    'tmmn': {'description': 'Minimum temperature', 'unit': 'K', 'convert_to_celsius': True},
    'pr': {'description': 'Precipitation', 'unit': 'mm'},
    'vpd': {'description': 'Vapor pressure deficit', 'unit': 'kPa'},
    'srad': {'description': 'Solar radiation', 'unit': 'W/m2'},
    'rmax': {'description': 'Maximum relative humidity', 'unit': '%'},
    'rmin': {'description': 'Minimum relative humidity', 'unit': '%'},
    'etr': {'description': 'Reference evapotranspiration', 'unit': 'mm'},
}

# GDD base temperature
GDD_BASE_TEMP = 5.0  # Celsius

def identify_image_type(filename):
    """Identify image type from filename"""
    filename_lower = filename.lower()
    if 's30_' in filename_lower:
        return 'S30'
    elif 'l30_' in filename_lower:
        return 'L30'
    elif 's1_' in filename_lower:
        return 'S1'
    elif 'planet_' in filename_lower:
        return 'Planet'
    return None

def parse_dates_from_csv(row):
    """Parse dates from CSV row with proper format handling"""
    dates = {}
    
    # S30 uses mm/dd/yyyy format
    s30_date_str = str(row.get('S30_Date', ''))
    try:
        dates['S30'] = datetime.strptime(s30_date_str, "%m/%d/%Y") if s30_date_str else None
    except:
        dates['S30'] = None
    
    # Others use yyyy-mm-dd format
    for sensor in ['L30', 'S1', 'Planet']:
        dates_str = str(row.get(f'{sensor}_Dates', '[]')).strip("[]")
        dates_list = [item.strip() for item in dates_str.split(",") if item.strip()]
        try:
            # Take the last date
            dates[sensor] = datetime.strptime(dates_list[-1], "%Y-%m-%d") if dates_list else None
        except:
            dates[sensor] = None
    
    return dates

def parse_ids_from_csv(row):
    """Parse IDs from CSV row"""
    ids = {}
    
    # S30 has single ID
    ids['S30'] = str(row.get('S30_IDS', '')) if row.get('S30_IDS') else None
    
    # Others have lists
    for sensor in ['L30', 'S1', 'Planet']:
        ids_str = str(row.get(f'{sensor}_IDs', '[]')).strip("[]")
        ids_list = [item.strip().strip("'\"") for item in ids_str.split(",") if item.strip()]
        ids[sensor] = ids_list[-1] if ids_list else None
    
    return ids

def find_image_by_id(image_id, image_files):
    """Find image file path by ID"""
    if not image_id:
        return None
    
    for img_path in image_files:
        if image_id in os.path.basename(img_path):
            return img_path
    return None

def apply_quality_masks(data, image_type, unified_mask):
    """Apply quality checks and update unified mask"""
    if data is None or image_type not in BAND_INDICES:
        return
    
    config = BAND_INDICES[image_type]
    
    # Fmask check (S30 and L30 only)
    if 'fmask' in config and data.shape[0] > config['fmask']:
        fmask_band = data[config['fmask']]
        unified_mask[fmask_band != 64] = False
    
    # NDVI check for S30
    if image_type == 'S30' and 'red' in config and 'nir' in config:
        red = data[config['red']]
        nir = data[config['nir']]
        
        # Calculate NDVI where both bands are valid
        valid_mask = (red != 0) & (nir != 0) & np.isfinite(red) & np.isfinite(nir)
        ndvi = np.zeros_like(red, dtype=np.float32)
        
        denominator = nir + red
        ndvi_valid = valid_mask & (denominator != 0)
        ndvi[ndvi_valid] = (nir[ndvi_valid] - red[ndvi_valid]) / denominator[ndvi_valid]
        
        # Mark pixels with NDVI < 0.5 as invalid
        unified_mask[ndvi < 0.5] = False

def is_valid_patch(patch, zero_threshold=75.0):
    """Check if patch is valid based on zero percentage"""
    if patch is None or patch.size == 0:
        return False
    
    zero_count = np.sum(patch == 0)
    total_pixels = patch.size
    
    if total_pixels == 0:
        return False
    
    zero_percentage = (zero_count / total_pixels) * 100
    
    return zero_percentage <= zero_threshold

def find_gridmet_files(gridmet_folder, region_name, start_date, end_date):
    """Find GRIDMET files for a date range"""
    files = []
    
    # Generate date range
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y%m%d')
        
        # Look for GRIDMET file pattern: SA2_20220515_gridmet.tif
        patterns = [
            f"{region_name}_{date_str}_gridmet.tif",
            f"{region_name}_{date_str}.tif",
            f"gridmet_{date_str}.tif"
        ]
        
        for pattern in patterns:
            filepath = os.path.join(gridmet_folder, pattern)
            if os.path.exists(filepath):
                files.append((current_date, filepath))
                break
        else:
            # Try glob pattern if exact match not found
            glob_patterns = [
                f"*{date_str}*gridmet*.tif",
                f"*{date_str}*.tif"
            ]
            for pattern in glob_patterns:
                matching_files = glob.glob(os.path.join(gridmet_folder, pattern))
                if matching_files:
                    files.append((current_date, matching_files[0]))
                    break
        
        current_date += timedelta(days=1)
    
    return files

def process_gridmet_for_patch(gridmet_folder, region_name, sensor_date, s30_date, 
                             x_offset, y_offset, patch_size=128, s30_transform=None, s30_crs=None):
    """Process GRIDMET data for a patch based on temporal gap
    
    Returns averaged values instead of arrays due to resolution difference (4km)
    """
    
    # Calculate date range (exclusive of S30 date)
    if sensor_date >= s30_date:
        # Sensor is same day or after S30, no gap
        return None, "no_gap"
    
    start_date = sensor_date
    end_date = s30_date - timedelta(days=1)  # Exclusive of S30 date
    
    # Find GRIDMET files
    gridmet_files = find_gridmet_files(gridmet_folder, region_name, start_date, end_date)
    
    if not gridmet_files:
        return None, f"No GRIDMET files found for {start_date} to {end_date}"
    
    # Calculate the bounds of the 30m patch in projected coordinates
    if s30_transform and s30_crs:
        # Get patch bounds in 30m projection
        left_30m = x_offset
        top_30m = y_offset
        right_30m = x_offset + patch_size
        bottom_30m = y_offset + patch_size
        
        # Convert to world coordinates in 30m projection
        west_30m, north_30m = s30_transform * (left_30m, top_30m)
        east_30m, south_30m = s30_transform * (right_30m, bottom_30m)
        
        patch_bounds_30m = {
            'west': west_30m,
            'east': east_30m,
            'north': north_30m,
            'south': south_30m
        }
    else:
        return None, "missing_transform_info"
    
    # Initialize accumulation dictionaries
    accumulated = {var: [] for var in GRIDMET_VARS.keys()}
    accumulated['gdd'] = []
    accumulated['avg_humidity'] = []  # Average of rmax and rmin
    
    # Process files
    num_days = 0
    gridmet_pixels_used = set()
    
    # Get band mapping from first file to understand structure
    if gridmet_files:
        with rasterio.open(gridmet_files[0][1]) as src:
            num_bands = src.count
            band_descriptions = src.descriptions
            
            # Try to auto-detect band mapping from descriptions
            band_mapping = {}
            if band_descriptions:
                for i, desc in enumerate(band_descriptions):
                    if desc:
                        desc_lower = desc.lower()
                        # Map common variations
                        if 'precip' in desc_lower or 'pr' in desc_lower:
                            band_mapping['pr'] = i
                        elif 'tmax' in desc_lower or 'tmmx' in desc_lower:
                            band_mapping['tmmx'] = i
                        elif 'tmin' in desc_lower or 'tmmn' in desc_lower:
                            band_mapping['tmmn'] = i
                        elif 'vpd' in desc_lower:
                            band_mapping['vpd'] = i
                        elif 'srad' in desc_lower or 'solar' in desc_lower:
                            band_mapping['srad'] = i
                        elif 'rmax' in desc_lower:
                            band_mapping['rmax'] = i
                        elif 'rmin' in desc_lower:
                            band_mapping['rmin'] = i
                        elif 'etr' in desc_lower or 'eto' in desc_lower:
                            band_mapping['etr'] = i
            
            # Fallback to default mapping if auto-detection fails
            if not band_mapping:
                band_mapping = {
                    'pr': 0, 'rmax': 1, 'rmin': 2, 'srad': 4,
                    'tmmx': 7, 'tmmn': 6, 'vpd': 15, 'etr': 14
                }
    
    for date, filepath in gridmet_files:
        # -------------------- 使用 patch 中心对应的 GRIDMET 像素 ---------------------
        try:
            with rasterio.open(filepath) as src:
                # 获取必要的变换器和元数据
                gridmet_crs = src.crs
                gridmet_transform = src.transform
                data = src.read()  # shape: (bands, height, width)
        
                # transformer: 30m CRS → GRIDMET CRS
                transformer_to_gridmet = Transformer.from_crs(s30_crs, gridmet_crs, always_xy=True)
        
                # 计算 patch 中心（以 projected 坐标表示）
                center_x = (patch_bounds_30m['west'] + patch_bounds_30m['east']) / 2.0
                center_y = (patch_bounds_30m['south'] + patch_bounds_30m['north']) / 2.0
        
                # 转换为 GRIDMET 坐标系
                center_gm_x, center_gm_y = transformer_to_gridmet.transform(center_x, center_y)
        
                # 计算 GRIDMET 中的行列号
                col = int((center_gm_x - gridmet_transform.c) / gridmet_transform.a)
                row = int((center_gm_y - gridmet_transform.f) / gridmet_transform.e)
        
                # 检查是否在 GRIDMET 数据内
                if 0 <= row < src.height and 0 <= col < src.width:
                    daily_values = {}
                    for var, band_idx in band_mapping.items():
                        if var in GRIDMET_VARS and band_idx < data.shape[0]:
                            val = data[band_idx, row, col]
                            if not np.isnan(val):
                                # 转换温度单位（K → ℃）
                                if var in ['tmmx', 'tmmn'] and GRIDMET_VARS[var].get('convert_to_celsius'):
                                    val -= 273.15
                                daily_values[var] = val
        
                    # GDD
                    if 'tmmx' in daily_values and 'tmmn' in daily_values:
                        tavg = (daily_values['tmmx'] + daily_values['tmmn']) / 2.0
                        daily_values['gdd'] = max(0, tavg - GDD_BASE_TEMP)
        
                    # 平均湿度
                    if 'rmax' in daily_values and 'rmin' in daily_values:
                        daily_values['avg_humidity'] = (daily_values['rmax'] + daily_values['rmin']) / 2.0
        
                    # 累积保存
                    for var, value in daily_values.items():
                        if not np.isnan(value):
                            accumulated[var].append(value)
        
                    num_days += 1
                    gridmet_pixels_used.add("1x1_center")
        
        except Exception as e:
            print(f"Error processing GRIDMET file {filepath}: {e}")

    # Calculate final aggregated values
    result = {}
    
    # Cumulative variables
    for var in ['pr', 'vpd', 'srad', 'gdd', 'etr']:
        if var in accumulated and accumulated[var]:
            result[f"{var}_sum"] = float(np.sum(accumulated[var]))
        else:
            result[f"{var}_sum"] = None
    
    # Average variables
    # for var in ['tmmx', 'tmmn', 'rmax', 'rmin', 'avg_humidity']:
    for var in ['avg_humidity']:   
        if var in accumulated and accumulated[var]:
            result[f"{var}_avg"] = float(np.mean(accumulated[var]))
        else:
            result[f"{var}_avg"] = None
    
    # Add metadata
    result['num_days'] = num_days
    result['gridmet_pixels'] = ','.join(sorted(gridmet_pixels_used)) if gridmet_pixels_used else "none"
    
    return result, None

def create_patches_with_overlap(image_path, patch_size=128, overlap=0.5):
    """Generate patches with overlap from an image"""
    try:
        with rasterio.open(image_path) as src:
            height = src.height
            width = src.width
            step_size = int(patch_size * (1 - overlap))
            
            if height < patch_size or width < patch_size:
                print(f"Warning: Image {os.path.basename(image_path)} ({width}x{height}) "
                      f"is smaller than patch size ({patch_size}x{patch_size}). Skipping.")
                return
            
            for y in range(0, height - patch_size + 1, step_size):
                for x in range(0, width - patch_size + 1, step_size):
                    window = Window(x, y, patch_size, patch_size)
                    try:
                        patch = src.read(window=window)
                        if patch is not None and patch.size > 0:
                            yield patch, (x, y)
                    except Exception as e:
                        print(f"Error reading window {window} from {image_path}: {e}")
                        continue
    
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return

def process_region(region_folder, gridmet_dir, patch_size=128, overlap=0.5, zero_threshold=75.0):
    """Process a single region folder to generate patches"""
    
    # Extract region name (e.g., SA2 from SA2_Exports_2022_masked)
    folder_parts = os.path.basename(region_folder).split('_')
    region_name = folder_parts[0]
    
    # Extract year from folder name
    year = None
    for part in folder_parts:
        if part.isdigit() and len(part) == 4:
            year = part
            break
    if not year:
        year = '2022'  # Default year
    
    # Find GRIDMET folder
    gridmet_folder = os.path.join(gridmet_dir, f"GRIDMET_{region_name}_{year}")
    
    if not os.path.exists(gridmet_folder):
        print(f"Warning: GRIDMET folder not found: {gridmet_folder}")
        gridmet_folder = None
    
    # Find CSV file
    csv_file = None
    for filename in os.listdir(region_folder):
        if filename.endswith('.csv'):
            csv_file = os.path.join(region_folder, filename)
            break
    
    if not csv_file:
        print(f"No CSV file found in {region_folder}")
        return
    
    print(f"Using CSV: {csv_file}")
    
    # Read CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Scan for image files
    image_files = {'S30': [], 'L30': [], 'S1': [], 'Planet': []}
    for filename in os.listdir(region_folder):
        if filename.lower().endswith(('.tif', '.tiff')):
            full_path = os.path.join(region_folder, filename)
            img_type = identify_image_type(filename)
            if img_type:
                image_files[img_type].append(full_path)
    
    print(f"Found images - S30: {len(image_files['S30'])}, L30: {len(image_files['L30'])}, "
          f"S1: {len(image_files['S1'])}, Planet: {len(image_files['Planet'])}")
    
    # Create output directories
    output_dirs = {}
    
    parts = region_folder.split("/")
    outdir = "/".join(parts[:-1]) + "/"+folder_parts[0]+"_"+folder_parts[2]+"_cliped"
    for sensor in ['S30', 'L30', 'S1', 'Planet']:
        output_dirs[sensor] = os.path.join(outdir, f"patches_{sensor.lower()}_30")        
        os.makedirs(output_dirs[sensor], exist_ok=True)
    
    # Process each row
    patch_correspondences = []
    processed_s30_paths = set()
    
    # Get CRS from first S30 image for coordinate transformation
    transformer = None
    if image_files['S30']:
        with rasterio.open(image_files['S30'][0]) as src:
            if src.crs:
                try:
                    transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                except:
                    print("Warning: Could not create coordinate transformer")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        # Parse dates and IDs
        dates = parse_dates_from_csv(row)
        ids = parse_ids_from_csv(row)
        
        # Find corresponding images
        image_paths = {}
        for sensor in ['S30', 'L30', 'S1', 'Planet']:
            image_paths[sensor] = find_image_by_id(ids[sensor], image_files[sensor])
        
        # Skip if no S30 image
        if not image_paths['S30']:
            continue
        
        if image_paths['S30'] not in processed_s30_paths:
            processed_s30_paths.add(image_paths['S30'])
            tqdm.write(f"Processing: S30={os.path.basename(image_paths['S30'])}, "
                      f"L30={os.path.basename(image_paths['L30']) if image_paths['L30'] else 'None'}, "
                      f"S1={os.path.basename(image_paths['S1']) if image_paths['S1'] else 'None'}, "
                      f"Planet={os.path.basename(image_paths['Planet']) if image_paths['Planet'] else 'None'}")
        
        # Open all source images
        sources = {}
        transforms = {}
        metas = {}
        descriptions = {}
        
        try:
            for sensor in ['S30', 'L30', 'S1', 'Planet']:
                if image_paths[sensor]:
                    sources[sensor] = rasterio.open(image_paths[sensor])
                    transforms[sensor] = sources[sensor].transform
                    metas[sensor] = sources[sensor].meta.copy()
                    descriptions[sensor] = sources[sensor].descriptions
            
            # Generate patches from S30
            step_size = int(patch_size * (1 - overlap))
            patch_count = 0
            saved_count = 0
            
            for s30_patch, (x_offset, y_offset) in create_patches_with_overlap(
                image_paths['S30'], patch_size, overlap):
                
                patch_count += 1
                patch_name_base = f"patch_r{index}_x{x_offset}_y{y_offset}_o{int(overlap*100)}"
                
                # Read corresponding patches from other sensors
                patches = {'S30': s30_patch}
                
                # All images are already at 30m resolution after preprocessing
                for sensor in ['L30', 'S1', 'Planet']:
                    if sensor in sources:
                        window = Window(x_offset, y_offset, patch_size, patch_size)
                        try:
                            patch_data = sources[sensor].read(window=window)
                            patches[sensor] = patch_data
                        except:
                            patches[sensor] = None
                    else:
                        patches[sensor] = None
                
                # Create unified quality mask
                unified_mask = np.ones((patch_size, patch_size), dtype=bool)
                
                # Apply quality checks for each sensor
                for sensor in ['S30', 'L30', 'S1', 'Planet']:
                    if patches[sensor] is not None:
                        apply_quality_masks(patches[sensor], sensor, unified_mask)
                
                # Apply mask to all patches
                masked_patches = {}
                for sensor, patch in patches.items():
                    if patch is not None:
                        masked_patch = patch.copy()
                        for b in range(masked_patch.shape[0]):
                            masked_patch[b][~unified_mask] = 0
                        masked_patches[sensor] = masked_patch
                    else:
                        masked_patches[sensor] = None
                
                # Check validity
                valid_patches = {}
                for sensor in ['S30', 'L30', 'S1', 'Planet']:
                    if masked_patches[sensor] is None:
                        valid_patches[sensor] = True  # Missing patches are considered valid
                    else:
                        valid_patches[sensor] = is_valid_patch(masked_patches[sensor], zero_threshold)
                
                # Save if all patches are valid
                if all(valid_patches.values()):
                    saved_count += 1
                    output_paths = {}
                    
                    # Calculate center coordinates
                    center_lon, center_lat = None, None
                    if transformer and transforms.get('S30'):
                        try:
                            center_col = x_offset + patch_size / 2.0
                            center_row = y_offset + patch_size / 2.0
                            center_x, center_y = transforms['S30'] * (center_col, center_row)
                            center_lon, center_lat = transformer.transform(center_x, center_y)
                        except:
                            pass
                    
                    # Save patches
                    for sensor in ['S30', 'L30', 'S1', 'Planet']:
                        if masked_patches[sensor] is not None:
                            output_path = os.path.join(output_dirs[sensor], 
                                                     f"{patch_name_base}_{sensor.lower()}.tif")
                            
                            window = Window(x_offset, y_offset, patch_size, patch_size)
                            out_meta = metas[sensor].copy()
                            out_meta.update({
                                "height": patch_size,
                                "width": patch_size,
                                "transform": rasterio.windows.transform(window, transforms[sensor])
                            })
                            
                            with rasterio.open(output_path, "w", **out_meta) as dst:
                                dst.write(masked_patches[sensor])
                                if descriptions[sensor] and len(descriptions[sensor]) == dst.count:
                                    dst.descriptions = descriptions[sensor]
                            
                            output_paths[sensor] = os.path.relpath(output_path, region_folder)
                    
                    # Calculate date differences
                    date_diffs = {}
                    if dates['S30']:
                        for sensor in ['L30', 'S1', 'Planet']:
                            if dates[sensor]:
                                # Day difference
                                date_diffs[f'S30_{sensor}_day_diff'] = (dates['S30'] - dates[sensor]).days
                                
                                # DOY difference
                                s30_doy = dates['S30'].timetuple().tm_yday
                                sensor_doy = dates[sensor].timetuple().tm_yday
                                
                                # Handle year boundary (e.g., S30 in January, other sensor in December)
                                if dates['S30'].year != dates[sensor].year:
                                    # If S30 is in the next year
                                    if dates['S30'].year > dates[sensor].year:
                                        # Add days in the previous year to the difference
                                        days_in_prev_year = 366 if dates[sensor].year % 4 == 0 and (dates[sensor].year % 100 != 0 or dates[sensor].year % 400 == 0) else 365
                                        doy_diff = s30_doy + (days_in_prev_year - sensor_doy)
                                    else:
                                        # S30 is in the previous year
                                        days_in_s30_year = 366 if dates['S30'].year % 4 == 0 and (dates['S30'].year % 100 != 0 or dates['S30'].year % 400 == 0) else 365
                                        doy_diff = -(sensor_doy + (days_in_s30_year - s30_doy))
                                else:
                                    # Same year, simple difference
                                    doy_diff = s30_doy - sensor_doy
                                
                                date_diffs[f'S30_{sensor}_doy_diff'] = doy_diff
                    
                    # Store correspondence
                    patch_info = {
                        "s30_path": output_paths.get('S30'),
                        "l30_path": output_paths.get('L30'),
                        "s1_path": output_paths.get('S1'),
                        "pl_path": output_paths.get('Planet'),
                        "original_row_index": index,
                        "x_offset": x_offset,
                        "y_offset": y_offset,
                        "center_latitude": center_lat,
                        "center_longitude": center_lon
                    }
                    
                    # Add dates and DOY
                    for sensor in ['S30', 'L30', 'S1', 'Planet']:
                        if dates[sensor]:
                            patch_info[f"{sensor.lower()}_date"] = dates[sensor].strftime('%Y-%m-%d')
                            patch_info[f"{sensor.lower()}_doy"] = dates[sensor].timetuple().tm_yday
                    
                    # Add date differences
                    patch_info.update(date_diffs)
                    
                    # Process and save GRIDMET data if available
                    if gridmet_folder and dates['S30'] and transforms.get('S30') and 'S30' in sources:
                        for sensor in ['L30', 'S1', 'Planet']:
                            if dates[sensor] and masked_patches[sensor] is not None:
                                # Process GRIDMET data
                                gridmet_values, error_msg = process_gridmet_for_patch(
                                    gridmet_folder, region_name, dates[sensor], dates['S30'],
                                    x_offset, y_offset, patch_size,
                                    transforms['S30'], sources['S30'].crs
                                )
                                
                                if gridmet_values:
                                    # Add GRIDMET values directly to patch_info
                                    for key, value in gridmet_values.items():
                                        patch_info[f"gridmet_{sensor.lower()}_{key}"] = value
                                    patch_info[f"gridmet_{sensor.lower()}_status"] = "processed"
                                    patch_info[f"gridmet_{sensor.lower()}_gap_days"] = (dates['S30'] - dates[sensor]).days
                                else:
                                    patch_info[f"gridmet_{sensor.lower()}_status"] = error_msg
                                    if error_msg == "no_gap":
                                        patch_info[f"gridmet_{sensor.lower()}_gap_days"] = 0
                            else:
                                patch_info[f"gridmet_{sensor.lower()}_status"] = "no_sensor_data"
                    
                    patch_correspondences.append(patch_info)
            
            tqdm.write(f"Row {index}: Generated {patch_count} patches, saved {saved_count} valid patches")
        
        finally:
            # Close all sources
            for sensor in sources:
                sources[sensor].close()
    
    # Save correspondences
    # region_folder
    correspondences_path = os.path.join(outdir, "patch_correspondences_gridmet.json")
    with open(correspondences_path, "w") as f:
        json.dump(patch_correspondences, f, indent=4)
    
    print(f"Saved {len(patch_correspondences)} patch correspondences to {correspondences_path}")

def process_all_regions(base_folder, gridmet_dir, patch_size=128, overlap=0.5, zero_threshold=75.0):
    """Process all region folders"""
    
    if not os.path.isdir(base_folder):
        print(f"Error: Base folder '{base_folder}' not found")
        return
    
    # Find region folders
    region_folders = []
    for d in os.listdir(base_folder):
        full_path = os.path.join(base_folder, d)
        if os.path.isdir(full_path) and d.endswith('_masked'):
            region_folders.append(full_path)
    
    print(f"Found {len(region_folders)} region folders to process")
    
    for region_folder in region_folders:
        print(f"\nProcessing: {region_folder}")
        process_region(region_folder, gridmet_dir, patch_size, overlap, zero_threshold)
    
    print("\nAll regions processed!")


if __name__ == "__main__":
    # Configuration
    base_folder = 'D:/DigAgLab/Alfalfa/satellite/data/FusionDataset/FullDataset/'
    gridmet_dir = 'D:/DigAgLab/Alfalfa/satellite/data/Raw/GRIDMET'  # Update this path
    # Global configuration
    PATCH_SIZE = 64  # Standard patch size for deep learning
    OVERLAP_PERCENT = 0.3  # 50% overlap
    ZERO_THRESHOLD = 80.0  # Maximum allowed percentage of zero pixels
    # Process all regions
    process_all_regions(base_folder, gridmet_dir, PATCH_SIZE, OVERLAP_PERCENT, ZERO_THRESHOLD)

# if __name__ == "__main__":
#     # Configuration
#     base_folder = 'D:/DigAgLab/Alfalfa/satellite/data/FusionDataset/FullDataset'
#     daymet_dir='D:/DigAgLab/Alfalfa/satellite/data/Raw/Daymet'
#     # Global configuration
#     PATCH_SIZE = 64  # Standard patch size for deep learning
#     OVERLAP_PERCENT = 0.5  # 50% overlap
#     ZERO_THRESHOLD = 75.0  # Maximum allowed percentage of zero pixels
#     # Process all regions
#     process_all_regions(base_folder, PATCH_SIZE, OVERLAP_PERCENT, ZERO_THRESHOLD)