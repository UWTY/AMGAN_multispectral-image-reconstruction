# -*- coding: utf-8 -*-
"""
Created on Sun May 25 19:44:54 2025

@author: Tong Yu
"""
import os
import shutil
import glob
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import box
import numpy as np
from pathlib import Path

def get_reference_params(reference_raster):
    """Get spatial parameters from reference raster"""
    with rasterio.open(reference_raster) as src:
        bounds = src.bounds
        res = src.res
        crs = src.crs
        transform = src.transform
    return bounds, res, crs, transform

def get_valid_range(data_type, band_idx=None, custom_ranges=None):
    """Get valid data range for different remote sensing data types
    
    Parameters:
    -----------
    data_type: str
        Type of data (S30, L30, S1, Planet)
    band_idx: int, optional
        Band index (not used currently, but available for band-specific ranges)
    custom_ranges: dict, optional
        Custom valid ranges to override defaults
    
    Returns:
    --------
    tuple: (min_valid, max_valid) for the data type
    """
    # Define default valid ranges for different data types
    default_ranges = {
        'S30': (0.001, 1),      # Sentinel-2 typical range (0-10000 for L2A reflectance)
        'L30': (0.001, 1),      # Landsat similar range
        'S1': (-35, 10),        # Sentinel-1 SAR in dB (typical range -30 to 10 dB)
        'Planet': (0.001, 1),   # Planet similar to optical sensors
    }
    
    # Use custom ranges if provided
    if custom_ranges and data_type in custom_ranges:
        return custom_ranges[data_type]
    
    # Default range if data type not recognized
    default_range = (0, 1)
    
    return default_ranges.get(data_type, default_range)

def clean_array(array, data_type, filename, custom_ranges=None):
    """Clean array by removing NaN, Inf, and extreme values
    
    Parameters:
    -----------
    array: numpy array with shape (bands, height, width)
    data_type: str, type of data (S30, L30, S1, Planet)
    filename: str, filename for logging purposes
    custom_ranges: dict, optional, custom valid ranges
    
    Returns:
    --------
    cleaned array with problematic values set to 0 (keeps all original bands)
    """
    # Get valid range for this data type
    min_valid, max_valid = get_valid_range(data_type, custom_ranges=custom_ranges)
    
    # Define which bands to check for each data type
    check_bands = {
        'S30': [2, 3, 7],      # Red, NIR, SWIR bands (0-based: bands 3, 4, 8)
        'L30': [2, 3, 4],      # Red, NIR, SWIR1 bands
        'S1': [0, 1],          # VV, VH bands
        'Planet': [3, 5, 7]    # Specific Planet bands
    }
    
    # Get bands to check for this data type
    bands_to_check = check_bands.get(data_type, list(range(array.shape[0])))
    
    # Initialize a single-band mask (all False = all pixels valid initially)
    height, width = array.shape[1], array.shape[2]
    combined_mask = np.zeros((height, width), dtype=bool)
    
    # Check each specified band and combine masks
    print(f"    Data quality check for {filename}:")
    total_nan = 0
    total_inf = 0
    total_extreme = 0
    
    for band_idx in bands_to_check:
        if band_idx < array.shape[0]:  # Make sure band exists
            band_data = array[band_idx, :, :]
            
            # Count problematic values in this band
            nan_mask = np.isnan(band_data)
            inf_mask = np.isinf(band_data)
            low_mask = band_data < min_valid
            high_mask = band_data > max_valid
            
            # Combine all invalid conditions for this band
            band_invalid_mask = nan_mask | inf_mask | low_mask | high_mask
            
            # Add to combined mask (pixel is invalid if ANY checked band is invalid)
            combined_mask |= band_invalid_mask
            
            # Count for reporting
            nan_count = nan_mask.sum()
            inf_count = inf_mask.sum()
            extreme_count = (low_mask | high_mask).sum()
            
            if band_invalid_mask.any():
                print(f"      Band {band_idx + 1}: {nan_count} NaN, {inf_count} Inf, {extreme_count} extreme values")
                
            total_nan += nan_count
            total_inf += inf_count
            total_extreme += extreme_count
    
    # Report total invalid pixels
    invalid_pixel_count = combined_mask.sum()
    if invalid_pixel_count > 0:
        print(f"      Total invalid pixels (checking bands {[b+1 for b in bands_to_check]}): {invalid_pixel_count:,}")
        print(f"      Setting these pixels to 0 in ALL bands")
        
        # Apply the mask to ALL bands
        for i in range(array.shape[0]):
            array[i, combined_mask] = 0
    else:
        print(f"      ✓ All checked bands within valid range")
    
    return array  # Return the full array with all original bands

def identify_data_type(filename):
    """Identify data type from filename"""
    filename_lower = filename.lower()
    if filename_lower.startswith('s30'):
        return 'S30'
    elif filename_lower.startswith('l30'):
        return 'L30'
    elif filename_lower.startswith('s1_'):
        return 'S1'
    elif filename_lower.startswith('planet_'):
        return 'Planet'
    else:
        return 'Unknown'

def align_bounds(bounds, res, reference_transform):
    """Align bounds to reference raster pixel grid"""
    # Get reference raster origin
    ref_origin_x = reference_transform.c
    ref_origin_y = reference_transform.f
    
    # Calculate aligned bounds
    left = ref_origin_x + np.floor((bounds.left - ref_origin_x) / res[0]) * res[0]
    right = ref_origin_x + np.ceil((bounds.right - ref_origin_x) / res[0]) * res[0]
    bottom = ref_origin_y + np.floor((bounds.bottom - ref_origin_y) / res[1]) * res[1]
    top = ref_origin_y + np.ceil((bounds.top - ref_origin_y) / res[1]) * res[1]
    
    return rasterio.coords.BoundingBox(left, bottom, right, top)
def align_bounds_minimal(bounds, res, reference_transform):
    """最小化偏移的对齐方法"""
    ref_origin_x = reference_transform.c
    ref_origin_y = reference_transform.f
    
    # 使用round而不是floor/ceil，减少偏移
    left = ref_origin_x + round((bounds.left - ref_origin_x) / res[0]) * res[0]
    right = ref_origin_x + round((bounds.right - ref_origin_x) / res[0]) * res[0]
    bottom = ref_origin_y + round((bounds.bottom - ref_origin_y) / res[1]) * res[1]
    top = ref_origin_y + round((bounds.top - ref_origin_y) / res[1]) * res[1]
    
    # 确保边界框有效
    if right <= left:
        right = left + res[0]
    if top <= bottom:
        top = bottom + res[1]
    
    return rasterio.coords.BoundingBox(left, bottom, right, top)
def determine_utm_zone(lon, lat):
    """Determine UTM zone from longitude and latitude"""
    utm_zone = int((lon + 180) / 6) + 1
    
    # Determine EPSG code
    if lat >= 0:
        epsg_code = 32600 + utm_zone  # Northern hemisphere
    else:
        epsg_code = 32700 + utm_zone  # Southern hemisphere
    
    return epsg_code, utm_zone

def clip_and_align_raster(input_raster, output_raster, shape_gdf, reference_crs, 
                         reference_res, reference_transform, aligned_bounds, 
                         resample_method='auto', custom_valid_ranges=None):
    """Clip and align single raster
    
    Parameters:
    -----------
    resample_method: str
        Resampling method: 'auto' (automatic selection), 'nearest', 'bilinear', 'cubic', 'average'
    """
    
    # Identify data type early
    data_type = identify_data_type(os.path.basename(input_raster))
    
    with rasterio.open(input_raster) as src:
        # Check if source CRS is geographic
        if src.crs.is_geographic:
            print(f"  Source CRS is geographic ({src.crs})")
            # For geographic CRS, we'll reproject to the reference CRS
            need_reproject = True
            # Can't compare resolution directly for geographic vs projected
            need_resample = True
        else:
            # Both are projected, can compare normally
            src_res = src.res
            res_ratio_x = src_res[0] / reference_res[0]
            res_ratio_y = src_res[1] / reference_res[1]
            
            print(f"  Original resolution: {src_res[0]:.2f} x {src_res[1]:.2f}")
            print(f"  Target resolution: {reference_res[0]:.2f} x {reference_res[1]:.2f}")
            print(f"  Resolution ratio: {res_ratio_x:.2f} x {res_ratio_y:.2f}")
            
            need_reproject = src.crs != reference_crs
            need_resample = abs(res_ratio_x - 1) > 0.01 or abs(res_ratio_y - 1) > 0.01
        
        # Automatically select resampling method
        if resample_method == 'auto':
            # Identify data type from filename
            filename_lower = os.path.basename(input_raster).lower()
            
            if src.crs.is_geographic:
                # For geographic data being reprojected, use bilinear
                resampling = Resampling.bilinear
                print(f"  Using bilinear resampling for geographic to projected transformation")
            elif need_resample and not src.crs.is_geographic:
                # Only check resolution ratios if source is projected
                if res_ratio_x > 1.5 or res_ratio_y > 1.5:
                    resampling = Resampling.average
                    print(f"  Using average resampling (downsampling)")
                elif res_ratio_x < 0.67 or res_ratio_y < 0.67:
                    resampling = Resampling.cubic
                    print(f"  Using cubic resampling (upsampling)")
                else:
                    resampling = Resampling.bilinear
                    print(f"  Using bilinear resampling (similar resolution)")
            else:
                resampling = Resampling.bilinear
                print(f"  Using bilinear resampling")
                
            # Special handling for specific data types
            if 'planet' in filename_lower and not src.crs.is_geographic:
                if res_ratio_x > 5 or res_ratio_y > 5:
                    print(f"  Note: Planet data detected with large resolution difference")
        else:
            resampling_methods = {
                'nearest': Resampling.nearest,
                'bilinear': Resampling.bilinear,
                'cubic': Resampling.cubic,
                'average': Resampling.average
            }
            resampling = resampling_methods.get(resample_method, Resampling.bilinear)
        
        if need_reproject or need_resample:
            if need_reproject:
                print(f"  Converting CRS: {src.crs} -> {reference_crs}")
            if need_resample:
                print(f"  Resampling to target resolution")
            
            # Calculate new transform*****************************************
            # transform, width, height = calculate_default_transform(
            #     src.crs, reference_crs, src.width, src.height, 
            #     *src.bounds, resolution=reference_res
            # )
            # Use the reference transform and bounds for strict alignment
            transform = reference_transform
            aligned_width = int((aligned_bounds.right - aligned_bounds.left) / reference_res[0])
            aligned_height = int((aligned_bounds.top - aligned_bounds.bottom) / reference_res[1])
            width, height = aligned_width,aligned_height
            
            
            # Create temporary file for reprojection/resampling
            temp_file = output_raster.replace('.tif', '_temp.tif')
            
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': reference_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'nodata': 0,  # Set nodata value to 0 in temporary file
                'dtype': 'float32' if data_type == 'Planet' else src.meta['dtype']  # Use float32 for Planet
            })
            
            # Get original band descriptions
            band_descriptions = []
            for i in range(1, src.count + 1):
                band_descriptions.append(src.descriptions[i-1] if src.descriptions[i-1] else f"Band_{i}")
            
            with rasterio.open(temp_file, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    # Read band data
                    band_data = src.read(i)
                    
                    # Scale Planet data from 0-10000 to 0-1
                    if data_type == 'Planet':
                        band_data = band_data.astype('float32') / 10000.0
                        if i == 1:  # Print message only once
                            print(f"  Scaling Planet data from [0-10000] to [0-1]")
                    
                    # Reproject
                    reprojected = np.zeros((height, width), dtype=kwargs['dtype'])
                    reproject(
                        source=band_data,
                        destination=reprojected,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=reference_crs,
                        resampling=resampling
                    )
                    dst.write(reprojected, i)
                    
                # Set band descriptions in temporary file
                for i in range(1, dst.count + 1):
                    dst.set_band_description(i, band_descriptions[i-1])
            
            # Use reprojected file for clipping
            clip_input = temp_file
        else:
            clip_input = input_raster
            temp_file = None
        
        # Open raster to clip
        with rasterio.open(clip_input) as src2:
            # Reproject shapefile to raster CRS
            shape_gdf_reproj = shape_gdf.to_crs(reference_crs)
            
            # Clip raster
            out_image, out_transform = mask(src2, shape_gdf_reproj.geometry, crop=True)
            
            # If Planet data wasn't already scaled (no reprojection case), scale it now
            if data_type == 'Planet' and temp_file is None:
                out_image = out_image.astype('float32') / 10000.0
                print(f"  Scaling Planet data from [0-10000] to [0-1]")
            
            # Calculate aligned width and height
            aligned_width = int((aligned_bounds.right - aligned_bounds.left) / reference_res[0])
            aligned_height = int((aligned_bounds.top - aligned_bounds.bottom) / reference_res[1])
            
            # Create aligned transform
            aligned_transform = rasterio.transform.from_bounds(
                aligned_bounds.left, aligned_bounds.bottom,
                aligned_bounds.right, aligned_bounds.top,
                aligned_width, aligned_height
            )

            
            # Create output array
            dtype = 'float32' if data_type == 'Planet' else out_image.dtype
            aligned_image = np.full(
                (src2.count, aligned_height, aligned_width),
                0,  # Use 0 as fill value for nodata areas
                dtype=dtype
            )
            
            # Calculate offsets
            row_off = int((aligned_bounds.top - out_transform.f) / reference_res[1])
            col_off = int((out_transform.c - aligned_bounds.left) / reference_res[0])
            
            # Ensure offsets are non-negative
            row_off = max(0, row_off)
            col_off = max(0, col_off)
            
            # Place clipped data into aligned array
            out_h, out_w = out_image.shape[1], out_image.shape[2]
            
            # Calculate valid copy region
            src_row_start = 0
            src_col_start = 0
            dst_row_start = row_off
            dst_col_start = col_off
            
            copy_height = min(out_h, aligned_height - dst_row_start)
            copy_width = min(out_w, aligned_width - dst_col_start)
            
            if copy_height > 0 and copy_width > 0:
                aligned_image[:, dst_row_start:dst_row_start+copy_height, 
                            dst_col_start:dst_col_start+copy_width] = \
                    out_image[:, src_row_start:src_row_start+copy_height,
                            src_col_start:src_col_start+copy_width]
            
            # Clean the aligned image to remove NaN, Inf, and extreme values
            aligned_image = clean_array(aligned_image, data_type, os.path.basename(input_raster), custom_valid_ranges)
            
            # Update metadata
            out_meta = src2.meta.copy()
            # out_meta.update({
            #     "transform": out_transform,  # 使用mask操作返回的transform
            #     "height": out_image.shape[1],
            #     "width": out_image.shape[2],
            # })
            out_meta.update({
                "driver": "GTiff",
                "height": aligned_height,
                "width": aligned_width,
                "transform": aligned_transform,
                "crs": reference_crs,
                "compress": "lzw",
                "nodata": 0,  # Set nodata value to 0
                "dtype": dtype  # Use appropriate dtype
            })
            
            # Preserve band descriptions
            band_descriptions = []
            for i in range(1, src2.count + 1):
                band_descriptions.append(src2.descriptions[i-1] if src2.descriptions[i-1] else f"Band_{i}")
            
            # Write output file
            with rasterio.open(output_raster, "w", **out_meta) as dest:
                dest.write(aligned_image)
                # Set band descriptions for all bands
                for i in range(1, dest.count + 1):
                    dest.set_band_description(i, band_descriptions[i-1])
        
        # Delete temporary file
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

def process_folder(input_folder, shp_file, output_folder, resample_method='auto', custom_valid_ranges=None):
    """Process all remote sensing images in folder
    
    Parameters:
    -----------
    resample_method: str
        Resampling method: 'auto' (automatic selection), 'nearest', 'bilinear', 'cubic', 'average'
    custom_valid_ranges: dict, optional
        Custom valid ranges for each data type, e.g., {'S30': (0, 12000), 'S1': (-35, 15)}
    """
    
    # Update valid ranges if custom ones provided
    if custom_valid_ranges:
        for data_type, range_values in custom_valid_ranges.items():
            print(f"Using custom valid range for {data_type}: {range_values}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Read shapefile
    print(f"Reading shapefile: {shp_file}")
    shape_gdf = gpd.read_file(shp_file)
    
    # Find different types of files
    file_patterns = {
        'S30': 'S30_*.tif',
        'L30': 'L30_*.tif',
        'S1': 'S1_*.tif',
        'Planet': 'Planet_*.tif'
    }
    
    found_files = {}
    for data_type, pattern in file_patterns.items():
        files = glob.glob(os.path.join(input_folder, pattern))
        if files:
            found_files[data_type] = files
            print(f"Found {data_type} files: {len(files)}")
    
    if not found_files:
        print("Error: No matching remote sensing image files found")
        return
    
    # Analyze spatial resolution and CRS for each data type
    print("\nAnalyzing coordinate systems and resolutions:")
    crs_info = {}
    
    for data_type, files in found_files.items():
        with rasterio.open(files[0]) as src:
            crs_info[data_type] = {
                'crs': src.crs,
                'res': src.res,
                'is_geographic': src.crs.is_geographic
            }
            
            if src.crs.is_geographic:
                # Estimate resolution in meters for geographic CRS
                bounds = src.bounds
                center_lat = (bounds.top + bounds.bottom) / 2
                meters_per_degree_lat = 111320.0
                meters_per_degree_lon = 111320.0 * np.cos(np.radians(center_lat))
                res_x_meters = src.res[0] * meters_per_degree_lon
                res_y_meters = src.res[1] * meters_per_degree_lat
                print(f"{data_type}: {src.crs} (Geographic) - Resolution: ~{res_x_meters:.1f} x {res_y_meters:.1f} meters")
            else:
                print(f"{data_type}: {src.crs} (Projected) - Resolution: {src.res[0]:.1f} x {src.res[1]:.1f} meters")
    
    # Use S30 as reference (mandatory)
    reference_file = None
    if 'S30' in found_files and found_files['S30']:
        s30_files = found_files['S30']
        
        if len(s30_files) == 1:
            reference_file = s30_files[0]
        else:
            print(f"\nFound {len(s30_files)} S30 files. Selecting the largest file as reference...")
            
            # Select based on file size (larger file likely has more valid data)
            file_sizes = [(f, os.path.getsize(f)) for f in s30_files]
            file_sizes.sort(key=lambda x: x[1], reverse=True)
            reference_file = file_sizes[0][0]
            
            # Show all files with sizes for transparency
            print("  S30 files by size:")
            for f, size in file_sizes:
                size_mb = size / (1024 * 1024)
                selected = " <- SELECTED" if f == reference_file else ""
                print(f"    {os.path.basename(f)}: {size_mb:.2f} MB{selected}")
            
        print(f"\nUsing S30 as reference image: {os.path.basename(reference_file)}")
    else:
        print(f"\nERROR: S30 file is required as reference but not found!")
        print("S30 must be present in the input folder to ensure proper alignment.")
        print("Please add S30 data to the input folder and try again.")
        return
    
    # Get reference parameters
    ref_bounds, ref_res, ref_crs, ref_transform = get_reference_params(reference_file)
    
    # Handle geographic reference CRS
    if ref_crs.is_geographic:
        print(f"\nReference S30 has geographic CRS ({ref_crs})")
        print("Determining appropriate UTM zone for processing...")
        
        with rasterio.open(reference_file) as src:
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.top + bounds.bottom) / 2
            
            # Determine UTM zone
            epsg_code, utm_zone = determine_utm_zone(center_lon, center_lat)
            ref_crs = CRS.from_epsg(epsg_code)
            
            print(f"Converting to UTM Zone {utm_zone}{'N' if center_lat >= 0 else 'S'} (EPSG:{epsg_code})")
            
            # Estimate resolution in meters
            meters_per_degree_lat = 111320.0
            meters_per_degree_lon = 111320.0 * np.cos(np.radians(center_lat))
            ref_res = (ref_res[0] * meters_per_degree_lon, ref_res[1] * meters_per_degree_lat)
    
    print(f"\nReference CRS for processing: {ref_crs}")
    print(f"Reference resolution: {ref_res[0]:.2f} x {ref_res[1]:.2f} meters")
    print("All data will be reprojected/resampled/aligned to S30 grid")
    
    # Reproject shapefile to reference CRS and get bounds
    shape_gdf_ref = shape_gdf.to_crs(ref_crs)
    shape_bounds = shape_gdf_ref.total_bounds
    shape_bounds_box = rasterio.coords.BoundingBox(
        shape_bounds[0], shape_bounds[1], shape_bounds[2], shape_bounds[3]
    )
    
    # For geographic reference, we need to recalculate the transform after projection
    if crs_info['S30']['is_geographic']:
        # Open S30 and calculate projected transform
        with rasterio.open(reference_file) as src:
            # Calculate transform for projected CRS
            transform, width, height = calculate_default_transform(
                src.crs, ref_crs, src.width, src.height,
                *src.bounds, resolution=ref_res
            )
            ref_transform = transform
    
    # Align bounds to reference raster pixel grid
    aligned_bounds = align_bounds(shape_bounds_box, ref_res, ref_transform)
    print(f"Aligned bounds: {aligned_bounds}")
    
    # Process all files
    print("\nStarting image processing...")
    for data_type, files in found_files.items():
        print(f"\nProcessing {data_type} data:")
        
        for input_file in files:
            filename = os.path.basename(input_file)
            output_file = os.path.join(output_folder, filename)  # Keep original filename
            
            print(f"  Processing: {filename}")
            try:
                clip_and_align_raster(
                    input_file, output_file, shape_gdf,
                    ref_crs, ref_res, ref_transform, aligned_bounds,
                    resample_method=resample_method,
                    custom_valid_ranges=custom_valid_ranges
                )
                print(f"  Completed: {output_file}")
            except Exception as e:
                print(f"  Error: {filename} - {str(e)}")
                import traceback
                traceback.print_exc()
    
    print("\nAll processing completed!")
    
    # Verify alignment
    print("\nVerifying output image alignment:")
    output_files = glob.glob(os.path.join(output_folder, "*.tif"))  # All .tif files
    
    if len(output_files) > 1:
        first_file = output_files[0]
        with rasterio.open(first_file) as src1:
            ref_transform = src1.transform
            ref_shape = (src1.height, src1.width)
            ref_bounds = src1.bounds
            ref_res = src1.res
            
        all_aligned = True
        for output_file in output_files[1:]:
            with rasterio.open(output_file) as src2:
                if (src2.transform != ref_transform or 
                    (src2.height, src2.width) != ref_shape or
                    src2.bounds != ref_bounds):
                    all_aligned = False
                    print(f"  Warning: {os.path.basename(output_file)} not fully aligned")
        
        if all_aligned:
            print("  ✓ All output images are perfectly aligned!")
            print(f"  Unified resolution: {ref_res[0]:.2f} x {ref_res[1]:.2f} meters")
            print(f"  Unified dimensions: {ref_shape[1]} x {ref_shape[0]} pixels")
        else:
            print("  ✗ Some images are not aligned, please check outputs")
#%%main 
# Usage example
if __name__ == "__main__":
    # Study area lists
    # SALists = ["SA2", "SA3", "SA4", "SA6", "SA7", "SA9", "SA10", "SA12"]
    SALists =["SA2", "SA4", "SA6", "SA7", "SA9", "SA12"]
    # SALists = ["SA4", "SA7", "SA9", "SA12", "SA13"]
    
    # Process each study area
    for SA in SALists:
        year = "2023"
        
        print(f"\n{'='*60}")
        print(f"Processing Study Area: {SA}, Year: {year}")
        print(f"{'='*60}")
        
        # Set paths based on year
        # if year == "2022":
        #     shapefile_path = f'D:/DigAgLab/Alfalfa/satellite/ArcGISResults/RasterT_alfalf_EliminatePoly_{SA}.shp'
        # else:
        #     shapefile_path = f'D:/DigAgLab/Alfalfa/satellite/ArcGISResults/polygon_{year}_{SA}.shp'
        shapefile_path = f'D:/DigAgLab/Alfalfa/satellite/ArcGISResults/polygon_{year}_{SA}.shp'
        
        input_raster_path = f'D:/DigAgLab/Alfalfa/satellite/data/FusionDataset/FullDataset/{SA}_Exports_{year}/'
        output_folder_for_masked = f'D:/DigAgLab/Alfalfa/satellite/data/FusionDataset/FullDataset/{SA}_Exports_{year}_masked/'
        
        # Check if paths exist
        if not os.path.exists(shapefile_path):
            print(f"Warning: Shapefile not found for {SA}: {shapefile_path}")
            print(f"Skipping {SA}...")
            continue
            
        if not os.path.exists(input_raster_path):
            print(f"Warning: Input folder not found for {SA}: {input_raster_path}")
            print(f"Skipping {SA}...")
            continue
        
        try:
            # Execute processing with automatic resampling
            # S30 is mandatory reference for alignment and resampling
            
            # Optional: Define custom valid ranges for specific data types
            # custom_ranges = {
            #     'S30': (0, 12000),    # Adjust if your S30 data has different range
            #     'L30': (0, 12000),    # Adjust for Landsat
            #     'S1': (-35, 15),      # Adjust for SAR data in dB
            #     'Planet': (0, 15000)  # Adjust for Planet data
            # }
            
            process_folder(input_raster_path, shapefile_path, output_folder_for_masked, 
                          resample_method='auto', custom_valid_ranges=None)
            
            print(f"\nSuccessfully completed processing for {SA}")
            
        except Exception as e:
            print(f"\nError processing {SA}: {str(e)}")
            if "S30 file is required" in str(e):
                print(f"Skipping {SA} - S30 reference data is missing")
            print(f"Continuing with next study area...")
            continue
        # Set your source and destination directories
        source_folder = input_raster_path
        destination_folder = output_folder_for_masked
        
        # Create the destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)
        
        # Loop through all files in the source folder
        for filename in os.listdir(source_folder):
            if filename.lower().endswith('.csv'):
                full_file_name = os.path.join(source_folder, filename)
                shutil.copy(full_file_name, destination_folder)
        print("CSV files copied successfully.")
    
    print(f"\n{'='*60}")
    print("All study areas processing completed!")
    print(f"{'='*60}")

