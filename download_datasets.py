"""
Download Salinas and Indian Pines hyperspectral datasets.

This script downloads the .mat files for both datasets from their
public repositories and organizes them into the datasets/ directory.
"""

import os
import urllib.request
import sys
from pathlib import Path


# Dataset URLs
DATASETS = {
    'salinas': {
        'data': 'https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
        'gt': 'https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat',
        'dir': 'datasets/salinas'
    },
    'indian_pines': {
        'data': 'https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
        'gt': 'https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',
        'dir': 'datasets/indian_pines'
    }
}


def download_file(url, destination):
    """
    Download a file from URL to destination with progress reporting.
    
    Args:
        url: URL to download from
        destination: Local file path to save to
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Downloading: {url}")
        print(f"Saving to: {destination}")
        
        def report_progress(block_num, block_size, total_size):
            """Report download progress."""
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {percent:.1f}% ({downloaded_mb:.2f}/{total_mb:.2f} MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print()  # New line after progress
        
        # Verify file exists and has content
        if os.path.exists(destination) and os.path.getsize(destination) > 0:
            file_size_mb = os.path.getsize(destination) / (1024 * 1024)
            print(f"✓ Successfully downloaded ({file_size_mb:.2f} MB)")
            return True
        else:
            print(f"✗ Download failed or file is empty")
            return False
            
    except Exception as e:
        print(f"\n✗ Error downloading file: {e}")
        return False


def create_directory_structure():
    """Create the datasets directory structure."""
    print("\n" + "="*60)
    print("Creating directory structure...")
    print("="*60)
    
    for dataset_name, config in DATASETS.items():
        dataset_dir = config['dir']
        Path(dataset_dir).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dataset_dir}")


def download_dataset(dataset_name):
    """
    Download a specific dataset.
    
    Args:
        dataset_name: Name of dataset ('salinas' or 'indian_pines')
    
    Returns:
        bool: True if all files downloaded successfully
    """
    if dataset_name not in DATASETS:
        print(f"✗ Unknown dataset: {dataset_name}")
        return False
    
    config = DATASETS[dataset_name]
    print("\n" + "="*60)
    print(f"Downloading {dataset_name.replace('_', ' ').title()} Dataset")
    print("="*60)
    
    success = True
    
    # Download data file
    data_filename = os.path.basename(config['data'])
    data_path = os.path.join(config['dir'], data_filename)
    
    if os.path.exists(data_path):
        file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
        print(f"\nData file already exists: {data_path} ({file_size_mb:.2f} MB)")
        print("Skipping download. Delete file to re-download.")
    else:
        if not download_file(config['data'], data_path):
            success = False
    
    # Download ground truth file
    gt_filename = os.path.basename(config['gt'])
    gt_path = os.path.join(config['dir'], gt_filename)
    
    if os.path.exists(gt_path):
        file_size_kb = os.path.getsize(gt_path) / 1024
        print(f"\nGround truth file already exists: {gt_path} ({file_size_kb:.2f} KB)")
        print("Skipping download. Delete file to re-download.")
    else:
        if not download_file(config['gt'], gt_path):
            success = False
    
    return success


def verify_downloads():
    """
    Verify that all dataset files have been downloaded.
    
    Returns:
        bool: True if all files exist
    """
    print("\n" + "="*60)
    print("Verifying Downloads")
    print("="*60)
    
    all_present = True
    
    for dataset_name, config in DATASETS.items():
        print(f"\n{dataset_name.replace('_', ' ').title()}:")
        
        data_filename = os.path.basename(config['data'])
        data_path = os.path.join(config['dir'], data_filename)
        
        gt_filename = os.path.basename(config['gt'])
        gt_path = os.path.join(config['dir'], gt_filename)
        
        if os.path.exists(data_path):
            file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
            print(f"  ✓ Data: {data_path} ({file_size_mb:.2f} MB)")
        else:
            print(f"  ✗ Data: {data_path} (missing)")
            all_present = False
        
        if os.path.exists(gt_path):
            file_size_kb = os.path.getsize(gt_path) / 1024
            print(f"  ✓ Ground truth: {gt_path} ({file_size_kb:.2f} KB)")
        else:
            print(f"  ✗ Ground truth: {gt_path} (missing)")
            all_present = False
    
    return all_present


def main():
    """Main function to download all datasets."""
    print("="*60)
    print("Hyperspectral Dataset Downloader")
    print("="*60)
    print("\nThis script will download:")
    print("  - Salinas dataset (512x217 pixels, 204 bands)")
    print("  - Indian Pines dataset (145x145 pixels, 200 bands)")
    print()
    
    # Create directory structure
    create_directory_structure()
    
    # Download datasets
    salinas_ok = download_dataset('salinas')
    indian_pines_ok = download_dataset('indian_pines')
    
    # Verify all downloads
    all_ok = verify_downloads()
    
    # Final summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    
    if all_ok:
        print("\n✓ All datasets downloaded successfully!")
        print("\nDataset locations:")
        print(f"  - Salinas: {DATASETS['salinas']['dir']}/")
        print(f"  - Indian Pines: {DATASETS['indian_pines']['dir']}/")
        print("\nYou can now use hyperspectral_loader.py to load these datasets.")
        return 0
    else:
        print("\n✗ Some files are missing. Please check the errors above.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
