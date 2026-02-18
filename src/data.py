import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from zipfile import ZipFile
from pathlib import Path
import random

# Paths
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SIZE = (224, 224)

def setup_directories():
    """Create necessary directories"""
    print("\n=== Setting Up Directories ===")
    
    # Create raw and processed directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Processed data directory: {PROCESSED_DIR}")

def unzip_dataset(zip_path):
    """Directly unzip dataset to raw folder"""
    print(f"\n=== Unzipping Dataset: {zip_path} ===")
    
    if not zip_path.exists():
        print(f"Zip file not found at: {zip_path}")
        print("Please upload the zip file to this location")
        return False
    
    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            # Extract all contents directly to raw folder
            zip_ref.extractall(RAW_DATA_DIR)
        print(f"Successfully unzipped to {RAW_DATA_DIR}")
        
        # List extracted contents
        print("\nExtracted contents:")
        for item in RAW_DATA_DIR.iterdir():
            print(f"  - {item.name}")
        
        return True
    except Exception as e:
        print(f"Error unzipping: {e}")
        return False

def verify_dataset_structure():
    """Verify the dataset has the expected structure"""
    print("\n=== Verifying Dataset Structure ===")
    
    # Check for PetImages directory (common in Kaggle dataset)
    pet_images_dir = RAW_DATA_DIR / "PetImages"
    
    if pet_images_dir.exists():
        print(f"Found PetImages directory at: {pet_images_dir}")
        
        # Check cats and dogs subdirectories
        cats_dir = pet_images_dir / "Cat"
        dogs_dir = pet_images_dir / "Dog"
        
        # Note: Sometimes it's "Cat" and "Dog", sometimes "cats" and "dogs"
        # Let's check both possibilities
        possible_cats_dirs = [pet_images_dir / "Cat", pet_images_dir / "cats", pet_images_dir / "cat"]
        possible_dogs_dirs = [pet_images_dir / "Dog", pet_images_dir / "dogs", pet_images_dir / "dog"]
        
        cats_dir = next((d for d in possible_cats_dirs if d.exists()), None)
        dogs_dir = next((d for d in possible_dogs_dirs if d.exists()), None)
        
        if cats_dir and dogs_dir:
            cat_images = list(cats_dir.glob("*.jpg")) + list(cats_dir.glob("*.jpeg")) + list(cats_dir.glob("*.png"))
            dog_images = list(dogs_dir.glob("*.jpg")) + list(dogs_dir.glob("*.jpeg")) + list(dogs_dir.glob("*.png"))
            
            print(f"Found {len(cat_images)} cat images")
            print(f"Found {len(dog_images)} dog images")
            
            if len(cat_images) > 0 and len(dog_images) > 0:
                return True, cats_dir, dogs_dir
            else:
                print("No images found in the directories!")
                return False, None, None
        else:
            print("Could not find cats and dogs directories in PetImages")
            return False, None, None
    else:
        print("PetImages directory not found. Looking for alternative structures...")
        
        # Look for any directory containing cat/dog images
        all_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
        print(f"Found directories: {[d.name for d in all_dirs]}")
        
        # Try to find cats and dogs directories
        cats_dir = None
        dogs_dir = None
        
        for dir_path in all_dirs:
            dir_name = dir_path.name.lower()
            if 'cat' in dir_name:
                cats_dir = dir_path
            if 'dog' in dir_name:
                dogs_dir = dir_path
        
        if cats_dir and dogs_dir:
            cat_images = list(cats_dir.glob("*.jpg")) + list(cats_dir.glob("*.jpeg")) + list(cats_dir.glob("*.png"))
            dog_images = list(dogs_dir.glob("*.jpg")) + list(dogs_dir.glob("*.jpeg")) + list(dogs_dir.glob("*.png"))
            
            print(f"Found {len(cat_images)} cat images in {cats_dir}")
            print(f"Found {len(dog_images)} dog images in {dogs_dir}")
            
            if len(cat_images) > 0 and len(dog_images) > 0:
                return True, cats_dir, dogs_dir
        
        return False, None, None

def process_and_split_images(cats_dir, dogs_dir):
    """Process images and split into train/val/test"""
    print("\n=== Processing and Splitting Images ===")
    
    # Create processed directories
    for split in ["train", "val", "test"]:
        for class_name in ["cats", "dogs"]:
            os.makedirs(PROCESSED_DIR / split / class_name, exist_ok=True)
    
    # Process cats
    print("\nProcessing cats...")
    cat_images = list(cats_dir.glob("*.jpg")) + list(cats_dir.glob("*.jpeg")) + list(cats_dir.glob("*.png"))
    random.shuffle(cat_images)
    process_class_images(cat_images, "cats")
    
    # Process dogs
    print("\nProcessing dogs...")
    dog_images = list(dogs_dir.glob("*.jpg")) + list(dogs_dir.glob("*.jpeg")) + list(dogs_dir.glob("*.png"))
    random.shuffle(dog_images)
    process_class_images(dog_images, "dogs")

def process_class_images(image_paths, class_name):
    """Process images for a specific class and split them"""
    
    if len(image_paths) == 0:
        print(f"No images found for {class_name}")
        return
    
    print(f"Total {class_name} images: {len(image_paths)}")
    
    # Calculate split sizes
    train_size = int(0.8 * len(image_paths))
    val_size = int(0.1 * len(image_paths))
    test_size = len(image_paths) - train_size - val_size
    
    print(f"Splitting: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Split the data
    train_images = image_paths[:train_size]
    val_images = image_paths[train_size:train_size + val_size]
    test_images = image_paths[train_size + val_size:]
    
    # Process each split
    print(f"Processing {class_name} - Train split...")
    for i, img_path in enumerate(train_images):
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i}/{len(train_images)} training images")
        dest_path = PROCESSED_DIR / "train" / class_name / img_path.name
        process_single_image(img_path, dest_path)
    
    print(f"Processing {class_name} - Validation split...")
    for i, img_path in enumerate(val_images):
        if i % 100 == 0 and i > 0:
            print(f"  Processed {i}/{len(val_images)} validation images")
        dest_path = PROCESSED_DIR / "val" / class_name / img_path.name
        process_single_image(img_path, dest_path)
    
    print(f"Processing {class_name} - Test split...")
    for i, img_path in enumerate(test_images):
        if i % 100 == 0 and i > 0:
            print(f"  Processed {i}/{len(test_images)} test images")
        dest_path = PROCESSED_DIR / "test" / class_name / img_path.name
        process_single_image(img_path, dest_path)
    
    print(f"Completed processing {class_name}")

def process_single_image(src_path, dest_path):
    """Process and save a single image"""
    try:
        # Read image
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"Warning: Could not read image {src_path}")
            return False
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, SIZE)
        
        # Save (convert back to BGR for cv2.imwrite)
        cv2.imwrite(str(dest_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False

def print_statistics():
    """Print final dataset statistics"""
    print("\n=== Final Dataset Statistics ===")
    
    total_images = 0
    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()} SET:")
        for class_name in ["cats", "dogs"]:
            count = len(list((PROCESSED_DIR / split / class_name).glob("*.jpg")))
            total_images += count
            print(f"  {class_name}: {count} images")
    
    print(f"\nTotal processed images: {total_images}")
    
    # Verify paths
    print(f"\nProcessed data saved to: {PROCESSED_DIR}")

def main():
    """Main execution function"""
    print("="*50)
    print("CATS VS DOGS DATA PREPROCESSING")
    print("="*50)
    
    # Step 1: Setup directories
    setup_directories()
    
    # Step 2: Check if zip file exists and unzip
    zip_path = RAW_DATA_DIR / "dogs-vs-cats.zip"
    
    # If zip doesn't exist, look for any zip file
    if not zip_path.exists():
        zip_files = list(RAW_DATA_DIR.glob("*.zip"))
        if zip_files:
            zip_path = zip_files[0]
            print(f"Found alternative zip file: {zip_path.name}")
        else:
            print("\nNo zip file found!")
            print(f"Please upload your Kaggle dataset zip file to: {RAW_DATA_DIR}")
            print("The zip file should be named 'dogs-vs-cats.zip' or similar")
            return
    
    # Step 3: Unzip the dataset
    if not unzip_dataset(zip_path):
        print("Failed to unzip dataset")
        return
    
    # Step 4: Verify dataset structure
    success, cats_dir, dogs_dir = verify_dataset_structure()
    if not success:
        print("\nCould not find cats and dogs images in the expected structure.")
        print("Please check the dataset structure manually.")
        return
    
    # Step 5: Process and split images
    process_and_split_images(cats_dir, dogs_dir)
    
    # Step 6: Print statistics
    print_statistics()
    
    print("\n Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()