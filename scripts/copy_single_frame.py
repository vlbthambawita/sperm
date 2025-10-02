#!/usr/bin/env python3
"""
Copy one frame from each unique video to create a representative dataset.

This script identifies unique videos by their video ID (first part of filename)
and copies one frame (with its annotation) from each video to a new directory.

Filename format: {video_id}.{field_of_view}_{timestamp}-frame{number}_clean.jpg
Example: 1.1_1sec-frame0_clean.jpg -> video_id = "1", field_of_view = "1"
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

def parse_filename(filename):
    """
    Parse filename to extract video_id and other components.
    
    Args:
        filename: e.g., "1.1_1sec-frame0_clean.jpg"
    
    Returns:
        dict with video_id, field_of_view, frame_number, etc.
    """
    stem = Path(filename).stem  # Remove extension
    
    # Remove '_clean' suffix if present
    if stem.endswith('_clean'):
        stem = stem[:-6]
    
    try:
        # Split by underscore: "1.1_1sec-frame0" -> ["1.1", "1sec-frame0"]
        parts = stem.split('_', 1)
        if len(parts) != 2:
            return None
            
        video_field = parts[0]  # "1.1"
        frame_part = parts[1]   # "1sec-frame0"
        
        # Extract video_id (first part before dot)
        video_id = video_field.split('.')[0]  # "1"
        field_of_view = video_field.split('.')[1] if '.' in video_field else "unknown"
        
        # Extract frame number
        if '-frame' in frame_part:
            frame_str = frame_part.split('-frame')[1]
            frame_number = int(frame_str) if frame_str.isdigit() else 0
        else:
            frame_number = 0
            
        return {
            'video_id': video_id,
            'field_of_view': field_of_view,
            'frame_number': frame_number,
            'full_prefix': video_field,
            'original_stem': stem
        }
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse filename {filename}: {e}")
        return None


def main():
    # Configuration
    source_dir = Path("/work/vajira/DATA/SPERM_data_2025/finalOutput")
    target_dir = Path("/work/vajira/DATA/SPERM_data_2025/clean_100_frames")
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [f for f in source_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} image files in {source_dir}")
    
    # Group files by patient_id (video_id)
    patients = defaultdict(list)
    
    for img_file in image_files:
        parsed = parse_filename(img_file.name)
        if parsed:
            patients[parsed['video_id']].append({
                'file': img_file,
                'parsed': parsed
            })
    
    print(f"Found {len(patients)} unique patients:")
    for patient_id, frames in patients.items():
        print(f"  Patient {patient_id}: {len(frames)} frames")
    
    # Copy one frame from each patient (prefer sample A over B, then frame 0)
    copied_count = 0
    
    for patient_id, frames in patients.items():
        # Group by field_of_view and sample type
        field_samples = defaultdict(lambda: {'A': [], 'B': [], 'unknown': []})
        
        for frame_info in frames:
            parsed = frame_info['parsed']
            # Extract sample type from filename (A or B)
            filename = frame_info['file'].name
            if '.A_' in filename:
                sample_type = 'A'
            elif '.B_' in filename:
                sample_type = 'B'
            else:
                sample_type = 'unknown'
            
            field_samples[parsed['field_of_view']][sample_type].append(frame_info)
        
        # Select best frame: prefer sample A, then lowest frame number
        best_frame = None
        
        for field_of_view, samples in field_samples.items():
            # First try sample A
            if samples['A']:
                candidates = samples['A']
                candidates.sort(key=lambda x: x['parsed']['frame_number'])
                best_frame = candidates[0]
                break
            # If no sample A, try sample B
            elif samples['B']:
                candidates = samples['B']
                candidates.sort(key=lambda x: x['parsed']['frame_number'])
                best_frame = candidates[0]
                break
            # If no A or B, try unknown
            elif samples['unknown']:
                candidates = samples['unknown']
                candidates.sort(key=lambda x: x['parsed']['frame_number'])
                best_frame = candidates[0]
                break
        
        # If no A or B found, take any available frame
        if not best_frame:
            frames.sort(key=lambda x: x['parsed']['frame_number'])
            best_frame = frames[0]
        
        selected_frame = best_frame
        img_file = selected_frame['file']
        parsed = selected_frame['parsed']
        
        # Find corresponding annotation file
        txt_file = img_file.with_name(f"{parsed['original_stem']}.txt")
        
        # Determine sample type for logging
        sample_type = 'A' if '.A_' in img_file.name else ('B' if '.B_' in img_file.name else 'unknown')
        
        # Copy image file
        target_img = target_dir / img_file.name
        shutil.copy2(img_file, target_img)
        print(f"Copied: {img_file.name} (patient {patient_id}, sample {sample_type}, frame {parsed['frame_number']})")
        
        # Copy annotation file if it exists
        if txt_file.exists():
            target_txt = target_dir / txt_file.name
            shutil.copy2(txt_file, target_txt)
            print(f"  + annotation: {txt_file.name}")
        else:
            print(f"  - no annotation found for {txt_file.name}")
        
        copied_count += 1
    
    print(f"\n✅ Successfully copied {copied_count} frames from {len(patients)} unique patients")
    print(f"📁 Output directory: {target_dir}")
    
    # Show summary of what was copied
    target_files = list(target_dir.glob("*"))
    images = [f for f in target_files if f.suffix.lower() in image_extensions]
    annotations = [f for f in target_files if f.suffix == '.txt']
    
    print(f"📊 Summary:")
    print(f"   Images: {len(images)}")
    print(f"   Annotations: {len(annotations)}")


if __name__ == "__main__":
    main()
