#!/usr/bin/env python3
import os
import sys
import numpy as np
import cv2
import mdai
from dotenv import load_dotenv
import traceback
import logging
import time
import glob
import argparse
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("upload_masks.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set environment variables for debugging
os.environ['DEBUG_UPLOADS'] = '1'
os.environ['FORCE_UPLOAD'] = '1'

# Load environment variables
load_dotenv('.env')

# Initialize constants from environment
MDAI_TOKEN = os.getenv('MDAI_TOKEN')
DOMAIN = os.getenv('DOMAIN', 'ucsf.md.ai')
PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv('DATASET_ID')
LABEL_ID_FLUID_OF = os.getenv('LABEL_ID_FLUID_OF')
LABEL_ID_MACHINE_GROUP = os.getenv('LABEL_ID_MACHINE_GROUP')

def print_header(message):
    """Print a nicely formatted header"""
    logging.info("=" * 70)
    logging.info(message)
    logging.info("=" * 70)

def initialize_mdai_client():
    """Initialize MD.ai client"""
    print_header("Initializing MD.ai client")
    
    try:
        client = mdai.Client(domain=DOMAIN, access_token=MDAI_TOKEN)
        logging.info("Successfully connected to MD.ai")
        
        # Test project access
        project_info = client.project(PROJECT_ID)
        logging.info(f"Successfully accessed project: {PROJECT_ID}")
        
        return client
    except Exception as e:
        logging.error(f"Failed to connect to MD.ai: {str(e)}")
        traceback.print_exc()
        return None

def delete_existing_annotations(client, study_uid, series_uid):
    """Delete existing annotations for this study/series"""
    print_header(f"Deleting existing annotations")
    
    try:
        # Get project with annotations
        project = client.project(
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            annotations_only=True
        )
        
        # Count and collect annotation IDs to delete
        deleted_count = 0
        
        if hasattr(project, 'annotations'):
            for ann in project.annotations:
                if (ann.StudyInstanceUID == study_uid and 
                    ann.SeriesInstanceUID == series_uid and
                    ann.labelId == LABEL_ID_FLUID_OF and
                    hasattr(ann, 'groupId') and ann.groupId == LABEL_ID_MACHINE_GROUP):
                    
                    try:
                        client.delete_annotation(ann.id)
                        logging.info(f"Deleted annotation {ann.id} for frame {ann.frameNumber}")
                        deleted_count += 1
                    except Exception as e:
                        logging.error(f"Failed to delete annotation {ann.id}: {str(e)}")
        
        logging.info(f"Deleted {deleted_count} existing annotations")
        return deleted_count
    
    except Exception as e:
        logging.error(f"Error checking/deleting annotations: {str(e)}")
        traceback.print_exc()
        return 0

def load_metadata(masks_dir):
    """Load metadata from the masks directory"""
    metadata_path = os.path.join(masks_dir, "metadata.txt")
    
    # Default values
    metadata = {
        "study_uid": None,
        "series_uid": None,
        "video_path": None,
        "exam_id": None
    }
    
    # First check for metadata.txt
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            for line in f:
                if "Study UID:" in line:
                    metadata["study_uid"] = line.split("Study UID:")[1].strip()
                elif "Series UID:" in line:
                    metadata["series_uid"] = line.split("Series UID:")[1].strip()
                elif "Video path:" in line:
                    metadata["video_path"] = line.split("Video path:")[1].strip()
                elif "Video:" in line:
                    metadata["video_path"] = line.split("Video:")[1].strip()
                elif "Exam ID:" in line:
                    metadata["exam_id"] = line.split("Exam ID:")[1].strip()
    
    # If we have a video path but not UIDs, try to extract them from the path
    if metadata["video_path"] and (not metadata["study_uid"] or not metadata["series_uid"]):
        try:
            parts = metadata["video_path"].split('/')
            # Look for the UIDs in the path (assumes standard format)
            for i, part in enumerate(parts):
                if part.startswith("1.2.826.0.1.3680043.8.498."):
                    # This might be a study UID
                    metadata["study_uid"] = part
                    # Next part might be series UID
                    if i+1 < len(parts) and parts[i+1].startswith("1.2.826.0.1.3680043.8.498."):
                        metadata["series_uid"] = parts[i+1]
                    break
        except Exception as e:
            logging.warning(f"Could not extract UIDs from video path: {e}")
    
    # If metadata is still incomplete, use environment defaults
    if not metadata["study_uid"]:
        metadata["study_uid"] = os.getenv('STUDY_UID', 
             '1.2.826.0.1.3680043.8.498.18050612380255098469086741540114763661')
    
    if not metadata["series_uid"]:
        metadata["series_uid"] = os.getenv('SERIES_UID',
             '1.2.826.0.1.3680043.8.498.72553010565308306328905938562604820392')
    
    if metadata["exam_id"]:
        logging.info(f"Loaded metadata: study_uid={metadata['study_uid']}, series_uid={metadata['series_uid']}, exam_id={metadata['exam_id']}")
    else:
        logging.info(f"Loaded metadata: study_uid={metadata['study_uid']}, series_uid={metadata['series_uid']}")
    return metadata

def load_masks_from_directory(masks_dir):
    """Load masks from the directory"""
    print_header("Loading masks")
    
    # First look for numpy files (preferred)
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "mask_*.npy")))
    
    # If no numpy files, try PNG files
    if not mask_files:
        mask_files = sorted(glob.glob(os.path.join(masks_dir, "mask_*.png")))
    
    logging.info(f"Found {len(mask_files)} mask files")
    
    if not mask_files:
        logging.error("No mask files found")
        return []
    
    # Load masks
    masks = []
    for mask_file in mask_files:
        try:
            # Extract frame number from filename
            frame_number = int(os.path.basename(mask_file).split('_')[1].split('.')[0])
            
            # Load mask based on file type
            if mask_file.endswith(".npy"):
                mask = np.load(mask_file)
            else:  # PNG or other image format
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                # Convert to proper binary format (0 or 1)
                mask = (mask > 127).astype(np.uint8)
            
            if mask is not None:
                masks.append((frame_number, mask))
                if len(masks) <= 5 or len(masks) % 20 == 0:  # Log first 5 and then every 20th
                    logging.info(f"Loaded mask for frame {frame_number}, shape={mask.shape}, sum={np.sum(mask)}")
        except Exception as e:
            logging.error(f"Error loading mask file {mask_file}: {str(e)}")
    
    logging.info(f"Successfully loaded {len(masks)} masks")
    return masks

def prepare_mask_for_mdai(mask, frame_number):
    """Convert mask to MD.ai format"""
    try:
        # Ensure the mask is binary (0 or 1 values)
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Convert mask to MD.ai format using their utility function
        mask_data = mdai.common_utils.convert_mask_data(binary_mask)
        
        if not mask_data:
            logging.warning(f"Failed to convert mask for frame {frame_number}")
            return None
        
        return mask_data
    
    except Exception as e:
        logging.error(f"Error preparing mask for frame {frame_number}: {str(e)}")
        traceback.print_exc()
        return None

def upload_masks_to_mdai(client, masks, study_uid, series_uid, batch_size=10):
    """Upload masks to MD.ai in batches"""
    print_header(f"Uploading {len(masks)} masks to MD.ai")
    
    successful = 0
    failed = 0
    
    # Process in batches to avoid overwhelming the API
    for batch_start in range(0, len(masks), batch_size):
        batch_end = min(batch_start + batch_size, len(masks))
        current_batch = masks[batch_start:batch_end]
        
        logging.info(f"Processing batch {batch_start//batch_size + 1}/{(len(masks) + batch_size - 1)//batch_size}: frames {batch_start} to {batch_end-1}")
        
        # Prepare batch annotations
        annotations = []
        for frame_number, mask in current_batch:
            try:
                # Prepare the mask for MD.ai
                mask_data = prepare_mask_for_mdai(mask, frame_number)
                
                if not mask_data:
                    failed += 1
                    continue
                
                # Create annotation with correct structure
                annotation = {
                    'labelId': LABEL_ID_FLUID_OF,
                    'StudyInstanceUID': study_uid,
                    'SeriesInstanceUID': series_uid,
                    'frameNumber': int(frame_number),
                    'data': mask_data,  # Important: This is the key field for MD.ai
                    'groupId': LABEL_ID_MACHINE_GROUP,
                    'note': f"Uploaded at {time.strftime('%Y-%m-%d %H:%M:%S')}"
                }
                
                annotations.append(annotation)
                
            except Exception as e:
                logging.error(f"Error preparing annotation for frame {frame_number}: {str(e)}")
                failed += 1
        
        if not annotations:
            logging.error(f"No valid annotations in batch {batch_start//batch_size + 1}")
            continue
        
        # Upload batch to MD.ai
        try:
            response = client.import_annotations(
                annotations=annotations,
                project_id=PROJECT_ID,
                dataset_id=DATASET_ID
            )
            
            if response and isinstance(response, list) and len(response) > 0:
                # Some annotations failed
                logging.error(f"Batch upload partially failed: {len(response)} failures out of {len(annotations)}")
                for i, error in enumerate(response[:3]):  # Show first 3 errors
                    logging.error(f"  Error {i+1}: {error}")
                
                successful += len(annotations) - len(response)
                failed += len(response)
            else:
                # All annotations succeeded
                logging.info(f"Successfully uploaded batch of {len(annotations)} annotations")
                successful += len(annotations)
                
        except Exception as e:
            logging.error(f"Error uploading batch: {str(e)}")
            traceback.print_exc()
            failed += len(annotations)
        
        # Sleep briefly to avoid rate limiting
        time.sleep(1)
    
    logging.info(f"Upload results: {successful} successful, {failed} failed")
    return successful, failed

def verify_uploads_on_mdai(client, frame_numbers, study_uid, series_uid):
    """Verify that a sample of uploaded masks exist on MD.ai"""
    print_header("Verifying uploads on MD.ai")
    
    try:
        # Use a direct API call to get annotations
        project_with_annotations = client.project(
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            annotations_only=True
        )
        
        # Check how many of our frames exist
        found_frames = 0
        verified_frames = []
        
        # Get a sample of frame numbers to verify
        sample_size = min(10, len(frame_numbers))
        sample_frames = sorted(frame_numbers)[:sample_size]  # First few frames
        
        for frame_number in sample_frames:
            found = False
            for ann in project_with_annotations.annotations:
                if (ann.StudyInstanceUID == study_uid and 
                    ann.SeriesInstanceUID == series_uid and
                    ann.frameNumber == frame_number and
                    ann.labelId == LABEL_ID_FLUID_OF):
                    
                    found = True
                    found_frames += 1
                    verified_frames.append(frame_number)
                    break
            
            if not found:
                logging.warning(f"No annotation found for frame {frame_number}")
        
        # Log results
        if found_frames > 0:
            logging.info(f"Verified {found_frames} out of {sample_size} sampled frames on MD.ai")
            logging.info(f"Verified frames: {verified_frames}")
            if found_frames == sample_size:
                logging.info("All sampled frames were successfully verified!")
            return True
        else:
            logging.error("Could not verify any uploads on MD.ai")
            return False
            
    except Exception as e:
        logging.error(f"Error verifying uploads: {str(e)}")
        traceback.print_exc()
        return False

def create_upload_summary(masks_dir, successful, total):
    """Create a summary file in the masks directory"""
    summary_path = os.path.join(masks_dir, "upload_summary.json")
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_masks": total,
        "successful_uploads": successful,
        "failed_uploads": total - successful,
        "success_rate": round(successful / total * 100, 2) if total > 0 else 0,
        "mdai_project": PROJECT_ID,
        "mdai_dataset": DATASET_ID
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Created upload summary at {summary_path}")

def upload_masks_from_directory(masks_dir, force=False, batch_size=10):
    """
    Function to upload masks from a directory - for direct import in other scripts
    
    Args:
        masks_dir: Directory containing masks
        force: Whether to force upload even if verification fails
        batch_size: Batch size for uploads
        
    Returns:
        True if successful, False otherwise
    """
    print_header("STARTING MASK UPLOAD PROCESS (direct import)")
    
    # Check if directory exists
    if not os.path.exists(masks_dir):
        logging.error(f"Masks directory not found: {masks_dir}")
        return False
    
    # Load metadata
    metadata = load_metadata(masks_dir)
    
    # Initialize MD.ai client
    client = initialize_mdai_client()
    if not client:
        logging.error("Failed to initialize MD.ai client")
        return False
    
    # Delete existing annotations
    delete_existing_annotations(client, metadata["study_uid"], metadata["series_uid"])
    
    # Load masks from directory
    masks = load_masks_from_directory(masks_dir)
    
    if not masks:
        logging.error("No masks to upload")
        return False
    
    # Upload masks to MD.ai
    successful, failed = upload_masks_to_mdai(
        client, 
        masks, 
        metadata["study_uid"], 
        metadata["series_uid"],
        batch_size=batch_size
    )
    
    # Create summary
    create_upload_summary(masks_dir, successful, len(masks))
    
    # Verify uploads
    frame_numbers = [frame for frame, _ in masks]
    verify_result = verify_uploads_on_mdai(client, frame_numbers, metadata["study_uid"], metadata["series_uid"])
    
    # Final report
    if successful > 0:
        print_header("UPLOAD SUCCESSFUL")
        logging.info(f"Successfully uploaded {successful} out of {len(masks)} masks")
        logging.info(f"Upload success rate: {successful / len(masks) * 100:.2f}%")
        
        if verify_result or force:
            logging.info("Uploads completed and verified")
            return True
        else:
            logging.warning("Verification couldn't find all uploads in MD.ai - check permissions")
            return force  # Return True only if forcing
    else:
        print_header("UPLOAD FAILED")
        logging.error("Failed to upload any masks")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload saved masks to MD.ai")
    parser.add_argument("--masks-dir", type=str, required=True, help="Directory containing saved masks")
    parser.add_argument("--no-delete", action="store_true", help="Skip deleting existing annotations")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of masks to upload in each batch")
    args = parser.parse_args()
    
    print_header("STARTING MASK UPLOAD PROCESS")
    
    # Check if directory exists
    if not os.path.exists(args.masks_dir):
        logging.error(f"Masks directory not found: {args.masks_dir}")
        sys.exit(1)
    
    # Load metadata
    metadata = load_metadata(args.masks_dir)
    
    # Initialize MD.ai client
    client = initialize_mdai_client()
    if not client:
        logging.error("Failed to initialize MD.ai client")
        sys.exit(1)
    
    # Delete existing annotations if requested
    if not args.no_delete:
        delete_existing_annotations(client, metadata["study_uid"], metadata["series_uid"])
    
    # Load masks from directory
    masks = load_masks_from_directory(args.masks_dir)
    
    if not masks:
        logging.error("No masks to upload")
        sys.exit(1)
    
    # Upload masks to MD.ai
    successful, failed = upload_masks_to_mdai(
        client, 
        masks, 
        metadata["study_uid"], 
        metadata["series_uid"],
        batch_size=args.batch_size
    )
    
    # Create summary
    create_upload_summary(args.masks_dir, successful, len(masks))
    
    # Verify uploads
    frame_numbers = [frame for frame, _ in masks]
    verify_result = verify_uploads_on_mdai(client, frame_numbers, metadata["study_uid"], metadata["series_uid"])
    
    # Final report
    if successful > 0:
        print_header("UPLOAD SUCCESSFUL")
        logging.info(f"Successfully uploaded {successful} out of {len(masks)} masks")
        logging.info(f"Upload success rate: {successful / len(masks) * 100:.2f}%")
        
        if verify_result:
            logging.info("Verification confirmed uploads are visible in MD.ai")
        else:
            logging.warning("Verification couldn't find all uploads in MD.ai - check permissions")
    else:
        print_header("UPLOAD FAILED")
        logging.error("Failed to upload any masks")
    
    print_header("PROCESS COMPLETED")

if __name__ == "__main__":
    main() 