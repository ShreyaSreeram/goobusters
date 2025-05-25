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
import json
import tempfile
import subprocess
import re

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum information
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("force_upload.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set environment variables to force uploads and debugging
os.environ['DEBUG_UPLOADS'] = '1'
os.environ['FORCE_UPLOAD'] = '1'
os.environ['MDAI_UPLOAD_FIX'] = '1'
os.environ['DEBUG_TRACKING'] = '1'
os.environ['MASK_FORMAT_DEBUG'] = '1'

# Load environment variables
load_dotenv('.env')

# Initialize constants
MDAI_TOKEN = os.getenv('MDAI_TOKEN')
DOMAIN = os.getenv('DOMAIN', 'ucsf.md.ai')
PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv('DATASET_ID')
LABEL_ID_FLUID_OF = os.getenv('LABEL_ID_FLUID_OF')
LABEL_ID_MACHINE_GROUP = os.getenv('LABEL_ID_MACHINE_GROUP')

# Exam #64 specific paths
VIDEO_PATH = 'data/mdai_ucsf_project_x9N2LJBZ_images_dataset_D_V688LQ_2025-05-19-171632/1.2.826.0.1.3680043.8.498.18050612380255098469086741540114763661/1.2.826.0.1.3680043.8.498.72553010565308306328905938562604820392.mp4'
STUDY_UID = '1.2.826.0.1.3680043.8.498.18050612380255098469086741540114763661'
SERIES_UID = '1.2.826.0.1.3680043.8.498.72553010565308306328905938562604820392'

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
        return None

def delete_existing_annotations(client):
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
                if (ann.StudyInstanceUID == STUDY_UID and 
                    ann.SeriesInstanceUID == SERIES_UID and
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

def run_algorithm_with_mask_interception():
    """Run the algorithm with a special command to intercept and save masks"""
    print_header("Running algorithm to generate and intercept masks")
    
    # Create a temporary directory to store masks
    temp_dir = tempfile.mkdtemp(prefix="masks_")
    os.environ['FORCE_SAVE_MASKS'] = temp_dir
    logging.info(f"Saving masks to temporary directory: {temp_dir}")
    
    # Run the algorithm with special flag to save masks
    cmd = [
        "python", 
        "src/consolidated_tracking.py", 
        "--video-path", VIDEO_PATH,
        "--debug",
        "--feedback-loop",
        "--skip-mdai"  # Skip the built-in upload to intercept masks
    ]
    
    logging.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Process output line by line
        for line in process.stdout:
            # Only log important lines
            if "mask" in line.lower() or "upload" in line.lower() or "error" in line.lower() or "warning" in line.lower():
                logging.info(f"Process: {line.strip()}")
        
        process.wait()
        logging.info(f"Algorithm completed with code: {process.returncode}")
        
        # Check for mask files
        mask_files = sorted(glob.glob(os.path.join(temp_dir, "*.npy")))
        logging.info(f"Found {len(mask_files)} mask files")
        
        # Load mask data
        masks = []
        for mask_file in mask_files:
            try:
                # Extract frame number from filename
                match = re.search(r'frame_(\d+)\.npy', mask_file)
                if match:
                    frame_number = int(match.group(1))
                    mask_data = np.load(mask_file)
                    masks.append((frame_number, mask_data))
                    logging.info(f"Loaded mask for frame {frame_number}, shape={mask_data.shape}, sum={np.sum(mask_data)}")
            except Exception as e:
                logging.error(f"Error loading mask file {mask_file}: {str(e)}")
        
        return masks
    
    except Exception as e:
        logging.error(f"Error running algorithm: {str(e)}")
        traceback.print_exc()
        return []

def create_test_masks():
    """Create test masks for upload verification"""
    print_header("Creating test masks for verification")
    
    try:
        # Open video to get dimensions
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            logging.error(f"Could not open video: {VIDEO_PATH}")
            return []
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"Video dimensions: {width}x{height}, {frame_count} frames")
        
        # Create a circular mask in the center
        masks = []
        for frame_number in range(0, min(frame_count, 10), 1):  # Create 10 test masks
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Draw a circle at center
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4
            cv2.circle(mask, (center_x, center_y), radius, 1, -1)
            
            # Add frame number dependent detail
            offset = frame_number * 20
            cv2.circle(mask, (center_x + offset, center_y - offset), radius // 2, 1, -1)
            
            masks.append((frame_number, mask))
            logging.info(f"Created test mask for frame {frame_number}, sum={np.sum(mask)}")
        
        cap.release()
        return masks
    
    except Exception as e:
        logging.error(f"Error creating test masks: {str(e)}")
        traceback.print_exc()
        return []

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
        
        logging.info(f"Successfully prepared mask for frame {frame_number}, type={type(mask_data)}")
        return mask_data
    
    except Exception as e:
        logging.error(f"Error preparing mask for frame {frame_number}: {str(e)}")
        traceback.print_exc()
        return None

def upload_masks_to_mdai(client, masks):
    """Force upload masks to MD.ai"""
    print_header(f"Forcing upload of {len(masks)} masks to MD.ai")
    
    successful = 0
    failed = 0
    
    for frame_number, mask in masks:
        try:
            # Prepare the mask for MD.ai
            mask_data = prepare_mask_for_mdai(mask, frame_number)
            
            if not mask_data:
                failed += 1
                continue
            
            # Create annotation with correct structure
            annotation = {
                'labelId': LABEL_ID_FLUID_OF,
                'StudyInstanceUID': STUDY_UID,
                'SeriesInstanceUID': SERIES_UID,
                'frameNumber': int(frame_number),
                'data': mask_data,  # Important: This is the key field for MD.ai
                'groupId': LABEL_ID_MACHINE_GROUP,
                'note': f"Force uploaded at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            }
            
            # Log detailed debug info
            logging.debug(f"Annotation data for frame {frame_number}:")
            logging.debug(f"  labelId: {annotation['labelId']}")
            logging.debug(f"  StudyInstanceUID: {annotation['StudyInstanceUID'][:15]}...")
            logging.debug(f"  SeriesInstanceUID: {annotation['SeriesInstanceUID'][:15]}...")
            logging.debug(f"  frameNumber: {annotation['frameNumber']}")
            logging.debug(f"  groupId: {annotation['groupId']}")
            logging.debug(f"  data type: {type(annotation['data'])}")
            
            # Upload to MD.ai
            response = client.import_annotations(
                annotations=[annotation],
                project_id=PROJECT_ID,
                dataset_id=DATASET_ID
            )
            
            if response and len(response) > 0:
                logging.error(f"Upload failed for frame {frame_number}: {response}")
                failed += 1
            else:
                logging.info(f"Successfully uploaded mask for frame {frame_number}")
                successful += 1
                
            # Sleep briefly to avoid rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"Error uploading mask for frame {frame_number}: {str(e)}")
            traceback.print_exc()
            failed += 1
    
    logging.info(f"Upload results: {successful} successful, {failed} failed")
    return successful, failed

def verify_mask_exists_on_mdai(client, frame_number):
    """Verify if a mask for a specific frame exists on MD.ai"""
    try:
        # Use a direct API call to get annotations
        project_with_annotations = client.project(
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            annotations_only=True
        )
        
        # Look for our specific annotation
        for ann in project_with_annotations.annotations:
            if (ann.StudyInstanceUID == STUDY_UID and 
                ann.SeriesInstanceUID == SERIES_UID and
                ann.frameNumber == frame_number and
                ann.labelId == LABEL_ID_FLUID_OF):
                
                logging.info(f"✓ Found annotation for frame {frame_number} on MD.ai")
                return True
        
        logging.warning(f"✗ No annotation found for frame {frame_number} on MD.ai")
        return False
    
    except Exception as e:
        logging.error(f"Error verifying annotation: {str(e)}")
        return False

def main():
    print_header("STARTING FORCED UPLOAD PROCESS")
    
    # Initialize MD.ai client
    client = initialize_mdai_client()
    if not client:
        logging.error("Failed to initialize MD.ai client")
        sys.exit(1)
    
    # Delete existing annotations
    delete_existing_annotations(client)
    
    # Test upload with a simple mask first
    logging.info("Testing MD.ai upload with basic test masks")
    test_masks = create_test_masks()
    if test_masks:
        successful, failed = upload_masks_to_mdai(client, test_masks[:1])  # Upload first test mask
        if successful == 0:
            logging.error("Failed to upload test mask. Checking MD.ai permissions...")
            sys.exit(1)
        
        # Verify the test upload
        if not verify_mask_exists_on_mdai(client, test_masks[0][0]):
            logging.error("Test mask verification failed. The mask was uploaded but cannot be retrieved.")
            logging.error("This suggests a permissions or visibility issue in MD.ai.")
            sys.exit(1)
    
    # Generate and upload real algorithm masks
    logging.info("Now running the algorithm to generate real masks for upload")
    algorithm_masks = run_algorithm_with_mask_interception()
    
    if not algorithm_masks:
        logging.error("No algorithm-generated masks to upload")
        sys.exit(1)
    
    logging.info(f"Successfully extracted {len(algorithm_masks)} algorithm-generated masks")
    successful, failed = upload_masks_to_mdai(client, algorithm_masks)
    
    # Final verification
    if successful > 0:
        print_header("UPLOAD SUCCESSFUL!")
        logging.info(f"Successfully uploaded {successful} out of {len(algorithm_masks)} masks")
        logging.info("Check MD.ai to see the uploaded annotations")
        
        # Verify a sample of uploads
        verify_frame = algorithm_masks[len(algorithm_masks)//2][0]  # Verify middle frame
        if verify_mask_exists_on_mdai(client, verify_frame):
            logging.info("Verification confirmed that masks are visible in MD.ai")
        else:
            logging.warning("Verification couldn't find uploaded masks in MD.ai - check permissions")
    else:
        print_header("UPLOAD FAILED!")
        logging.error("Failed to upload any algorithm-generated masks")
    
    print_header("PROCESS COMPLETED")

if __name__ == "__main__":
    main() 