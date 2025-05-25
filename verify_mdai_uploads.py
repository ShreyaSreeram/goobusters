#!/usr/bin/env python3
import os
import sys
import numpy as np
import mdai
from dotenv import load_dotenv
import json
import time
import argparse

# Load environment variables
load_dotenv('.env')

def print_header(text):
    """Print a nicely formatted header"""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80)

def load_env_variables():
    """Load and verify all required environment variables"""
    print_header("CHECKING ENVIRONMENT VARIABLES")
    
    env_vars = {
        'MDAI_TOKEN': os.getenv('MDAI_TOKEN'),
        'DOMAIN': os.getenv('DOMAIN', 'annotate.md.ai'),
        'PROJECT_ID': os.getenv('PROJECT_ID'),
        'DATASET_ID': os.getenv('DATASET_ID'),
        'LABEL_ID_FLUID_OF': os.getenv('LABEL_ID_FLUID_OF'),
        'LABEL_ID_NO_FLUID': os.getenv('LABEL_ID_NO_FLUID'),
        'LABEL_ID_MACHINE_GROUP': os.getenv('LABEL_ID_MACHINE_GROUP')
    }
    
    # Check for missing variables
    missing = [key for key, value in env_vars.items() if not value]
    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        print("Please check your .env file")
        return None
    
    # Print info for all variables (mask sensitive info)
    for key, value in env_vars.items():
        if key == 'MDAI_TOKEN':
            print(f"{key}: {value[:5]}..." if value else f"{key}: Missing!")
        else:
            print(f"{key}: {value}" if value else f"{key}: Missing!")
    
    return env_vars

def connect_to_mdai(env_vars):
    """Attempt to connect to MD.ai and verify connection"""
    print_header("CONNECTING TO MD.AI")
    
    try:
        client = mdai.Client(domain=env_vars['DOMAIN'], access_token=env_vars['MDAI_TOKEN'])
        print("✓ Successfully connected to MD.ai")
        
        # Verify project access
        project_info = client.project(env_vars['PROJECT_ID'])
        print(f"✓ Successfully accessed project: {env_vars['PROJECT_ID']}")
        
        # Safely try to get project name if available
        try:
            if hasattr(project_info, 'name'):
                print(f"Project name: {project_info.name}")
        except:
            pass
        
        return client
    except Exception as e:
        print(f"ERROR: Failed to connect to MD.ai: {str(e)}")
        return None

def check_annotations(client, env_vars, study_uid=None, series_uid=None):
    """Check for existing annotations in the project/dataset"""
    print_header("CHECKING ANNOTATIONS")
    
    if study_uid and series_uid:
        print(f"Checking annotations for specific study/series:")
        print(f"Study UID: {study_uid}")
        print(f"Series UID: {series_uid}")
    else:
        print("Checking annotations for the entire dataset")
    
    try:
        # Get project with annotations
        project = client.project(
            project_id=env_vars['PROJECT_ID'],
            dataset_id=env_vars['DATASET_ID'],
            annotations_only=True
        )
        
        # Count total annotations
        total_annotations = 0
        fluid_annotations = 0
        no_fluid_annotations = 0
        machine_annotations = 0
        
        for ann in project.annotations:
            if study_uid and series_uid:
                if ann.StudyInstanceUID != study_uid or ann.SeriesInstanceUID != series_uid:
                    continue
            
            total_annotations += 1
            
            if ann.labelId == env_vars['LABEL_ID_FLUID_OF']:
                fluid_annotations += 1
                if hasattr(ann, 'groupId') and ann.groupId == env_vars['LABEL_ID_MACHINE_GROUP']:
                    machine_annotations += 1
            
            if ann.labelId == env_vars['LABEL_ID_NO_FLUID']:
                no_fluid_annotations += 1
        
        print(f"Total annotations: {total_annotations}")
        print(f"Free fluid annotations: {fluid_annotations}")
        print(f"No fluid annotations: {no_fluid_annotations}")
        print(f"Machine-generated annotations: {machine_annotations}")
        
        return total_annotations
    except Exception as e:
        print(f"ERROR: Failed to check annotations: {str(e)}")
        return 0

def upload_test_annotation(client, env_vars, study_uid, series_uid, frame_num):
    """Upload a test annotation to verify upload capability"""
    print_header(f"UPLOADING TEST ANNOTATION TO FRAME {frame_num}")
    
    try:
        # Create a test mask (small circle)
        height, width = 480, 640  # Typical ultrasound dimensions
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        # Create a small circle
        radius = 20
        mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2).astype(np.uint8)
        
        # Convert to MD.ai format
        mask_data = mdai.common_utils.convert_mask_data(mask)
        
        # Create the annotation
        annotation = {
            'labelId': env_vars['LABEL_ID_FLUID_OF'],
            'StudyInstanceUID': study_uid,
            'SeriesInstanceUID': series_uid,
            'frameNumber': int(frame_num),
            'data': mask_data,
            'groupId': env_vars['LABEL_ID_MACHINE_GROUP'],
            'note': f"Test annotation created at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        }
        
        print(f"Uploading test annotation to study {study_uid}, series {series_uid}, frame {frame_num}")
        response = client.import_annotations(
            annotations=[annotation],
            project_id=env_vars['PROJECT_ID'],
            dataset_id=env_vars['DATASET_ID']
        )
        
        if response and len(response) > 0:
            print(f"❌ Upload failed: {response}")
            return False
        else:
            print("✓ Test annotation uploaded successfully!")
            return True
    
    except Exception as e:
        print(f"ERROR: Failed to upload test annotation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Verify MD.ai upload capabilities")
    parser.add_argument("--study", type=str, help="Study UID to check annotations for")
    parser.add_argument("--series", type=str, help="Series UID to check annotations for")
    parser.add_argument("--frame", type=int, default=1, help="Frame number to test upload with")
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading test annotation")
    args = parser.parse_args()
    
    # Load environment variables
    env_vars = load_env_variables()
    if not env_vars:
        sys.exit(1)
    
    # Connect to MD.ai
    client = connect_to_mdai(env_vars)
    if not client:
        sys.exit(1)
    
    # Check annotations
    if args.study and args.series:
        annotation_count = check_annotations(client, env_vars, args.study, args.series)
    else:
        annotation_count = check_annotations(client, env_vars)
    
    # Upload test annotation if requested
    if not args.skip_upload and args.study and args.series:
        upload_success = upload_test_annotation(client, env_vars, args.study, args.series, args.frame)
        
        if upload_success:
            print_header("UPLOAD TEST SUCCESSFUL")
            print("Your MD.ai configuration is working correctly for uploads.")
            print("You should now be able to run the algorithm with --upload flag and see masks appear.")
        else:
            print_header("UPLOAD TEST FAILED")
            print("There might be issues with your MD.ai configuration or permissions.")
            print("Check the error messages above for more details.")
    elif not args.skip_upload:
        print("\nSkipping test upload - need both --study and --series arguments.")
        print("Run again with: python verify_mdai_uploads.py --study YOUR_STUDY_UID --series YOUR_SERIES_UID")
    
    print_header("DIAGNOSTICS COMPLETE")

if __name__ == "__main__":
    main() 