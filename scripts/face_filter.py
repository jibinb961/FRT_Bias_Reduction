import os
import shutil
import cv2
import argparse
import numpy as np
from PIL import Image
import mediapipe as mp
import time

class FaceFilterer:
    """
    Filters images containing clear, front-facing faces without occlusions
    like glasses, hats or hands covering the face.
    """
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Face detection model with high min_detection_confidence for quality
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full range detection
            min_detection_confidence=0.7  # High confidence threshold
        )
        
        # Face mesh for detailed face landmark detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,  # We only care about the main face
            refine_landmarks=True,
            min_detection_confidence=0.7
        )
        
        # Cascade classifiers for additional checks
        try:
            # For detecting glasses and other occlusions
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        except Exception as e:
            print(f"Warning: Could not load cascade classifiers: {e}")
            self.eye_cascade = None
            self.glasses_cascade = None
    
    def is_good_face(self, image_path):
        """
        Checks if the image contains a clear, front-facing face without occlusions.
        
        Args:
            image_path (str): Path to the image to analyze
            
        Returns:
            bool: True if image contains a good face, False otherwise
            dict: Additional info about why image was accepted/rejected
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return False, {"reason": "Could not read image"}
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape
            
            # Run face detection
            detection_results = self.face_detection.process(image_rgb)
            
            # Check if any faces were detected
            if not detection_results.detections:
                return False, {"reason": "No face detected"}
            
            # Get the first (presumably largest/most prominent) face
            face = detection_results.detections[0]
            
            # Calculate face size relative to image
            bbox = face.location_data.relative_bounding_box
            face_width = bbox.width * w
            face_height = bbox.height * h
            
            # Check if face is large enough (occupies at least 20% of image)
            face_area_ratio = (face_width * face_height) / (w * h)
            if face_area_ratio < 0.2:
                return False, {"reason": "Face too small", "ratio": face_area_ratio}
            
            # Check face orientation (should be relatively front-facing)
            # Front-facing faces typically have symmetric bounding boxes
            aspect_ratio = face_width / face_height
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                return False, {"reason": "Face not front-facing", "aspect_ratio": aspect_ratio}
            
            # Run face mesh to get detailed landmarks
            mesh_results = self.face_mesh.process(image_rgb)
            
            # If no face mesh landmarks, likely not a clear face
            if not mesh_results.multi_face_landmarks:
                return False, {"reason": "No clear facial landmarks"}
            
            # Get face landmarks
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            
            # Check if both eyes are visible (key for front-facing detection)
            # Left eye landmarks (around indices 130-159)
            # Right eye landmarks (around indices 360-386)
            left_eye_landmarks = [landmarks[i] for i in range(130, 160)]
            right_eye_landmarks = [landmarks[i] for i in range(360, 387)]
            
            # Check eye visibility by landmarks spread
            left_eye_x = [lm.x for lm in left_eye_landmarks]
            right_eye_x = [lm.x for lm in right_eye_landmarks]
            
            # If eyes are not well-separated, face might not be front-facing
            eye_distance = abs(sum(right_eye_x)/len(right_eye_x) - sum(left_eye_x)/len(left_eye_x))
            if eye_distance < 0.05:  # Eyes too close, likely profile view
                return False, {"reason": "Face appears to be in profile view"}
            
            # Check for occlusions like glasses (if cascade available)
            if self.eye_cascade is not None:
                # Get face region
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                face_region = image[y:y+height, x:x+width]
                
                # Check if eyes are detectable
                eyes = self.eye_cascade.detectMultiScale(face_region)
                if len(eyes) < 2:
                    # If eyes aren't detected, check if glasses are obscuring them
                    glasses = self.glasses_cascade.detectMultiScale(face_region)
                    if len(glasses) > 0:
                        return False, {"reason": "Glasses detected"}
                    # If neither eyes nor glasses detected, face might be occluded
                    else:
                        return False, {"reason": "Eyes not clearly visible"}
            
            # Calculate symmetry score (front-facing faces are more symmetrical)
            symmetry_score = self._calculate_symmetry(landmarks, w, h)
            if symmetry_score > 0.2:  # Higher value means less symmetrical
                return False, {"reason": "Face not symmetrical enough", "score": symmetry_score}
            
            # Image passes all checks
            return True, {"reason": "Good face image", "confidence": face.score[0]}
            
        except Exception as e:
            return False, {"reason": f"Error processing image: {str(e)}"}
    
    def _calculate_symmetry(self, landmarks, width, height):
        """Calculate how symmetrical the face is - front facing faces are more symmetrical"""
        # Get 2D points from landmarks
        points = [(lm.x * width, lm.y * height) for lm in landmarks]
        points = np.array(points)
        
        # Find vertical midline of face
        left_points = points[:len(points)//2]
        right_points = points[len(points)//2:]
        midline_x = np.mean(points[:, 0])
        
        # Calculate distances from each landmark to the midline
        left_distances = [abs(p[0] - midline_x) for p in left_points]
        right_distances = [abs(p[0] - midline_x) for p in right_points]
        
        # Compare symmetry (if perfectly symmetrical, difference would be 0)
        # Only compare the same number of points
        min_len = min(len(left_distances), len(right_distances))
        symmetry_diff = np.mean(np.abs(np.array(left_distances[:min_len]) - np.array(right_distances[:min_len])))
        
        # Normalize by face width
        symmetry_score = symmetry_diff / width
        return symmetry_score

def process_directory(source_dir, target_dir, filterer, max_files=None, verbose=False):
    """
    Process all images in source_dir and copy good face images to target_dir
    
    Args:
        source_dir (str): Directory containing source images
        target_dir (str): Directory to copy good face images to
        filterer (FaceFilterer): Initialized face filterer
        max_files (int, optional): Maximum number of files to process
        verbose (bool): Whether to print detailed info
    
    Returns:
        tuple: (total_processed, total_copied, failed)
    """
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(valid_extensions)]
    
    # Limit number if specified
    if max_files is not None:
        image_files = image_files[:max_files]
    
    total_files = len(image_files)
    copied_files = 0
    failed_files = 0
    start_time = time.time()
    
    print(f"Processing {total_files} images...")
    
    # Process each file
    for i, filename in enumerate(image_files):
        source_path = os.path.join(source_dir, filename)
        
        # Progress update
        if i % 10 == 0 and i > 0:
            elapsed = time.time() - start_time
            files_per_sec = i / elapsed if elapsed > 0 else 0
            print(f"Processed {i}/{total_files} images ({files_per_sec:.2f} images/sec)")
        
        try:
            # Check if it's a good face image
            is_good, info = filterer.is_good_face(source_path)
            
            if verbose:
                print(f"Image {filename}: {'PASS' if is_good else 'FAIL'} - {info['reason']}")
            
            if is_good:
                # Copy to target directory
                target_path = os.path.join(target_dir, filename)
                shutil.copy2(source_path, target_path)
                copied_files += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed_files += 1
    
    # Final stats
    elapsed = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Processed {total_files} images in {elapsed:.2f} seconds")
    print(f"Copied {copied_files} good face images to {target_dir}")
    print(f"Failed to process {failed_files} images")
    
    return total_files, copied_files, failed_files

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Filter and copy good face images')
    parser.add_argument('source_dir', help='Directory containing source images')
    parser.add_argument('target_dir', help='Directory to copy good face images to')
    parser.add_argument('--max', type=int, help='Maximum number of files to process')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed information')
    args = parser.parse_args()
    
    # Initialize face filterer
    filterer = FaceFilterer()
    
    # Process directory
    process_directory(args.source_dir, args.target_dir, filterer, args.max, args.verbose)

if __name__ == "__main__":
    main() 