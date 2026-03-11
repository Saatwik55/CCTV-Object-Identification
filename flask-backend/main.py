import os
import shutil
import hashlib
from glob import glob
from tqdm import tqdm
import torch
from PIL import Image
import cv2

from yolo import YOLODetector
from compare import batch_load_and_embed as batch_load_and_embed
from face import batch_load_and_embed as fblae
import capture

# -----------------------------
# Config
# -----------------------------
lookup_class = {"humans": 0, "bikes": 3, "cars": 2}

WORKSPACE_ROOT = "workspace"
REFERENCE_DIR = "reference"
REFERENCE_CROPS_DIR = "reference_crops"
OUTPUT_DIR = "output"

# Global state for caching
detector = None
current_video_path = None
current_frames_dir = None
current_candidates_dir = None

# -----------------------------
# Utils
# -----------------------------
def get_fast_hash(video_path, target_class=None):
    stats = os.stat(video_path)
    ident_str = f"{os.path.basename(video_path)}_{stats.st_size}"
    if target_class:
        ident_str += f"_{target_class}"
    return hashlib.sha256(ident_str.encode()).hexdigest()[:16]

def is_image_too_small(image_path, model_type):
    try:
        img = Image.open(image_path)
        w, h = img.size
        if model_type == "dino":
            return w < 128 or h < 128
        elif model_type == "face":
            return w < 64 or h < 64
    except Exception:
        return True
    return False

# -----------------------------
# Init
# -----------------------------
def initialize(target_class_name, fps, video_path=None):
    global detector, current_video_path, current_frames_dir, current_candidates_dir
    
    if video_path:
        current_video_path = video_path
        vid_id = get_fast_hash(video_path)
        
        current_frames_dir = os.path.join(WORKSPACE_ROOT, vid_id, "frames")
        current_candidates_dir = os.path.join(WORKSPACE_ROOT, vid_id, f"candidates_{target_class_name}")
        
        os.makedirs(current_candidates_dir, exist_ok=True)
        target_folder = current_candidates_dir
    else:
        target_folder = "."

    target_class_index = lookup_class[target_class_name]
    detector = YOLODetector(
        output_folder=target_folder,
        targetClass=target_class_index,
        buffer_interval=fps
    )



def generate_frames(video_path):
    global current_video_path, current_frames_dir
    current_video_path = video_path

    vid_id = get_fast_hash(video_path)
    current_frames_dir = os.path.join(WORKSPACE_ROOT, vid_id, "frames")
    os.makedirs(current_frames_dir, exist_ok=True)

    # Always read FPS from video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if not os.listdir(current_frames_dir):
        print(f"[INFO] Extracting frames to: {current_frames_dir}")
        capture.extract_frames(video_path, current_frames_dir)
    else:
        print(f"[INFO] Using cached frames from {current_frames_dir}")

    return fps

def process_reference_images():
    print(f"[INFO] Processing reference images in {REFERENCE_DIR}")
    detector.process_reference_folder(
        ref_folder=REFERENCE_DIR,
        reference_crops_folder=REFERENCE_CROPS_DIR
    )
    print(f"[INFO] Reference crops saved to {REFERENCE_CROPS_DIR}")

def process_all_frames():
    global current_candidates_dir, detector
    
    target_class_name = [k for k, v in lookup_class.items() if v == detector.targetClass][0]
    
    current_candidates_dir = os.path.join(WORKSPACE_ROOT, get_fast_hash(current_video_path), f"candidates_{target_class_name}")
    os.makedirs(current_candidates_dir, exist_ok=True)
    
    detector.output_folder = current_candidates_dir

    if not os.listdir(current_candidates_dir):
        frame_paths = sorted(glob(os.path.join(current_frames_dir, "*.jpg")))
        if not frame_paths:
            print("[WARN] No frames found.")
            return

        print(f"[INFO] Processing {len(frame_paths)} video frames...")
        for frame in tqdm(frame_paths, desc="YOLO on frames"):
            detector.process_image(frame)

        print(f"[INFO] Candidate crops saved to {current_candidates_dir}")
    else:
        print(f"[INFO] Using cached candidates from {current_candidates_dir}")

def prepare_match_inputs(reference_dir, candidate_dir, model_type):
    reference_images = sorted(glob(os.path.join(reference_dir, "*.jpg")))
    candidate_images = sorted(glob(os.path.join(candidate_dir, "*.jpg")))

    if not reference_images or not candidate_images:
        print("[WARN] No reference or candidate images.")
        return None, None

    reference_images = [
        p for p in reference_images
        if not is_image_too_small(p, model_type)
    ]
    candidate_images = [
        p for p in candidate_images
        if not is_image_too_small(p, model_type)
    ]

    return reference_images, candidate_images

def vectorized_match_and_save(
    ref_embeds,
    candidate_embeds,
    candidate_paths,
    prefix,
    top_k=10,
    threshold=0.0
):
    # [C, R]
    scores = candidate_embeds @ ref_embeds.T
    mean_scores = scores.mean(dim=1)

    results = [
        (path, score.item())
        for path, score in zip(candidate_paths, mean_scores)
        if score.item() >= threshold
    ]

    if not results:
        print(f"[INFO] No {prefix.upper()} matches above threshold.")
        return False

    results.sort(key=lambda x: x[1], reverse=True)
    top_matches = results[:top_k]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for rank, (img_path, score) in enumerate(top_matches, start=1):
        base = os.path.basename(img_path)
        new_name = f"{prefix}_{rank:02d}_sim_{score:.3f}_{base}"
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, new_name))

    print(f"[INFO] Top {len(top_matches)} {prefix.upper()} matches saved.")
    return True

def find_best_face_matches(top_k=10, fallback_to_dino=True):
    ref_paths, cand_paths = prepare_match_inputs(
        REFERENCE_CROPS_DIR, current_candidates_dir, model_type="face"
    )
    if not ref_paths:
        return

    ref_map = fblae(ref_paths)
    cand_map = fblae(cand_paths)
    cand_paths = [p for p in cand_paths if p in cand_map]
    ref_paths = [p for p in ref_paths if p in ref_map]
    if not ref_paths or not cand_paths:
        return
    ref_embeds = torch.cat([ref_map[p] for p in ref_paths], dim=0)  # [R, D]

    cand_embeds = torch.cat([cand_map[p] for p in cand_paths], dim=0)  # [C, D]
    success = vectorized_match_and_save(
        ref_embeds,
        cand_embeds,
        cand_paths,
        prefix="face",
        top_k=top_k
    )

    if not success and fallback_to_dino:
        print("[INFO] No confident face match found. Falling back to DINO...")
        find_best_dino_matches(top_k=top_k)

def find_best_dino_matches(top_k=10):
    ref_paths, cand_paths = prepare_match_inputs(
        REFERENCE_CROPS_DIR,
        current_candidates_dir,
        model_type="dino"
    )
    if not ref_paths:
        return

    ref_map = batch_load_and_embed(ref_paths)
    cand_map = batch_load_and_embed(cand_paths)
    cand_paths = [p for p in cand_paths if p in cand_map]
    ref_paths = [p for p in ref_paths if p in ref_map]
    if not ref_paths or not cand_paths:
        return
    ref_embeds = torch.cat([ref_map[p] for p in ref_paths], dim=0)  # [R, D]

    cand_embeds = torch.cat([cand_map[p] for p in cand_paths], dim=0)  # [C, D]
    
    
    
    vectorized_match_and_save(
        ref_embeds,
        cand_embeds,
        cand_paths,
        prefix="dino",
        top_k=top_k
    )

def find_best_matches(target_class_name, top_k=10):
    if target_class_name == "humans":
        find_best_face_matches(top_k=top_k)
    else:
        find_best_dino_matches(top_k=top_k)

def cleanup():
    shutil.rmtree(REFERENCE_CROPS_DIR, ignore_errors=True)
    shutil.rmtree(REFERENCE_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    print("[INFO] Cleanup complete. Workspace cache preserved.")