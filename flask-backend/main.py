import os
import shutil
from glob import glob
from tqdm import tqdm
import torch
from PIL import Image

from yolo import YOLODetector
from compare import load_and_embed, batch_load_and_embed
from face import embed_face
import capture

# -----------------------------
# Config
# -----------------------------
lookup_class = {"humans": 0, "bikes": 3, "cars": 2}

FRAMES_DIR = "frames"
CANDIDATES_DIR = "candidates"
REFERENCE_DIR = "reference"
REFERENCE_CROPS_DIR = "reference_crops"
OUTPUT_DIR = "output"

detector = None

# -----------------------------
# Utils
# -----------------------------
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
def initialize(target_class_name, fps):
    global detector
    target_class_index = lookup_class[target_class_name]
    detector = YOLODetector(
        output_folder=CANDIDATES_DIR,
        targetClass=target_class_index,
        buffer_interval=fps
    )

def generate_frames(video_path):
    return capture.extract_frames(video_path, FRAMES_DIR)

def process_reference_images():
    print(f"[INFO] Processing reference images in {REFERENCE_DIR}")
    detector.process_reference_folder(
        ref_folder=REFERENCE_DIR,
        reference_crops_folder=REFERENCE_CROPS_DIR
    )
    print(f"[INFO] Reference crops saved to {REFERENCE_CROPS_DIR}")

def process_all_frames():
    frame_paths = sorted(glob(os.path.join(FRAMES_DIR, "*.jpg")))
    if not frame_paths:
        print("[WARN] No frames found.")
        return

    print(f"[INFO] Processing {len(frame_paths)} video frames...")
    for frame in tqdm(frame_paths, desc="YOLO on frames"):
        detector.process_image(frame)

    print(f"[INFO] Candidate crops saved to {CANDIDATES_DIR}")

def prepare_match_inputs(reference_dir, candidate_dir, model_type):
    reference_images = sorted(glob(os.path.join(reference_dir, "*.jpg")))
    candidate_images = sorted(glob(os.path.join(candidate_dir, "*.jpg")))

    if not reference_images or not candidate_images:
        print("[WARN] No reference or candidate images.")
        return None, None

    # Filter small images early
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
    """
    ref_embeds: Tensor [R, D]
    candidate_embeds: Tensor [C, D]
    """

    # [C, R]
    scores = candidate_embeds @ ref_embeds.T
    max_scores, _ = scores.max(dim=1)

    results = [
        (path, score.item())
        for path, score in zip(candidate_paths, max_scores)
        if score.item() >= threshold
    ]

    if not results:
        print(f"[INFO] No {prefix.upper()} matches above threshold.")
        return False

    results.sort(key=lambda x: x[1], reverse=True)
    top_matches = results[:top_k]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for rank, (img_path, _) in enumerate(top_matches, start=1):
        new_name = f"{prefix}_{rank:02d}_{os.path.basename(img_path)}"
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, new_name))

    print(f"[INFO] Top {len(top_matches)} {prefix.upper()} matches saved.")
    return True

def find_best_face_matches(top_k=10, fallback_to_dino=True):
    ref_paths, cand_paths = prepare_match_inputs(
        REFERENCE_CROPS_DIR, CANDIDATES_DIR, model_type="face"
    )
    if not ref_paths:
        return

    ref_map = batch_load_and_embed(ref_paths)
    ref_embeds = torch.cat([ref_map[p] for p in ref_paths], dim=0)  # [R, D]

    cand_map = batch_load_and_embed(cand_paths)
    cand_embeds = torch.cat([cand_map[p] for p in cand_paths], dim=0)  # [C, D]

    success = vectorized_match_and_save(
        ref_embeds,
        cand_embeds,
        cand_paths,
        prefix="face",
        top_k=top_k,
        threshold=0.60
    )

    if not success and fallback_to_dino:
        print("[INFO] No confident face match found. Falling back to DINO...")
        find_best_dino_matches(top_k=top_k)

def find_best_dino_matches(top_k=10):
    ref_paths, cand_paths = prepare_match_inputs(
        REFERENCE_CROPS_DIR,
        CANDIDATES_DIR,
        model_type="dino"
    )
    if not ref_paths:
        return

    ref_map = batch_load_and_embed(ref_paths)
    ref_embeds = torch.cat([ref_map[p] for p in ref_paths], dim=0)  # [R, D]

    cand_map = batch_load_and_embed(cand_paths)
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
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    shutil.rmtree(CANDIDATES_DIR, ignore_errors=True)
    shutil.rmtree(REFERENCE_CROPS_DIR, ignore_errors=True)
    shutil.rmtree(REFERENCE_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)