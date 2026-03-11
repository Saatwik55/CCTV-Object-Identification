import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import os

# -----------------------------
# 1. Hardware & Model Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# MTCNN for face detection/alignment
mtcnn = MTCNN(
    image_size=160, 
    margin=20, 
    keep_all=False, 
    device=device, 
    post_process=True
)

# InceptionResnetV1 for generating 512-D embeddings
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Global cache to prevent redundant processing
FACE_EMB_CACHE = {}

# -----------------------------
# 2. Dataset & Collate 
# -----------------------------
class FaceImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            # Load as RGB; MTCNN handles the resizing internally
            img = Image.open(path).convert("RGB")
            return img, path
        except Exception as e:
            print(f"[ERROR] Could not load {path}: {e}")
            return None, path

def pil_collate_fn(batch):
    """
    Groups PIL images and paths into lists. 
    Required because input images often have different dimensions.
    """
    imgs = [item[0] for item in batch if item[0] is not None]
    paths = [item[1] for item in batch if item[0] is not None]
    return imgs, paths

# -----------------------------
# 3. Batch Processing Engine
# -----------------------------
def batch_load_and_embed(image_paths, batch_size=32, num_workers=4):
    """
    Processes a list of image paths. 
    Handles variable image sizes during detection and batches during embedding.
    """
    uncached_paths = [p for p in image_paths if p not in FACE_EMB_CACHE]

    if uncached_paths:
        dataset = FaceImageDataset(uncached_paths)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False, 
            collate_fn=pil_collate_fn
        )

        for imgs, paths in loader:
            if not imgs:
                continue

            # STEP A: Individual Detection (The Fix)
            # We process images individually to avoid "equal-dimension" errors.
            # MTCNN will output uniform 160x160 tensors.
            try:
                faces = [mtcnn(img) for img in imgs]
            except Exception as e:
                print(f"[WARN] MTCNN processing failed for a batch: {e}")
                continue

            # STEP B: Filter & Stack
            valid_faces = []
            valid_paths = []
            for face, path in zip(faces, paths):
                if face is not None:
                    valid_faces.append(face)
                    valid_paths.append(path)

            if not valid_faces:
                continue

            # STEP C: Batched Embedding (The Speedup)
            # Since all faces are now 160x160, we can stack them safely.
            face_tensors = torch.stack(valid_faces).to(device)
            
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                # Generate embeddings for the whole batch at once
                embeddings = model(face_tensors)
                embeddings = normalize(embeddings, dim=-1).cpu()

            # STEP D: Update Cache
            for p, e in zip(valid_paths, embeddings):
                FACE_EMB_CACHE[p] = e.unsqueeze(0)

    return {
        p: FACE_EMB_CACHE[p] 
        for p in image_paths 
        if p in FACE_EMB_CACHE
    }

# -----------------------------
# 4. Example Usage
# -----------------------------
if __name__ == "__main__":
    # Replace with your actual image paths
    test_paths = ["path/to/face1.jpg", "path/to/face2.png"] 
    
    # Filter for existing files only
    test_paths = [p for p in test_paths if os.path.exists(p)]
    
    if test_paths:
        results = batch_load_and_embed(test_paths)
        print(f"[INFO] Successfully embedded {len(results)} images.")
    else:
        print("[INFO] No valid image paths provided for testing.")