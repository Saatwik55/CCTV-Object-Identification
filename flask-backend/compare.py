import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
from PIL import Image
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model(
    "vit_base_patch14_dinov2.lvd142m",
    pretrained=True
)
model.eval()
model.to(device)

transform = T.Compose([
    T.Resize((518, 518)),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

DINO_EMB_CACHE = {}

def load_and_embed(image_path):
    if image_path in DINO_EMB_CACHE:
        return DINO_EMB_CACHE[image_path]

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            emb = model.forward_features(image_tensor)

        if isinstance(emb, dict):
            emb = emb["x_norm_clstoken"]

        emb = normalize(emb.flatten(1), dim=-1)

    emb = emb.cpu()
    DINO_EMB_CACHE[image_path] = emb
    return emb

class DinoImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), path

def batch_load_and_embed(image_paths, batch_size=16, num_workers=4):
    """
    image_paths: List[str]
    returns: Dict[path -> embedding tensor (1, D) on CPU]
    """
    uncached_paths = [p for p in image_paths if p not in DINO_EMB_CACHE]

    if not uncached_paths:
        return {p: DINO_EMB_CACHE[p] for p in image_paths}

    dataset = DinoImageDataset(uncached_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model.eval()

    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                emb = model.forward_features(images)

            if isinstance(emb, dict):
                emb = emb["x_norm_clstoken"]

            emb = normalize(emb.flatten(1), dim=-1)

            for p, e in zip(paths, emb):
                DINO_EMB_CACHE[p] = e.unsqueeze(0).cpu()

    return {p: DINO_EMB_CACHE[p] for p in image_paths if p in DINO_EMB_CACHE}