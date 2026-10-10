from pathlib import Path
import random

class LungDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.class_number = 1

        self.image_dir = self.root / "CXR_png"
        self.mask_dir = self.root / "masks"
        
        self.samples = []

        for image_path in self.image_dir.glob("*.png"):
            mask_path = self.mask_dir / f"{image_path.stem}_mask.png"

            if mask_path.exists():
                self.samples.append((image_path, mask_path))
            else:
                print("Mask not found for:", image_path)

    def sample_episode(self, episode_idx, n_shot):
        selected = random.sample(self.samples, n_shot + 1)

        return selected
