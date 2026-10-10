from pathlib import Path
import random

class DeepglobeDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.class_number = 6

        self.samples = {cls: [] for cls in range(1, self.class_number + 1)}

        for cls in range(1, self.class_number + 1):
            class_dir = self.root / f"{cls}/test"
            groundtruth_dir = class_dir / "groundtruth"
            origin_dir = class_dir / "origin"

            for image_path in origin_dir.glob("*.jpg"):
                mask_path = groundtruth_dir / f"{image_path.stem}.png"

                if mask_path.exists():
                    self.samples[cls].append((image_path, mask_path))
                else:
                    print("Mask not found for:", image_path)

    def sample_episode(self, episode_idx, n_shot):
            class_id = episode_idx % self.class_number + 1
            selected = random.sample(self.samples[class_id], n_shot + 1)
    
            return selected
