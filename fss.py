from pathlib import Path
import random

class FSSDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.class_number = 1000

        self.samples = {cls: [] for cls in range(0, self.class_number)}

        class_names = [path.name for path in self.root.iterdir() if path.is_dir()]

        for i, class_name in enumerate(class_names):
            class_dir = self.root / class_name

            for image_path in class_dir.glob("*.jpg"):
                mask_path = image_path.with_suffix(".png")

                if mask_path.exists():
                    self.samples[i].append((image_path, mask_path))
                else:
                    print("Mask not found for:", image_path)

    def sample_episode(self, episode_idx, n_shot):
            class_id = episode_idx % self.class_number
            selected = random.sample(self.samples[class_id], n_shot + 1)
    
            return selected
