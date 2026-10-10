from transformers import AutoModel
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import torch
import torch.nn.functional as F
import random
import argparse
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from lung import LungDataset
from deepglobe import DeepglobeDataset
from fss import FSSDataset

random.seed(47)

model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained(model_name)
model = model.to(device)
model.eval()

image_transform = transforms.Compose([
    transforms.Resize((480, 480), antialias = True), # 480x480 because HERA and RSRM experiments had such resize.
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


mask_transform = transforms.Compose([
    transforms.Resize((480, 480), interpolation = InterpolationMode.NEAREST), # NEAREST because mask can't have any value except 0 and 1.
    transforms.PILToTensor()
])


def preprocess_image(image):
    """Preprocess image for passing to model"""

    image = image_transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    return image


def preprocess_mask(mask):
    """Preprocess mask for passing to protyping and evaluation"""

    mask = mask_transform(mask)

    mask = (mask > 0).float() # ? просто ToTensor использовать вместо PILToTensor

    mask = mask.to(device)

    return mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test with DINOv3 baseline"
    )

    parser.add_argument('--dataset', type=str, default="lung")
    parser.add_argument('--datapath', type=str, default='./datasets/LungSegmentation')
    parser.add_argument('--n_shot', type=int, default=5)
    parser.add_argument('--num_episodes', type=int, default=600)

    return parser.parse_args()


def extract_last_layer_features(model: AutoModel, image):
    with torch.inference_mode():
        outputs = model(pixel_values = image)

    features = outputs.last_hidden_state

    patch_features = features[:, 5:, :] # removing CLS and 4 register tokens

    B, N, D = patch_features.shape

    H_p = image.shape[-2] // 16
    W_p = image.shape[-1] // 16

    # [B, N, D] -> [B, H_p, W_p, D]
    patch_features = patch_features.reshape(B, H_p, W_p, D)

    # -> [B, D, H_p, W_p]
    patch_features = patch_features.permute(0, 3, 1, 2)

    return patch_features


def make_prototypes(model: AutoModel, *support_pairs):
    """Returns background and foreground prototypes as mean of fg/bg features"""
    fg_sum = None
    bg_sum = None
    fg_count = 0
    bg_count = 0

    for image, mask in support_pairs:
        features = extract_last_layer_features(model, image)
        features = features.permute(0, 2, 3, 1)
        _, H_p, W_p, _ = features.shape

        mask_preprocess = F.interpolate(mask.unsqueeze(0) if mask.ndim == 3 else mask, size = (H_p, W_p), mode = "nearest").squeeze(1)

        fg_mask = mask_preprocess.bool()
        bg_mask = ~fg_mask

        current_fg = features[fg_mask]
        current_bg = features[bg_mask]

        if fg_sum is None:
            fg_sum = current_fg.sum(dim=0)
            bg_sum = current_bg.sum(dim=0)
        else:
            fg_sum += current_fg.sum(dim=0)
            bg_sum += current_bg.sum(dim=0)

        fg_count += current_fg.shape[0]
        bg_count += current_bg.shape[0]

    fg_prototype = fg_sum / fg_count
    bg_prototype = bg_sum / bg_count

    return (bg_prototype, fg_prototype)


def make_prediction(query_features, bg_prototype, fg_prototype):
    # [1, 768, 30, 30] -> [1, 30, 30, 768]
    query_features = query_features.permute(0, 2, 3, 1)

    fg_similarity = F.cosine_similarity(query_features, fg_prototype, dim = -1)
    bg_similarity = F.cosine_similarity(query_features, bg_prototype, dim = -1)

    fg_similarity = fg_similarity.unsqueeze(1)
    bg_similarity = bg_similarity.unsqueeze(1)

    # [1, 1, 30, 30] -> [1, 1, 480, 480]
    fg_similarity = F.interpolate(fg_similarity, size = (480, 480), mode = "bilinear", align_corners = False)
    bg_similarity = F.interpolate(bg_similarity, size = (480, 480), mode = "bilinear", align_corners = False)

    prediction = (fg_similarity > bg_similarity).long()

    # [1, 1, 480, 480] -> [1, 480, 480]
    prediction = prediction.squeeze(1)

    return prediction


def IoU(mask, prediction):
    """Counts IoU for one pair (Mask, Prediction)"""

    intersection = ((mask * prediction) == 1).sum()
    union = ((mask + prediction) > 0).sum()

    return intersection / union


def ASSD(mask, prediction):
    mask = mask.squeeze().detach().cpu().numpy().astype(bool)
    prediction = prediction.squeeze().detach().cpu().numpy().astype(bool)

    if not mask.any() and not prediction.any(): 
        return 0.0 # mask and predicition match perfectly

    if not mask.any() or not prediction.any():
        return np.nan # no boundaries in one of the maps, can't calculate ASSD

    mask_boundary = mask ^ binary_erosion(mask)
    prediction_boundary = prediction ^ binary_erosion(prediction)

    distance_to_mask = distance_transform_edt(~mask_boundary)
    distance_to_prediction = distance_transform_edt(~prediction_boundary)

    prediction_to_mask = distance_to_mask[prediction_boundary]
    mask_to_prediction = distance_to_prediction[mask_boundary]

    return (prediction_to_mask.sum() + mask_to_prediction.sum()) / (prediction_to_mask.size + mask_to_prediction.size)


class Evaluator:
    def __init__(self, class_number):
        self.class_IoU = {
            cls: {
                "total_intersection": torch.tensor(0, device=device),
                "total_union": torch.tensor(0, device=device),
            }
            for cls in range(0, class_number)
        }

        self.class_ASSD = {
            cls: {
                "total_assd": 0.0,
                "count": 0,
            }
            for cls in range(class_number)
        }

        self.invalid_assd = 0

    def accumulate(self, intersection, union, assd, episode_idx):
        class_idx = episode_idx % len(self.class_IoU)

        self.class_IoU[class_idx]["total_intersection"] += intersection
        self.class_IoU[class_idx]["total_union"] += union

        if np.isnan(assd):
            self.invalid_assd += 1
        else:
            self.class_ASSD[class_idx]["total_assd"] += assd
            self.class_ASSD[class_idx]["count"] += 1

    def evaluate_mIoU(self):
        """Evaluate mIoU between clases"""

        ious = []
        
        for cls, stats in self.class_IoU.items():
            iou = stats["total_intersection"] / stats["total_union"]
        
            print(f"Class {cls} IoU: {iou.item() * 100:.2f}")
            ious.append(iou)
        
        mIoU = torch.stack(ious).mean()

        return mIoU.item() * 100

    def evaluate_ASSD(self):
        """Evaluate mASSD between clases"""

        assds = []

        for cls, stats in self.class_ASSD.items():
            if stats["count"] == 0:
                continue

            class_assd = stats["total_assd"] / stats["count"]

            print(f"Class {cls} ASSD: {class_assd:.4f}")
            assds.append(class_assd)

        return np.mean(assds)


def extract_files(selected):
    pairs = []
    
    for image_path, mask_path in selected:
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
    
        image = preprocess_image(image)
        mask = preprocess_mask(mask)
    
        pairs.append((image, mask))
    
    return pairs


def run_test(dataset, num_episodes, n_shot):
    evaluator = Evaluator(dataset.class_number)

    for i in range(num_episodes):
        print(f"---------EPISODE {i}---------")

        selected = dataset.sample_episode(i, n_shot)

        pairs = extract_files(selected)

        support = pairs[:n_shot]
        query = pairs[-1]
                
        bg_prototype, fg_prototype = make_prototypes(model, *support)

        query_features = extract_last_layer_features(model, query[0])

        prediction = make_prediction(query_features, bg_prototype, fg_prototype)

        intersection = ((query[-1] * prediction) == 1).sum()
        union = ((query[-1] + prediction) > 0).sum()

        assd = ASSD(query[-1], prediction)

        evaluator.accumulate(intersection, union, assd, i)

        print(f"IoU: {IoU(query[-1], prediction) * 100}")
        print(f"ASSD: {assd}")

    print("--------------------------------------------")
    print("Testing finished!")

    print(f"mIoU: {evaluator.evaluate_mIoU()}")
    print(f"ASSD: {evaluator.evaluate_ASSD()}")
    print(f"Undefined ASSD episodes: {evaluator.invalid_assd}")


DATASETS = {
    "lung": LungDataset,
    "deepglobe": DeepglobeDataset,
    "fss": FSSDataset
}


def getDataset(dataset, datapath):
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available: {list(DATASETS.keys())}"
        )

    dataset_class = DATASETS[dataset]

    return dataset_class(datapath)


def main(args):
    print("----------------------------------")
    print("Starting test with DINOv3")
    print(f"Dataset: {args.dataset}")
    print(f"n_shot: {args.n_shot}")
    print(f"num_episodes: {args.num_episodes}")
    print(f"device: {device}")
    print("----------------------------------")

    dataset = getDataset(args.dataset, args.datapath)
    run_test(dataset, args.num_episodes, args.n_shot)


if __name__ == '__main__':
    args = parse_args()
    main(args)


# ---------------------------------------TEST SECTION-------------------------------------------

# print("-------------------------Getting Lung Dataset dataset-------------------------")

# dataset = LungDataset("datasets/LungSegmentation")

# print(len(dataset.samples))

# print(dataset.samples[0])

# print("-------------------------Getting DeepGlobe dataset-------------------------")

# datasetDG = DeepglobeDataset("datasets/Deepglobe")

# print(len(datasetDG.first_class_samples))

# print(datasetDG.first_class_samples[0])

# print("-------------------------Running test-------------------------")

# run_test(dataset, 1, 5)

# print("-------------------------Importing Support Image-------------------------")

# img = Image.open("datasets/LungSegmentation/CXR_png/CHNCXR_0001_0.png").convert("RGB")

# print(img)
# print(img.mode)

# img = preprocess_image(img)

# print(img.shape)

# features = extract_last_layer_features(model, img)

# print(features.shape)

# print("-------------------------Importing Support Mask-------------------------")

# mask = Image.open("datasets/LungSegmentation/masks/CHNCXR_0001_0_mask.png")

# print(mask)
# print(mask.mode)

# mask = preprocess_mask(mask)

# print(mask.shape)

# print("-------------------------Making prototypes-------------------------")

# bg_prot, fg_prot = make_prototypes(model, (img, mask))

# print(bg_prot.shape)

# print("----------------------------")

# print(fg_prot.shape)

# print("-------------------------Importing Query Image and Mask-------------------------")

# query = Image.open("datasets/LungSegmentation/CXR_png/CHNCXR_0002_0.png").convert("RGB")

# query_mask = Image.open("datasets/LungSegmentation/masks/CHNCXR_0002_0_mask.png")

# query = preprocess_image(query)

# query_mask = preprocess_mask(query_mask)

# print("Preprocessed query image and mask!")

# print("-------------------------Making prediction-------------------------")

# query_features = extract_last_layer_features(model, query)

# pred = make_prediction(query_features, bg_prot, fg_prot)

# print(pred.shape)

# print("-------------------------Testing-------------------------")

# run_test(dataset, 600, 5)
