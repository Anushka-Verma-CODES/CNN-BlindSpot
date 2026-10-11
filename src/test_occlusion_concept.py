from PIL import Image
import torch
from model import load_model, get_probabilities
from image_processing import apply_grid_mask

def run_test():
    print("Loading model and original image...")
    model, preprocess, categories = load_model()
    original_img = Image.open("../assets/images/dog.jpg").convert("RGB")

    # Get original score
    orig_probs = get_probabilities(original_img, model, preprocess)
    top_prob, top_catid = torch.max(orig_probs, 0)
    target_class_id = top_catid.item()
    target_class_name = categories[target_class_id]
    orig_score = top_prob.item()

    print(f"\nOriginal Prediction: {target_class_name} ({orig_score * 100:.2f}%)")

    # Mask the face (row 1, col 1 in 4x4 grid) and get new score
    print("Masking row 1, col 1...")
    masked_img = apply_grid_mask(original_img, grid_size=4, row=1, col=1)
    masked_probs = get_probabilities(masked_img, model, preprocess)
    masked_score = masked_probs[target_class_id].item()

    print(f"Masked Prediction (same class): {target_class_name} ({masked_score * 100:.2f}%)")
    print(f"Confidence Drop: {(orig_score - masked_score) * 100:.2f}%")

if __name__ == "__main__":
    run_test()
