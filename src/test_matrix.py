from PIL import Image
from model import load_model, get_probabilities
from occlusion import calculate_sensitivity_matrix
import torch

def run_test():
    model, preprocess, categories = load_model()
    img = Image.open("../assets/images/dog.jpg").convert("RGB")
    
    # Determine the original top class to use as our target
    orig_probs = get_probabilities(img, model, preprocess)
    target_class_id = torch.argmax(orig_probs).item()
    
    print(f"Target Class: {categories[target_class_id]}")
    print("Calculating 4x4 sensitivity matrix... (this runs inference 16 times)")
    
    matrix = calculate_sensitivity_matrix(img, model, preprocess, target_class_id, grid_size=4)
    
    print("\nSensitivity Matrix D(i):")
    # Multiply by 100 for easier readability (percentages)
    print(matrix * 100)

if __name__ == "__main__":
    run_test()
