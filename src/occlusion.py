import torch
import numpy as np
from PIL import Image
from model import get_probabilities
from image_processing import apply_grid_mask

def calculate_sensitivity_matrix(image: Image.Image, model, preprocess, target_class_id: int, grid_size: int = 4) -> np.ndarray:
    """
    Calculates the occlusion sensitivity matrix for a given image and target class.
    Returns a 2D numpy array of score differences D(i) = S_original - S_masked.
    Positive values mean the region was important (masking it hurt the score).
    """
    # 1. Get the original score
    orig_probs = get_probabilities(image, model, preprocess)
    orig_score = orig_probs[target_class_id].item()
    
    # 2. Initialize an empty matrix to hold our differences
    sensitivity_matrix = np.zeros((grid_size, grid_size))
    
    # 3. Loop through every row and column
    for row in range(grid_size):
        for col in range(grid_size):
            # Mask just this cell
            masked_img = apply_grid_mask(image, grid_size, row, col)
            
            # Run inference on the masked image
            masked_probs = get_probabilities(masked_img, model, preprocess)
            masked_score = masked_probs[target_class_id].item()
            
            # Calculate D(i)
            sensitivity_matrix[row, col] = orig_score - masked_score
            
    return sensitivity_matrix
