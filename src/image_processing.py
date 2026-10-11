import os
from PIL import Image, ImageDraw

def apply_grid_mask(image: Image.Image, grid_size: int, row: int, col: int, mask_color=(128, 128, 128)) -> Image.Image:
    """
    Divides the image into a `grid_size x grid_size` grid and masks the specified (row, col).
    """
    # Create a copy so we don't permanently alter the original image in memory
    masked_img = image.copy()
    draw = ImageDraw.Draw(masked_img)
    
    width, height = masked_img.size
    cell_width = width / grid_size
    cell_height = height / grid_size
    
    # Calculate pixel coordinates for the bounding box
    x0 = col * cell_width
    y0 = row * cell_height
    x1 = (col + 1) * cell_width
    y1 = (row + 1) * cell_height
    
    # Draw a solid gray rectangle over the calculated area
    draw.rectangle([x0, y0, x1, y1], fill=mask_color)
    
    return masked_img

if __name__ == "__main__":
    img_path = "assets/images/dog.jpg"
    if os.path.exists(img_path):
        original_img = Image.open(img_path).convert("RGB")
        print(f"Original image size: {original_img.size}")
        
        # Test: Mask row 1, column 1 (0-indexed) of a 4x4 grid. 
        # This is the second row, second column.
        masked = apply_grid_mask(original_img, grid_size=4, row=1, col=1)
        
        output_path = "assets/images/dog_masked_test.jpg"
        masked.save(output_path)
        print(f"Success! Masked image saved to {output_path}")
    else:
        print("Error: Could not find dog image.")
