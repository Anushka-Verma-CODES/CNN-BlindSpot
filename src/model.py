import torch
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image

def load_model():
    weights = MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    return model, preprocess, categories

def get_probabilities(image: Image.Image, model, preprocess):
    """Runs inference on a PIL image in memory and returns the probability tensor."""
    batch = preprocess(image).unsqueeze(0)
    with torch.inference_mode():
        prediction = model(batch).squeeze(0)
    probabilities = torch.nn.functional.softmax(prediction, dim=0)
    return probabilities
