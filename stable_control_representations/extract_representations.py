import torch
from PIL import Image
import torchvision.transforms as T
from omegaconf import OmegaConf
from vc_models.models.diffusion_model import DiffusionRepresentation

def extract_representations(image_path, prompt=""):
    # Load configuration
    config = {
        "model_name": "runwayml/stable-diffusion-v1-5",
        "unet_path": None,  # Change this if you have a local path to unet weights
        "noise_sampling": "per_image",
        "representation_layer_name": ["down_1", "down_2", "down_3", "mid"],
        "timestep": [0],  # Timestep 0 for clean image representations
        "tokenize_captions": True,
        "get_attention_maps": False,
        "input_image_size": 256,
        "flatten": True,
        "dtype": "float16",
    }
    
    # Create the diffusion model
    model = DiffusionRepresentation(**config)
    
    # Prepare the image
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Extract representations
    with torch.no_grad():
        representations = model(img_tensor, prompt=[prompt])
    
    return representations

# Example usage
if __name__ == "__main__":
    image_path = "/opt/data/borghei/lavis_dataset/coco/images/train2014/COCO_train2014_000000000009.jpg"
    prompt = ""  # Empty prompt or provide a text description
    
    representations = extract_representations(image_path, prompt)
    print(f"Extracted representations shape: {representations.shape}")
    
    # You can access the features for any downstream task
    # For example: predictions = your_classifier(representations)
