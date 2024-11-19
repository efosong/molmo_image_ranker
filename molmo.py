from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import pandas as pd
import numpy as np
import torch
import einops
import itertools
from tqdm import tqdm
from typing import List, Dict


def concatenate_images_horizontally(path_a: str, path_b: str) -> Image.Image:
    """
    Load two images and concatenate them horizontally with image A on the left and image B on the right.
    
    Args:
        path_a (str): Path to the first image to be placed on the left
        path_b (str): Path to the second image to be placed on the right
    
    Returns:
        PIL.Image.Image: A new image containing both input images side by side
        
    Raises:
        FileNotFoundError: If either image path is invalid
        IOError: If there are issues loading the images
    """
    try:
        # Load both images
        img_a = Image.open(path_a)
        img_b = Image.open(path_b)
        
        # Convert images to RGB if they're not already
        if img_a.mode != 'RGB':
            img_a = img_a.convert('RGB')
        if img_b.mode != 'RGB':
            img_b = img_b.convert('RGB')
        
        # Get dimensions
        width_a, height_a = img_a.size
        width_b, height_b = img_b.size
        
        # Calculate dimensions for the combined image
        total_width = width_a + width_b
        max_height = max(height_a, height_b)
        
        # Create a new blank image with the combined dimensions
        combined_img = Image.new('RGB', (total_width, max_height))
        
        # Paste the images side by side
        combined_img.paste(img_a, (0, 0))
        combined_img.paste(img_b, (width_a, 0))
        
        return combined_img
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find image file: {str(e)}")
    except Exception as e:
        raise IOError(f"Error processing images: {str(e)}")

def process_image_batch(
    processor: AutoProcessor,
    images: List[Image.Image],
    text: str,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device(0),
    ) -> Dict[str, torch.Tensor]:
    """
    Load two images and concatenate them horizontally with image A on the left and image B on the right.
    
    Args:
        images: List of images to process
        text: Text input for all prompts
    
    Returns:
        Dict: a processed input dict
    """
    inputs = torch.utils._pytree.tree_map(
        lambda *leaf: torch.stack(leaf).to(device),
        *(processor.process(images=[img], text=text) for img in images)
    )
    inputs["images"] = inputs["images"].to(dtype)
    return inputs

@torch.no_grad()
def compare_images(
    path_a: str, path_b: str,
    prompt: str,
    processor: AutoProcessor, model: AutoModelForCausalLM, r_l_tokens: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    img_lr = concatenate_images_horizontally(path_a, path_b)
    img_rl = concatenate_images_horizontally(path_b, path_a)
    batch_inputs = process_image_batch(
        processor=processor,
        images=[img_lr, img_rl],
        text=prompt,
        dtype=dtype,
        device=model.device,
    )
    return model(**batch_inputs).logits[:,-1,r_l_tokens].flatten()


DTYPE = torch.bfloat16
PROMPT = """This image contains two pictures: one in the left half (L), and one on the right half (R). You must indicate which picture you prefer by simply stating "L" or "R" followed by a newline.

The image I prefer is:"""
R_L_TOKEN_STRINGS = [" L", " R"]
N_IMAGES = 64
DATA_SAVE_NAME = "data.pkl"


# load the processor
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype=DTYPE,
    device_map='auto'
)
r_l_tokens = torch.tensor(processor.tokenizer.encode(R_L_TOKEN_STRINGS))

# load the model
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype=DTYPE,
    device_map='auto'
)

# compute the pairwise comparisons
pairwise_comparisons = {
    (a,b): compare_images(
        f"img/{a:02}.png", f"img/{b:02}.png",
        prompt=PROMPT, r_l_tokens=r_l_tokens,
        processor=processor, model=model, dtype=DTYPE
    ).to(torch.float32).cpu().numpy()
    for a,b in tqdm(list(itertools.combinations(range(1,N_IMAGES+1), 2)))
}

# save comparison data
df = pd.DataFrame.from_dict(
    pairwise_comparisons,
    orient="index",
    columns=["L", "R", "iL", "iR"]
)
df.index = pd.MultiIndex.from_tuples(df.index, names=["img_a", "img_b"])
df.to_pickle(DATA_SAVE_NAME)
