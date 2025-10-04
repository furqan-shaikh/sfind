import asyncio

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel



async def main():
    file_path = "/Users/furqanshaikh/Documents/dev/sfind/images/tennis.jpg"
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32", dtype=torch.bfloat16, attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    #
    image = Image.open(file_path)
    # labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
    labels = ["a photo of a dog", "man playing tennis", ""]
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    #
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    resolved_probs =  [[round(v, 10) for v in sublist] for sublist in probs.tolist()]
    for sublist in probs.tolist():
        for value in sublist:
            print(f"{value:.10f}")
    print(resolved_probs)
    most_likely_idx = probs.argmax(dim=1).item()
    most_likely_label = labels[most_likely_idx]
    print(f"Most likely label: {most_likely_label} with probability: {probs[0][most_likely_idx].item():.3f}")


    def read_image_bytes(file_path: str):
        try:
            with open(file_path, "rb") as file:
                image_bytes = file.read()
            return image_bytes
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None



if __name__ == "__main__":
    asyncio.run(main())
