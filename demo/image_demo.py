from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch
from PIL import Image

def load_stable_diffusion(model_name="stabilityai/stable-diffusion-2-1", device="cuda"):
    """Load a Stable Diffusion model."""
    print(f"Loading {model_name}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device)
    return pipe

def generate_image(pipe, prompt, output_path, num_inference_steps=50):
    """Generate an image using the model."""
    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
    ).images[0]
    
    # Save the image
    image.save(output_path)
    print(f"Image saved to {output_path}")
    return image

def main():
    # Example models and prompts
    model_configs = [
        {
            "name": "stabilityai/stable-diffusion-2-1",
            "prompts": [
                "A serene landscape with mountains and a lake at sunset",
                "A futuristic city with flying cars and neon lights"
            ]
        }
    ]
    
    for config in model_configs:
        try:
            pipe = load_stable_diffusion(config["name"])
            
            print(f"\nTesting {config['name']}:")
            for i, prompt in enumerate(config["prompts"]):
                output_path = f"output_{i}.png"
                print(f"\nPrompt: {prompt}")
                generate_image(pipe, prompt, output_path)
                print("-" * 80)
            
            # Clear GPU memory
            del pipe
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error with {config['name']}: {str(e)}")

if __name__ == "__main__":
    main()
