import base64
from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse, JSONResponse
import uvicorn
import torch
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from tqdm import tqdm
import io
import numpy as np
import os
import sys
import requests
from diffusers import StableDiffusionPipeline

# Add 'experiment/repos/latent-diffusion' to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # NOQA
latent_diffusion_path = os.path.join(current_dir, '../repos/latent-diffusion')  # NOQA
sys.path.append(latent_diffusion_path)  # NOQA

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

app = FastAPI()

# Initialize model and sampler at startup
model = None
sampler = None
diffusion_model = None
model_choice = None

@app.get("/")
async def index():
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Text to Image Generation</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .container {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }
                .form-group {
                    margin-bottom: 15px;
                }
                label {
                    display: block;
                    margin-bottom: 5px;
                }
                select, input[type="text"], input[type="number"] {
                    width: 100%;
                    padding: 8px;
                    margin-bottom: 10px;
                }
                button {
                    padding: 10px 20px;
                    background-color: #007bff;
                    color: white;
                    border: none;
                    cursor: pointer;
                }
                button:hover {
                    background-color: #0056b3;
                }
                #imageContainer {
                    margin-top: 20px;
                }
                #imageContainer img {
                    max-width: 100%;
                    height: auto;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Text to Image Generation</h1>
                <div class="form-group">
                    <label for="model">Choose a model:</label>
                    <select name="model" id="model">
                        <option value="stable-diffusion-v1-5">Stable Diffusion v1.5</option>
                        <option value="stable-diffusion-v1-0">Stable Diffusion v1.0</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="prompt">Enter your prompt:</label>
                    <input type="text" id="prompt" name="prompt" required>
                </div>
                
                <div class="form-group">
                    <label for="height">Image Height:</label>
                    <input type="number" id="height" name="height" value="512" min="256" max="1024" step="64">
                </div>
                
                <div class="form-group">
                    <label for="width">Image Width:</label>
                    <input type="number" id="width" name="width" value="512" min="256" max="1024" step="64">
                </div>
                
                <div class="form-group">
                    <label for="scale">Guidance Scale:</label>
                    <input type="number" id="scale" name="scale" value="7.5" min="1" max="20" step="0.5">
                </div>
                
                <button onclick="generateImage()">Generate Image</button>
                <button onclick="regenerateImage()" id="regenerateBtn" style="display: none;">Regenerate</button>
                
                <div id="imageContainer"></div>
            </div>
            
            <script>
            async function generateImage() {
                const prompt = document.getElementById('prompt').value;
                const height = document.getElementById('height').value;
                const width = document.getElementById('width').value;
                const scale = document.getElementById('scale').value;
                const model = document.getElementById('model').value;
                
                const formData = new FormData();
                formData.append('prompt', prompt);
                formData.append('height', height);
                formData.append('width', width);
                formData.append('scale', scale);
                formData.append('model_name', model);
                
                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    const imageContainer = document.getElementById('imageContainer');
                    imageContainer.innerHTML = '';
                    
                    // Display the final image
                    const finalImage = data.images.find(img => img.type === 'final');
                    if (finalImage) {
                        const img = document.createElement('img');
                        img.src = `data:image/png;base64,${finalImage.data}`;
                        imageContainer.appendChild(img);
                    }
                    
                    document.getElementById('regenerateBtn').style.display = 'block';
                } catch (error) {
                    console.error('Error:', error);
                }
            }
            
            function regenerateImage() {
                generateImage();
            }
            </script>
        </body>
    </html>
    """)

@app.post("/set_model")
async def set_model(model: str = Form(...)):
    global model_choice
    model_choice = model
    return RedirectResponse(url="/", status_code=303)

@app.on_event("startup")
async def startup_event():
    global model, sampler, diffusion_model
    
    # Initialize both models
    # 1. Stable Diffusion v1.5
    model_id = "sd-legacy/stable-diffusion-v1-5"
    diffusion_model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    diffusion_model = diffusion_model.to("cuda")
    
    # 2. Original model
    model_url = "https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt"
    cache_dir = os.path.expanduser("~/.cache/latent-diffusion")
    os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = os.path.join(cache_dir, "txt2img-f8-large_model.ckpt")
    config_path = os.path.join(latent_diffusion_path, "configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    download_model_checkpoint(model_url, checkpoint_path)
    model = load_model(config_path, checkpoint_path)
    sampler = DDIMSampler(model)

# Function to download model checkpoint
def download_model_checkpoint(url, path):
    if not os.path.exists(path):
        print("Downloading model checkpoint...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(path, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        if total_size != 0 and t.n != total_size:
            print("ERROR, something went wrong")
        else:
            print("Model checkpoint downloaded.")

# Load model configuration and instantiate model
def load_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.cuda().eval()
    return model

# Main route for generating images
@app.post("/generate")
async def generate_image(
    prompt: str = Form(...),
    height: int = Form(512),
    width: int = Form(512),
    scale: float = Form(7.5),
    model_name: str = Form("stable-diffusion-v1-5"),
    ddim_eta: float = Form(1.0),
    n_iter: int = Form(1)
):
    global diffusion_model, model, sampler
    n_samples = 1  # Ensure only one sample is generated
    images = []

    if model_name == "stable-diffusion-v1-5":
        # Use Stable Diffusion v1.5
        image = diffusion_model(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=scale,
            num_inference_steps=50
        ).images[0]
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
        images.append({"type": "final", "data": img_base64})
    else:
        with torch.no_grad():
            with model.ema_scope():
                # Set up unconditional conditioning if using guidance
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning(n_samples * [""])

                for _ in range(n_iter):
                    # Get conditioning
                    c = model.get_learned_conditioning(n_samples * [prompt])
                    shape = [4, height // 8, width // 8]

                    # Sample
                    samples, intermediates = sampler.sample(
                        S=200,
                        conditioning=c,
                        batch_size=n_samples,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        log_every_t = 20,
                    )

                    # Decode
                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    # Create output directory if it doesn't exist
                    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
                    os.makedirs(output_dir, exist_ok=True)

                    # Process final samples (full size)
                    for idx, x_sample in enumerate(x_samples):
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        pil_image = Image.fromarray(x_sample.astype(np.uint8))

                        # Save final image
                        filename = f"final_{idx}.png"
                        pil_image.save(os.path.join(output_dir, filename))

                        # Convert to base64 for JSON response
                        img_byte_arr = io.BytesIO()
                        pil_image.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
                        images.append({"type": "final", "data": img_base64})

                    # Process intermediate samples (smaller size)
                    total_steps = len(intermediates['x_inter'])

                    for step_idx, x_inter  in enumerate(intermediates['x_inter']):
                        decoded = model.decode_first_stage(x_inter)
                        decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)

                        for idx, sample in enumerate(decoded):
                            sample = 255. * rearrange(sample.cpu().numpy(), 'c h w -> h w c')
                            pil_image = Image.fromarray(sample.astype(np.uint8))

                            # Save intermediate image
                            filename = f"step_{step_idx}_{idx}.png"
                            pil_image.save(os.path.join(output_dir, filename))

                            # Create smaller version for web display
                            small_size = (height//4, width//4)  # 1/4 of original size
                            pil_image_small = pil_image.resize(small_size, Image.Resampling.LANCZOS)

                            # Convert to base64 for JSON response
                            img_byte_arr = io.BytesIO()
                            pil_image_small.save(img_byte_arr, format='PNG')
                            img_byte_arr.seek(0)
                            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
                            images.append({"type": "intermediate", "step": step_idx, "data": img_base64})

    return JSONResponse(content={"images": images, "prompt": prompt})

@app.get("/main")
async def main():
    content = """
    <html>
        <head>
            <title>Latent Diffusion Image Generator</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        </head>
        <body class="container">
            <h1 class="mt-5">Generate Image from Text</h1>
            <form id="generateForm" class="mb-4">
                <div class="form-group">
                    <label for="prompt">Enter Text Prompt:</label>
                    <input type="text" class="form-control" id="prompt" name="prompt" required>
                </div>
                <div class="form-group">
                    <label for="ddim_eta">DDIM Eta:</label>
                    <input type="number" step="0.1" class="form-control" id="ddim_eta" name="ddim_eta" value="1.0">
                </div>
                <div class="form-group">
                    <label for="n_iter">Number of Iterations:</label>
                    <input type="number" class="form-control" id="n_iter" name="n_iter" value="1">
                </div>
                <div class="form-group">
                    <label for="height">Height:</label>
                    <input type="number" class="form-control" id="height" name="height" value="512">
                </div>
                <div class="form-group">
                    <label for="width">Width:</label>
                    <input type="number" class="form-control" id="width" name="width" value="512">
                </div>
                <div class="form-group">
                    <label for="scale">Scale:</label>
                    <input type="number" step="0.1" class="form-control" id="scale" name="scale" value="5.0">
                </div>
                <button type="submit" class="btn btn-primary">Generate</button>
            </form>

            <div id="results" class="mt-4">
                <div id="promptDisplay" class="mb-3"></div>
                <div id="imageContainer" class="text-center">
                    <div class="loading d-none">Generating images...</div>
                    <div id="imagesGrid" class="d-flex flex-wrap justify-content-center"></div>
                </div>
            </div>

            <script>
            $(document).ready(function() {
                $('#generateForm').on('submit', function(e) {
                    e.preventDefault();

                    // Show loading state
                    $('.loading').removeClass('d-none');
                    $('#generatedImage').addClass('d-none');

                    const formData = new FormData(this);
                    const prompt = formData.get('prompt');

                    // Display prompt
                    $('#promptDisplay').html('<h4>Prompt: ' + prompt + '</h4>');

                    $.ajax({
                        url: '/generate',
                        type: 'POST',
                        data: formData,
                        processData: false,
                        contentType: false,
                        success: function(response) {
                            // Hide loading state
                            $('.loading').addClass('d-none');

                            // Clear previous images
                            $('#imagesGrid').empty();

                            // Display final image first
                            const finalImages = response.images.filter(img => img.type === 'final');
                            const intermediateImages = response.images.filter(img => img.type === 'intermediate');

                            // Create container for final images
                            const finalContainer = $('<div class="mb-4">');
                            finalContainer.append('<h3>Final Result</h3>');
                            finalImages.forEach(img => {
                                const imgContainer = $('<div class="text-center">');
                                const imgElement = $('<img>')
                                    .attr('src', 'data:image/png;base64,' + img.data)
                                    .addClass('img-fluid mb-2');
                                imgContainer.append(imgElement);
                                finalContainer.append(imgContainer);
                            });
                            $('#imagesGrid').append(finalContainer);

                            // Create container for intermediate steps
                            const intermediateContainer = $('<div>');
                            intermediateContainer.append('<h3>Generation Process</h3>');
                            const stepsGrid = $('<div class="d-flex flex-wrap justify-content-center">');

                            intermediateImages.forEach(img => {
                                const imgContainer = $('<div class="m-1 text-center">');
                                const imgElement = $('<img>')
                                    .attr('src', 'data:image/png;base64,' + img.data)
                                    .addClass('img-fluid mb-1')
                                    .css('max-width', '150px');
                                const caption = $('<p class="mb-0 small">').text(`Step ${img.step}`);

                                imgContainer.append(imgElement, caption);
                                stepsGrid.append(imgContainer);
                            });

                            intermediateContainer.append(stepsGrid);
                            $('#imagesGrid').append(intermediateContainer);
                        },
                        error: function() {
                            $('.loading').addClass('d-none');
                            $('#imageContainer').append(
                                '<div class="alert alert-danger">Error generating image</div>'
                            );
                        }
                    });
                });
            });
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30982)

    # Define cache directory in ~/.cache
    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'latent-diffusion')
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, 'txt2img-f8-large_model.ckpt')

    # Download the model checkpoint if it doesn't exist
    if not os.path.exists(model_path):
        download_model_checkpoint(model_url, model_path)

    # Load model
    config_path = os.path.join(current_dir, '../repos/latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml')
    model = load_model(config_path, model_path)
    sampler = DDIMSampler(model)

    uvicorn.run(app, host="0.0.0.0", port=30982)
