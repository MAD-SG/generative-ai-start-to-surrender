import base64
from fastapi import FastAPI, Form, BackgroundTasks, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import torch
from PIL import Image
import io
import os
import time
from pathlib import Path
from typing import Dict, Optional
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, AutoPipelineForText2Image
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
import torch
from threading import Lock
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class TaskStatus(Enum):
    QUEUED = "queued"
    LOADING_MODEL = "loading_model"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TaskProgress:
    status: TaskStatus
    progress: float = 0.0
    total_steps: int = 0
    current_step: int = 0
    start_time: float = None
    end_time: float = None
    error: str = None
    image: io.BytesIO = None  # Store the generated image

class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.lock = Lock()

    def create_task(self, task_id: str) -> TaskProgress:
        with self.lock:
            self.tasks[task_id] = TaskProgress(status=TaskStatus.QUEUED)
            return self.tasks[task_id]

    def get_task(self, task_id: str) -> TaskProgress:
        with self.lock:
            return self.tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs):
        with self.lock:
            if task_id in self.tasks:
                for key, value in kwargs.items():
                    setattr(self.tasks[task_id], key, value)

app = FastAPI()
task_manager = TaskManager()

# Get the directory containing app.py
BASE_DIR = Path(__file__).resolve().parent

# Mount static files directory
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

import yaml

# Add new import at the top
from collections import defaultdict

# Update config loading function
def load_model_config():
    config_path = BASE_DIR / 'config' / 'model_parameters.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

            # Keep configuration as-is, conversion will happen in ModelAdapter

            return config['models']
    except Exception as e:
        print(f"Error loading model configurations: {e}")
        raise

# Update global variable
MODEL_DEFS = load_model_config()


# Global variables for model management
class ModelAdapter:
    def __init__(self, model_id: str, model_config: dict):
        self.model_id = model_id
        self.model_config = model_config
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _create_pipeline(self, use_quantized: bool = False):
        config = self.model_config.copy()

        # Convert torch_dtype from string to actual dtype
        if 'torch_dtype' in config:
            dtype_str = config['torch_dtype']
            if dtype_str == 'float16':
                config['torch_dtype'] = torch.float16
            elif dtype_str == 'bfloat16':
                config['torch_dtype'] = torch.bfloat16
            elif dtype_str == 'float32':
                config['torch_dtype'] = torch.float32
            else:
                raise ValueError(f"Unsupported torch_dtype: {dtype_str}")

        repo_id = config.get('repo_id', self.model_id)
        other_config = {k: v for k, v in config.items() if k != 'repo_id'}



        # Handle different model types
        if self.model_id=='sd-v3.5-quant':
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=other_config['torch_dtype'],
            )
            model_nf4 = SD3Transformer2DModel.from_pretrained(
                repo_id,
                subfolder="transformer",
                quantization_config=nf4_config,
                torch_dtype=other_config['torch_dtype'],
            )
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                repo_id,
                transformer=model_nf4,
                torch_dtype=other_config['torch_dtype'],
            )
            return pipeline
        elif self.model_id == 'sd-v3.5':

            # from transformers import T5EncoderModel, CLIPTextModel, CLIPTokenizer
            # from diffusers import StableDiffusion3Pipeline
            # # 预加载文本编码器
            # text_encoder_1 = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
            # text_encoder_2 = T5EncoderModel.from_pretrained("google/t5-v1_1-xl")

            # # 预加载分词器
            # tokenizer_1 = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            # tokenizer_2 = CLIPTokenizer.from_pretrained("google/t5-v1_1-xl")

            pipeline = StableDiffusion3Pipeline.from_pretrained(repo_id,
                    # text_encoder=text_encoder_1,
                    # text_encoder_2=text_encoder_2,
                    # tokenizer=tokenizer_1,
                    # tokenizer_2=tokenizer_2,
                    **other_config
                )
            pipeline.vae.enable_tiling()
            return pipeline

        elif self.model_id == 'animagine-xl-4':
            return StableDiffusionXLPipeline.from_pretrained(repo_id, **other_config)

        elif self.model_id == 'lumina-2':
            from diffusers import Lumina2Text2ImgPipeline
            return Lumina2Text2ImgPipeline.from_pretrained(repo_id, **other_config)

        elif self.model_id.startswith('flux'):
            # Add specialized flux model handling here
            return AutoPipelineForText2Image.from_pretrained(repo_id, **other_config)
        elif self.model_id == 'sd-v3.0':
            return StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", **other_config)

        else:
            return AutoPipelineForText2Image.from_pretrained(repo_id, **other_config)

    def load(self, use_quantized: bool = False):
        try:
            if self.pipeline is None:
                self.pipeline = self._create_pipeline(use_quantized)
                self.pipeline.to(self.device)

                # Apply optimizations
                if self.model_config.get('use_dpm_solver', False):
                    self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                        self.pipeline.scheduler.config
                    )
            return self.pipeline
        except Exception as e:
            error_msg = f"Error loading model {self.model_id}: {str(e)}"
            print(error_msg)  # Log the error
            self.unload()  # Clean up any partially loaded model
            raise HTTPException(status_code=500, detail=error_msg)

    def unload(self):
        try:
            if self.pipeline is not None:
                try:
                    self.pipeline = self.pipeline.to("cpu")
                except Exception as e:
                    print(f"Warning: Error moving pipeline to CPU: {e}")
                try:
                    del self.pipeline
                except Exception as e:
                    print(f"Warning: Error deleting pipeline: {e}")
                self.pipeline = None
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Warning: Error clearing CUDA cache: {e}")
        except Exception as e:
            print(f"Error during model unload: {e}")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs
    ):
        try:
            if self.pipeline is None:
                raise RuntimeError("Model not loaded. Call load() first.")

            # Validate parameters
            if not isinstance(height, int) or not isinstance(width, int):
                raise ValueError("Height and width must be integers")
            if height < 128 or width < 128:
                raise ValueError("Height and width must be at least 128 pixels")
            if not prompt or not isinstance(prompt, str):
                raise ValueError("Prompt must be a non-empty string")

            # Generate image based on model type
            if self.model_id in ['flux1-schnell']:
                result = self.pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    **kwargs
                )
            else:
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    **kwargs
                )

            if not result.images or len(result.images) == 0:
                raise RuntimeError("No image was generated")

            return result.images[0]

        except Exception as e:
            error_msg = f"Error generating image with {self.model_id}: {str(e)}"
            print(error_msg)  # Log the error
            raise HTTPException(status_code=500, detail=error_msg)

class ModelManager:
    def __init__(self):
        self.adapters = {k: ModelAdapter(k, v['config'])
                        for k, v in MODEL_DEFS.items() if 'config' in v}
        self.last_used: Dict[str, float] = {k: 0 for k in MODEL_DEFS.keys()}
        self.lock = Lock()
        self.cleanup_interval = 600  # 10 minutes

    def get_model(self, model_id: str, use_quantized: bool = False):
        """Get or load a model pipeline

        Args:
            model_id (str): ID of the model to load
            use_quantized (bool): Whether to use quantization for supported models

        Returns:
            ModelAdapter: The loaded model adapter instance
        """
        with self.lock:
            if model_id not in self.adapters:
                return None
            self.last_used[model_id] = time.time()
            adapter = self.adapters[model_id]
            adapter.load(use_quantized)
            return adapter
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(repo_id, **other_config)
        pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model
        pipe = pipe.to('cuda')
        return pipe

    def _load_default_model(self, config, repo_id, other_config, model_id):
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            repo_id,
            revision=config['revision'],
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None if not config.get('safety_checker') else None
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        pipeline.to("cuda")
        pipeline.enable_attention_slicing()
        if config.get('type') == 'flux':
            pipeline.enable_vae_slicing()
            pipeline.enable_model_cpu_offload()
        if model_id in ['sd-v1.1', 'sd-v1.2', 'sd-v1.3']:
            pipeline.enable_sequential_cpu_offload()
            pipeline.enable_vae_slicing()
        return pipeline

    def cleanup_models(self):
        current_time = time.time()
        with self.lock:
            for model_id, last_used in self.last_used.items():
                if (current_time - last_used > self.cleanup_interval and
                    self.models[model_id] is not None):
                    print(f"Unloading model {model_id} due to inactivity")
                    self.models[model_id] = None
                    torch.cuda.empty_cache()

model_manager = ModelManager()

# Background task for model cleanup
def cleanup_task():
    while True:
        time.sleep(60)  # Check every minute
        model_manager.cleanup_models()

@app.on_event("startup")
async def startup_event():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(cleanup_task)

# Update index route grouping logic
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Build model groups from definitions
    model_groups = defaultdict(list)
    for model_id, model_def in MODEL_DEFS.items():
        if 'groups' not in model_def:
            continue

        for group_name in model_def['groups']:
            model_groups[group_name].append({
                "id": model_id,
                "name": model_def.get('display_name', model_id),
                "description": model_def.get('description', '')
            })

    # Sort groups and their contents
    sorted_groups = {}
    for group_name in sorted(model_groups.keys()):
        sorted_models = sorted(model_groups[group_name],
                              key=lambda x: x['name'])
        sorted_groups[group_name] = sorted_models

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_groups": sorted_groups
        }
    )

@app.get("/model-config/{model_name}")
async def get_model_config(model_name: str):
    """Get model-specific configuration parameters"""
    if model_name not in MODEL_DEFS:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    # Get base parameters
    config = MODEL_DEFS[model_name].copy()

    # Extract UI controls if present
    ui_controls = config.pop('ui_controls', {
        'show_prompt': True,
        'show_negative_prompt': True,
        'show_dimensions': True,
        'show_guidance_scale': True,
        'show_steps': True,
        'show_quantization': False
    })

    # Add model-specific settings
    model_config = MODEL_DEFS.get(model_name, {})

    # Add UI controls to config
    config['ui_controls'] = ui_controls

    # Add quantization option if enabled in UI controls
    if ui_controls.get('show_quantization'):
        config['use_quantized'] = {
            'type': 'boolean',
            'default': model_config['parameters'].get('use_quantized', False),
            'description': 'Use 4-bit quantization for reduced memory usage'
        }

    return config

@app.post("/generate")
async def generate_image(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    height: int = Form(512),
    width: int = Form(512),
    guidance_scale: float = Form(7.5),
    num_inference_steps: int = Form(50),
    model_name: str = Form("sd-v1.5"),
    negative_prompt: str = Form("", description="Optional negative prompt for better control"),
    use_quantized: bool = Form(None, description="Use quantization for supported models")
):
    """Generate image with validation for model-specific parameters"""
    # Get model configuration
    model_config = MODEL_DEFS.get(model_name, {})

    # Use model-specific negative prompt if available and no user-provided negative prompt
    if not negative_prompt and model_config.get('negative_prompt'):
        negative_prompt = model_config['negative_prompt']

    # Add aesthetic score for models that support it
    extra_args = {}
    if model_config.get('requires_aesthetic_score'):
        extra_args['aesthetic_score'] = model_config.get('aesthetic_score', 9.0)

    # Validate model exists
    if model_name not in MODEL_DEFS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model_name}")

    # Get model parameters
    params = MODEL_DEFS[model_name].get('parameters', {})
    if not params:
        raise HTTPException(status_code=400, detail=f"No parameters defined for model: {model_name}")

    # Validate parameters
    if not (params['height']['min'] <= height <= params['height']['max']):
        raise HTTPException(
            status_code=400,
            detail=f"Height must be between {params['height']['min']} and {params['height']['max']} for {model_name}"
        )

    if not (params['width']['min'] <= width <= params['width']['max']):
        raise HTTPException(
            status_code=400,
            detail=f"Width must be between {params['width']['min']} and {params['width']['max']} for {model_name}"
        )

    if not (params['guidance_scale']['min'] <= guidance_scale <= params['guidance_scale']['max']):
        raise HTTPException(
            status_code=400,
            detail=f"Guidance scale must be between {params['guidance_scale']['min']} and {params['guidance_scale']['max']} for {model_name}"
        )

    if not (params['num_inference_steps']['min'] <= num_inference_steps <= params['num_inference_steps']['max']):
        raise HTTPException(
            status_code=400,
            detail=f"Number of inference steps must be between {params['num_inference_steps']['min']} and {params['num_inference_steps']['max']} for {model_name}"
        )

    # Create a unique task ID
    task_id = f"{int(datetime.now().timestamp())}"
    task = task_manager.create_task(task_id)
    task.total_steps = num_inference_steps

    # Add the generation task to background tasks
    background_tasks.add_task(
        generate_image_task,
        task_id=task_id,
        model_name=model_name,
        use_quantized=use_quantized,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        extra_args=extra_args
    )

    # Return the task ID immediately
    return JSONResponse(
        content={"task_id": task_id},
        status_code=202  # Accepted
    )

# Add a new endpoint to get the generated image
@app.get("/image/{task_id}")
async def get_generated_image(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status == TaskStatus.FAILED:
        raise HTTPException(status_code=500, detail=task.error or "Generation failed")

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=425, detail="Image not ready yet")

    # Get the image from the task
    img_byte_arr = task.image  # We'll add this to TaskProgress
    if not img_byte_arr:
        raise HTTPException(status_code=500, detail="Image data not found")

    generation_time = round((task.end_time - task.start_time), 2) if task.end_time and task.start_time else 0

    return StreamingResponse(
        img_byte_arr,
        media_type="image/png",
        headers={
            "X-Generation-Time": str(generation_time)
        }
    )

# Add the background task function
async def generate_image_task(
    task_id: str,
    model_name: str,
    use_quantized: bool,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    extra_args: dict
):
    task = task_manager.get_task(task_id)
    if not task:
        print(f"Error: Task {task_id} not found")
        return

    adapter = None
    try:
        # Input validation
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        if height < 128 or width < 128:
            raise ValueError("Height and width must be at least 128 pixels")
        if num_inference_steps < 1:
            raise ValueError("Number of inference steps must be positive")
        if guidance_scale < 0:
            raise ValueError("Guidance scale must be non-negative")

        # Record start time and update status
        task.start_time = time.time()
        task_manager.update_task(
            task_id,
            status=TaskStatus.LOADING_MODEL,
            start_time=task.start_time
        )

        # Load model
        try:
            adapter = model_manager.get_model(model_name, use_quantized)
            if adapter is None:
                raise RuntimeError(f"Failed to get model adapter for {model_name}")
            pipeline = adapter.load(use_quantized)
            if pipeline is None:
                raise RuntimeError(f"Failed to load pipeline for {model_name}")
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

        # Update generation status
        task_manager.update_task(
            task_id,
            status=TaskStatus.GENERATING,
            total_steps=num_inference_steps
        )

        # Log generation parameters
        print(f"Generating with {model_name}:\n"
              f"Prompt: {prompt}\n"
              f"Negative: {negative_prompt}\n"
              f"Dimensions: {width}x{height}\n"
              f"Steps: {num_inference_steps}\n"
              f"Guidance: {guidance_scale}")

        # Generate image
        try:
            image = adapter.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **extra_args
            )
            if image is None:
                raise RuntimeError("No image was generated")
        except Exception as e:
            raise RuntimeError(f"Image generation failed: {str(e)}")

        # Save image
        try:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
        except Exception as e:
            raise RuntimeError(f"Failed to save generated image: {str(e)}")

        # Update task completion
        end_time = time.time()
        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            end_time=end_time,
            image=img_byte_arr,
            progress=1.0
        )

        print(f"Task {task_id} completed in {round(end_time - task.start_time, 2)}s")

    except Exception as e:
        error_msg = str(e)
        print(f"Error in generate_image_task: {error_msg}")
        # Update task failure status
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            error=error_msg,
            end_time=time.time()
        )

        # Clean up model on error
        if adapter and adapter.pipeline:
            try:
                adapter.unload()
            except Exception as cleanup_error:
                print(f"Warning: Error during model cleanup: {cleanup_error}")

@app.get("/task/{task_id}")
async def get_task_progress(task_id: str):
    """Get the progress of a specific task"""
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # Calculate progress percentage
    progress = 0.0
    if task.status == TaskStatus.COMPLETED:
        progress = 1.0
    elif task.status == TaskStatus.GENERATING and task.total_steps > 0:
        progress = min(0.9, task.current_step / task.total_steps)  # Cap at 90% until complete
    elif task.status == TaskStatus.LOADING_MODEL:
        progress = 0.1  # Show some progress while loading

    # Calculate time information
    current_time = time.time()
    time_elapsed = None
    time_remaining = None
    generation_time = None

    if task.start_time:
        if task.end_time:
            time_elapsed = round(task.end_time - task.start_time, 2)
            generation_time = time_elapsed
        else:
            time_elapsed = round(current_time - task.start_time, 2)
            if task.status == TaskStatus.GENERATING and progress > 0:
                # Estimate remaining time based on progress
                time_remaining = round((time_elapsed / progress) * (1 - progress), 2)

    response = {
        "status": task.status.value,
        "progress": progress,
        "total_steps": task.total_steps,
        "current_step": task.current_step,
        "time_elapsed": time_elapsed,
        "time_remaining": time_remaining,
        "generation_time": generation_time
    }

    if task.error:
        response["error"] = task.error

    # Add detailed status message
    status_message = "Initializing..."
    if task.status == TaskStatus.LOADING_MODEL:
        status_message = "Loading model..."
    elif task.status == TaskStatus.GENERATING:
        status_message = f"Generating image (Step {task.current_step}/{task.total_steps})"
        if time_remaining:
            status_message += f" - {time_remaining}s remaining"
    elif task.status == TaskStatus.COMPLETED:
        status_message = f"Generation completed in {generation_time}s"
    elif task.status == TaskStatus.FAILED:
        status_message = f"Generation failed: {task.error}"

    response["status_message"] = status_message

    return response

from fastapi import FastAPI
from pydantic import BaseModel

class InferenceRequest(BaseModel):
    model_id: str
    prompt: str
    output_path: str

@app.post("/inference")
async def generate_image(request: InferenceRequest):
    try:
        adapter = model_manager.get_model(request.model_id)
        image = adapter.generate(prompt=request.prompt)
        image.save(request.output_path, format='PNG')
        return {"message": f"Image saved to {request.output_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Local inference or run FastAPI server")
    parser.add_argument('--model-id', type=str, help='Model ID to use for inference')
    parser.add_argument('--prompt', type=str, help='Prompt for image generation')
    parser.add_argument('--output-path', type=str, help='Path to save the generated image')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the FastAPI server on (default: 8000)')

    # Parse arguments
    args = parser.parse_args()

    if args.model_id and args.prompt and args.output_path:
        # Perform local inference
        print(f"Performing inference with model {args.model_id} and prompt '{args.prompt}'")
        adapter = model_manager.get_model(args.model_id)
        image = adapter.generate(prompt=args.prompt)
        image.save(args.output_path, format='PNG')
        print(f"Image saved to {args.output_path}")
    else:
        # Start FastAPI server
        import uvicorn
        print(f"Starting FastAPI server on port {args.port}...")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
