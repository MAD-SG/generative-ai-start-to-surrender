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
    start_time: datetime = None
    end_time: datetime = None
    error: str = None

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

        # Configure quantization if requested and supported
        if use_quantized:
            other_config['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=other_config.get('torch_dtype', torch.float16)
            )

        # Handle different model types
        if self.model_id == 'sd-v3.5':
            if use_quantized:
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
            else:
                pipeline = StableDiffusion3Pipeline.from_pretrained(repo_id, **other_config)
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

        else:
            return AutoPipelineForText2Image.from_pretrained(repo_id, **other_config)

    def load(self, use_quantized: bool = False):
        if self.pipeline is None:
            self.pipeline = self._create_pipeline(use_quantized)
            self.pipeline.to(self.device)

            # Apply optimizations
            if self.model_config.get('use_dpm_solver', False):
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipeline.scheduler.config
                )
        return self.pipeline

    def unload(self):
        if self.pipeline is not None:
            self.pipeline = self.pipeline.to("cpu")
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()

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
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self.model_id in ['flux1-schnell']:
            return self.pipeline(
                prompt=prompt,
                # negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs
            ).images[0]
        else:
            return self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs
            ).images[0]

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

        # Get model adapter
    task.status = TaskStatus.LOADING_MODEL
    adapter = model_manager.get_model(model_name, use_quantized)
    if adapter is None:
        task.status = TaskStatus.FAILED
        task.error = f"Failed to load model {model_name}"
        return JSONResponse(
            status_code=500,
            content={"error": task.error, "task_id": task_id}
        )
    print(f"Loaded model {model_name}")

    # Generate image
    task.status = TaskStatus.GENERATING
    print(f"Generating with {model_name}:\nPrompt: {prompt}\nNegative: {negative_prompt}\nDimensions: {width}x{height}\nSteps: {num_inference_steps}\nGuidance: {guidance_scale}")
    task.start_time = datetime.now()

    # Generate with model-specific parameters
    image = adapter.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        # callback=callback,
        # callback_steps=1,
        **extra_args
    )

    # Convert to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Update task status
    task.status = TaskStatus.COMPLETED
    task.end_time = datetime.now()
    print(f"Task {task_id} completed in {task.end_time - task.start_time}")
    task.progress = 1.0

    return StreamingResponse(
        img_byte_arr,
        media_type="image/png",
        headers={
            "X-Task-Id": task_id,
            "X-Generation-Time": str((task.end_time - task.start_time).total_seconds())
        }
    )

@app.get("/task/{task_id}")
async def get_task_progress(task_id: str):
    """Get the progress of a specific task"""
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    response = {
        "status": task.status.value,
        "progress": task.progress,
        "total_steps": task.total_steps,
        "current_step": task.current_step,
    }

    if task.error:
        response["error"] = task.error

    if task.start_time:
        response["start_time"] = task.start_time.isoformat()

    if task.end_time:
        response["end_time"] = task.end_time.isoformat()
        response["generation_time"] = (task.end_time - task.start_time).total_seconds()

    return response

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Stable Diffusion Web Demo')
    parser.add_argument('--port', type=int, default=30982,
                        help='Port to run the server on (default: 30982)')

    args = parser.parse_args()

    print(f"Starting server on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
