import base64
from fastapi import FastAPI, Form, BackgroundTasks, Request
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

# Load model configurations from YAML file
def load_model_config():
    config_path = BASE_DIR / 'config' / 'model_parameters.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

            # Convert torch dtype strings to actual torch dtypes
            for model_config in config['model_configs'].values():
                if 'torch_dtype' in model_config:
                    if model_config['torch_dtype'] == 'float16':
                        model_config['torch_dtype'] = torch.float16
                    elif model_config['torch_dtype'] == 'float32':
                        model_config['torch_dtype'] = torch.float32
                # Add revision field if not present
                if 'revision' not in model_config:
                    model_config['revision'] = None

            return config['model_parameters'], config['model_configs']
    except Exception as e:
        print(f"Error loading model configurations from {config_path}: {e}")
        raise

# Load both model parameters and configurations
MODEL_PARAMETERS, MODEL_CONFIGS = load_model_config()

# Global variables for model management
class ModelManager:
    def __init__(self):
        self.models = {k: None for k in MODEL_CONFIGS.keys()}
        self.last_used: Dict[str, float] = {k: 0 for k in MODEL_CONFIGS.keys()}
        self.lock = Lock()
        self.cleanup_interval = 600  # 10 minutes

    def get_model(self, model_id: str):
        """Get or load a model pipeline

        Special handling for different model types:
        - Lumina models: Use Karras sigmas and custom VAE
        - FLUX models: Use custom pipeline
        - Standard models: Regular pipeline with safety checker
        """
        with self.lock:
            if model_id not in self.models:
                return None
            self._update_last_used(model_id)
            if self.models[model_id] is None:
                self.models[model_id] = self._load_model_pipeline(model_id)
            return self.models[model_id]

    def _update_last_used(self, model_id: str):
        self.last_used[model_id] = time.time()

    def _load_model_pipeline(self, model_id: str):
        config = MODEL_CONFIGS[model_id]
        repo_id = config['repo_id']
        other_config = {k: v for k, v in config.items() if k != 'repo_id'}
        if model_id == 'sd-v3.5':
            return self._load_sd_v3_5(repo_id, other_config)
        elif model_id == 'animagine-xl-4':
            return self._load_animagine_xl_4(repo_id, other_config)
        elif model_id == 'lumina-2':
            return self._load_lumina_2(repo_id, other_config)
        elif model_id.startswith('flux'):
            return self._load_flux(repo_id, other_config)
        else:
            return self._load_default_model(config, repo_id, other_config, model_id)

    def _load_sd_v3_5(self, repo_id, other_config):
        print(f"Loading SD 3.5 model from {repo_id}")
        if other_config.get('use_quantized') == True:
            try:
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
                pipeline = pipeline.to('cuda')
                pipeline.enable_model_cpu_offload()
                return pipeline
            except Exception as e:
                print(f"Error loading SD 3.5 model in quant mode: {str(e)}")
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")
                raise
        else:
            print(f"Loading non-quantized SD 3.5 model from {repo_id}")
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                repo_id,**other_config,
            )
            pipeline = pipeline.to('cuda')
            pipeline.enable_model_cpu_offload()
            pipeline.vae.enable_tiling()

            return pipeline

    def _load_animagine_xl_4(self, config, repo_id, other_config):
        from diffusers import StableDiffusionXLPipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            repo_id, **other_config
        )
        pipeline.to('cuda')
        return pipeline

    def _load_lumina_2(self, config, repo_id, other_config):
        from diffusers import Lumina2Text2ImgPipeline
        pipe = Lumina2Text2ImgPipeline.from_pretrained(repo_id, **other_config)
        pipe = pipe.to('cuda')
        pipe.enable_model_cpu_offload()
        return pipe

    def _load_flux(self, repo_id, other_config):
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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/model-config/{model_name}")
async def get_model_config(model_name: str):
    """Get model-specific configuration parameters"""
    if model_name not in MODEL_PARAMETERS:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    # Get base parameters
    config = MODEL_PARAMETERS[model_name].copy()

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
    model_config = MODEL_CONFIGS.get(model_name, {})

    # Add UI controls to config
    config['ui_controls'] = ui_controls

    # Add quantization option if enabled in UI controls
    if ui_controls.get('show_quantization'):
        config['use_quantized'] = {
            'type': 'boolean',
            'default': model_config.get('use_quantized', False),
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
    model_config = MODEL_CONFIGS.get(model_name, {})

    # Use model-specific negative prompt if available and no user-provided negative prompt
    if not negative_prompt and model_config.get('negative_prompt'):
        negative_prompt = model_config['negative_prompt']

    # Add aesthetic score for models that support it
    extra_args = {}
    if model_config.get('requires_aesthetic_score'):
        extra_args['aesthetic_score'] = model_config.get('aesthetic_score', 9.0)

    # Validate model exists
    if model_name not in MODEL_PARAMETERS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model_name}")

    # Get model parameters
    params = MODEL_PARAMETERS[model_name]

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
    task.start_time = datetime.now()
    task.total_steps = num_inference_steps

    try:
        # Update model config if use_quantized is specified
        if use_quantized is not None and model_name == 'sd-v3.5':
            MODEL_CONFIGS['sd-v3.5']['use_quantized'] = use_quantized

        # Get model
        task.status = TaskStatus.LOADING_MODEL
        pipe = model_manager.get_model(model_name)
        if pipe is None:
            task.status = TaskStatus.FAILED
            task.error = f"Failed to load model {model_name}"
            return JSONResponse(
                status_code=500,
                content={"error": task.error, "task_id": task_id}
            )


        # Generate image with model-specific parameters
        task.status = TaskStatus.GENERATING
        if model_name == 'sd-v3.5':
            # Print debug info
            print(f"Generating with SD 3.5:\nPrompt: {prompt}\nNegative: {negative_prompt}\nDimensions: {width}x{height}\nSteps: {num_inference_steps}\nGuidance: {guidance_scale}")

            # Print pipeline info
            print(f"Pipeline components:\n{pipe.components}")
            print(f"Pipeline device: {pipe.device}")

            # Generate with basic parameters first
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                # callback=callback,
                # callback_steps=1
            ).images[0]
        else:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                # callback=callback,
                # callback_steps=1
            ).images[0]

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Update task status
        task.status = TaskStatus.COMPLETED
        task.end_time = datetime.now()
        task.progress = 1.0

        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={
                "X-Task-Id": task_id,
                "X-Generation-Time": str((task.end_time - task.start_time).total_seconds())
            }
        )

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Full error traceback:\n{error_traceback}")

        # Update task status on error
        task.status = TaskStatus.FAILED
        task.error = str(e)
        task.end_time = datetime.now()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": error_traceback,
                "task_id": task_id,
                "generation_time": (task.end_time - task.start_time).total_seconds()
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
