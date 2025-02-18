class ImageGenerator {
    constructor() {
        this.currentTaskId = null;
        this.progressInterval = null;
        this.currentImageUrl = null;
        this.prompts = [
            "A majestic cosmic landscape, ethereal nebulas in vibrant purples and blues, floating islands with crystalline waterfalls, rays of golden light piercing through cosmic clouds, ultra detailed, cinematic lighting, 8k",
            "A mystical library with floating books, glowing particles of magic, intricate architecture, warm golden light streaming through stained glass windows, photorealistic, detailed textures, volumetric lighting",
            "An enchanted forest at twilight, bioluminescent flowers, ethereal mist, ancient twisted trees, moonlight filtering through branches, fireflies, magical atmosphere, ultra detailed, cinematic composition",
            "A futuristic cyberpunk cityscape at night, neon lights reflecting in rain puddles, holographic advertisements, flying vehicles, towering skyscrapers, detailed architecture, moody atmosphere, 8k resolution",
            "An underwater palace with bioluminescent sea creatures, crystal domes, ancient marble architecture, schools of colorful fish, rays of light piercing through water, detailed coral reefs, magical atmosphere",
            "A steampunk airship floating through golden clouds at sunset, intricate brass machinery, billowing steam, Victorian architecture, dramatic lighting, highly detailed metalwork, cinematic composition",
            "A crystalline ice palace in the northern lights, aurora borealis reflecting off transparent walls, snowflakes in the air, magical atmosphere, detailed ice formations, volumetric lighting, 8k resolution",
            "A magical garden with giant mushrooms, fairy lights, flowing streams, rainbow flowers, butterflies, mystical atmosphere, ultra detailed, soft lighting, photorealistic textures"
        ];
        this.setupEventListeners();
        this.setupRangeInputs();
        this.setupImageModal();
        this.setupBackgroundButtons();
        this.setupPromptSuggestions();
    }

    setupEventListeners() {
        // Form submission
        const form = document.getElementById('generate-form');
        if (form) {
            form.addEventListener('submit', (e) => this.handleSubmit(e));
        }

        // Model selection
        const modelSelect = document.getElementById('model_name');
        if (modelSelect) {
            modelSelect.addEventListener('change', () => this.updateModelConfig(modelSelect.value));
            // Initialize with current model
            this.updateModelConfig(modelSelect.value);
        }

        // Random prompt button
        const randomPromptBtn = document.getElementById('random-prompt');
        if (randomPromptBtn) {
            randomPromptBtn.addEventListener('click', () => this.setRandomPrompt());
        }

        // Clear result image when starting a new prompt
        const promptInput = document.getElementById('prompt');
        if (promptInput) {
            promptInput.addEventListener('input', () => {
                const resultImage = document.getElementById('result-image');
                const placeholder = document.getElementById('placeholder-message');
                const generationInfo = document.getElementById('generation-info');
                if (resultImage && placeholder && generationInfo) {
                    resultImage.style.display = 'none';
                    placeholder.style.display = 'block';
                    generationInfo.style.display = 'none';
                }
            });
        }

        // Setup fullscreen image click
        const resultImage = document.getElementById('result-image');
        if (resultImage) {
            resultImage.addEventListener('click', () => {
                if (this.currentImageUrl) {
                    this.showImageModal(this.currentImageUrl);
                }
            });
        }
    }

    setupRangeInputs() {
        // Setup range input value displays
        const rangeInputs = [
            { input: 'guidance_scale', display: 'guidance_scale_value', precision: 1 },
            { input: 'num_inference_steps', display: 'steps_value', precision: 0 }
        ];

        rangeInputs.forEach(({ input, display, precision }) => {
            const inputEl = document.getElementById(input);
            const displayEl = document.getElementById(display);
            if (inputEl && displayEl) {
                // Update display on input change
                inputEl.addEventListener('input', () => {
                    displayEl.textContent = Number(inputEl.value).toFixed(precision);
                });
                // Initialize display
                displayEl.textContent = Number(inputEl.value).toFixed(precision);
            }
        });
    }

    async updateModelConfig(modelName) {
        try {
            const response = await fetch(`/model-config/${modelName}`);
            const config = await response.json();

            // Handle UI visibility based on ui_controls
            if (config.ui_controls) {
                // Main form inputs
                const controls = {
                    'show_prompt': ['prompt'],
                    'show_negative_prompt': ['negative_prompt'],
                    'show_dimensions': ['height', 'width'],
                    'show_guidance_scale': ['guidance_scale'],
                    'show_steps': ['num_inference_steps'],
                };

                // Update visibility for each control
                Object.entries(controls).forEach(([control, elements]) => {
                    elements.forEach(elementId => {
                        const element = document.getElementById(elementId)?.closest('.input-group');
                        if (element) {
                            element.style.display = config.ui_controls[control] ? 'block' : 'none';
                        }
                    });
                });
            }

            // Update extra settings section
            const extraSettingsContainer = document.getElementById('extra-settings');
            if (extraSettingsContainer) {
                extraSettingsContainer.innerHTML = ''; // Clear existing settings

                // Show quantization option if enabled
                if (config.ui_controls?.show_quantization) {
                    const div = document.createElement('div');
                    div.className = 'form-check mb-2';
                    div.innerHTML = `
                        <input type="checkbox" name="use_quantized"
                               class="form-check-input"
                               id="use_quantized"
                               ${config.use_quantized?.default ? 'checked' : ''}>
                        <label class="form-check-label" for="use_quantized">
                            ${config.use_quantized?.description || 'Use quantization'}
                        </label>
                    `;
                    extraSettingsContainer.appendChild(div);
                }
            }

            // Update parameter ranges and values for visible inputs
            ['height', 'width', 'guidance_scale', 'num_inference_steps'].forEach(param => {
                const input = document.getElementById(param);
                if (input && config[param]) {
                    input.min = config[param].min;
                    input.max = config[param].max;
                    input.step = config[param].step;
                    input.value = config[param].default;
                }
            });
        } catch (error) {
            console.error('Error updating model config:', error);
            const errorMsg = document.getElementById('error-message');
            if (errorMsg) {
                errorMsg.textContent = `Error loading model configuration: ${error.message}`;
                errorMsg.style.display = 'block';
            }
        }
    }

    updateParameterRanges(config) {
        // Update ranges for height, width, etc.
        const params = ['height', 'width', 'guidance_scale', 'num_inference_steps'];
        params.forEach(param => {
            const input = document.getElementById(param);
            if (input && config[param]) {
                input.min = config[param].min;
                input.max = config[param].max;
                input.step = config[param].step;
                input.value = config[param].default;
            }
        });
    }

    async handleSubmit(e) {
        e.preventDefault();
        const form = e.target;
        const formData = new FormData(form);

        // Disable form while generating
        const submitButton = form.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.innerHTML;
        submitButton.disabled = true;
        submitButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Generating...';

        try {
            await this.startGeneration(formData);
        } finally {
            // Re-enable form
            submitButton.disabled = false;
            submitButton.innerHTML = originalButtonText;
        }
    }

    async startGeneration(formData) {
        this.showProgress();

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Generation failed');
            }

            this.currentTaskId = response.headers.get('X-Task-Id');
            const generationTime = response.headers.get('X-Generation-Time');

            // Start progress polling
            this.startProgressPolling();

            // Handle the image
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            this.displayResult(imageUrl, generationTime);
        } catch (error) {
            this.handleError(error);
        }
    }

    startProgressPolling() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }

        this.progressInterval = setInterval(() => this.checkProgress(), 500);
    }

    async checkProgress() {
        if (!this.currentTaskId) return;

        try {
            const response = await fetch(`/task/${this.currentTaskId}`);
            if (!response.ok) {
                throw new Error(`Failed to check progress: ${response.status}`);
            }

            const progress = await response.json();
            this.updateProgressUI(progress);

            if (['completed', 'failed'].includes(progress.status)) {
                clearInterval(this.progressInterval);
                this.progressInterval = null;

                if (progress.status === 'failed') {
                    this.handleError(new Error(progress.error || 'Generation failed'));
                }
            }
        } catch (error) {
            console.error('Error checking progress:', error);
            this.handleError(error);
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }

    showProgress() {
        const resultImage = document.getElementById('result-image');
        const placeholder = document.getElementById('placeholder-message');
        const progressContainer = document.getElementById('progress-container');
        const progressBar = document.getElementById('progress-bar');
        const generationInfo = document.getElementById('generation-info');

        if (resultImage) resultImage.style.display = 'none';
        if (placeholder) placeholder.style.display = 'none';
        if (progressContainer) progressContainer.style.display = 'block';
        if (progressBar) {
            progressBar.style.width = '0%';
            progressBar.setAttribute('aria-valuenow', '0');
        }
        if (generationInfo) generationInfo.style.display = 'none';
    }

    updateProgressUI(progress) {
        const progressBar = document.getElementById('progress-bar');
        if (!progressBar) return;

        const percent = Math.round((progress.step / progress.total_steps) * 100);
        progressBar.style.width = `${percent}%`;
        progressBar.setAttribute('aria-valuenow', percent);
        progressBar.textContent = `${percent}%`;
    }

    displayResult(imageUrl, generationTime) {
        const resultImage = document.getElementById('result-image');
        const placeholder = document.getElementById('placeholder-message');
        const progressContainer = document.getElementById('progress-container');
        const generationInfo = document.getElementById('generation-info');
        const downloadLink = document.getElementById('download-link');
        const modalDownload = document.getElementById('modal-download');
        const setBackground = document.getElementById('set-background');
        const generationTimeText = document.getElementById('generation-time');

        // Store current image URL
        this.currentImageUrl = imageUrl;

        if (resultImage) {
            resultImage.src = imageUrl;
            resultImage.style.display = 'block';
            resultImage.alt = document.getElementById('prompt')?.value || 'Generated image';

            // Set up download links in both main view and modal
            if (downloadLink) {
                this.setupDownloadLink(downloadLink, imageUrl);
                downloadLink.style.display = 'inline-block';
            }
            if (modalDownload) {
                this.setupDownloadLink(modalDownload, imageUrl);
            }
            if (setBackground) {
                setBackground.style.display = 'inline-block';
            }
        }
        if (placeholder) placeholder.style.display = 'none';
        if (progressContainer) progressContainer.style.display = 'none';
        if (generationInfo) generationInfo.style.display = 'block';
        if (generationTimeText) generationTimeText.textContent = `Generation completed in ${generationTime} seconds`;
    }

    setupImageModal() {
        // Initialize Bootstrap modal
        const modalElement = document.getElementById('imageModal');
        this.imageModal = new bootstrap.Modal(modalElement);
        
        // Setup click handler for result image
        const resultImage = document.getElementById('result-image');
        if (resultImage) {
            resultImage.addEventListener('click', () => {
                if (this.currentImageUrl) {
                    this.showImageModal(this.currentImageUrl);
                }
            });
        }

        // Setup keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (!this.imageModal._isShown) return;

            switch(e.key) {
                case 'Escape':
                    this.imageModal.hide();
                    break;
                case 'ArrowLeft':
                    // Could be used for previous image if implementing gallery
                    break;
                case 'ArrowRight':
                    // Could be used for next image if implementing gallery
                    break;
            }
        });

        // Setup modal events
        modalElement.addEventListener('shown.bs.modal', () => {
            // Focus handling or additional setup when modal is shown
        });

        modalElement.addEventListener('hidden.bs.modal', () => {
            // Cleanup or state reset when modal is hidden
        });
    }

    setupDownloadLink(link, imageUrl) {
        // Get model name and prompt for filename
        const modelSelect = document.getElementById('model_name');
        const prompt = document.getElementById('prompt')?.value || '';
        const modelName = modelSelect?.options[modelSelect.selectedIndex]?.text?.replace(/[^a-z0-9]/gi, '_').toLowerCase() || 'unknown_model';
        
        // Create a clean filename from the prompt
        const cleanPrompt = prompt.slice(0, 30).replace(/[^a-z0-9]/gi, '_').toLowerCase() || 'generated';
        const timestamp = new Date().toISOString().replace(/[^0-9]/g, '').slice(0, 14);
        const filename = `${cleanPrompt}_${modelName}_${timestamp}.png`;

        // Setup the download link
        link.href = imageUrl;
        link.download = filename;
        link.style.display = 'inline-block';

        // Add click handler to handle download errors
        link.onclick = async (e) => {
            try {
                // Fetch the image as blob
                const response = await fetch(imageUrl);
                if (!response.ok) throw new Error('Network response was not ok');
                
                const blob = await response.blob();
                const blobUrl = window.URL.createObjectURL(blob);
                
                // Create a temporary link and click it
                const tempLink = document.createElement('a');
                tempLink.href = blobUrl;
                tempLink.download = filename;
                document.body.appendChild(tempLink);
                tempLink.click();
                document.body.removeChild(tempLink);
                
                // Clean up the blob URL
                window.URL.revokeObjectURL(blobUrl);
            } catch (error) {
                console.error('Download failed:', error);
                alert('Failed to download image. Please try again.');
            }
            return false; // Prevent default link behavior
        };
    }

    showImageModal(imageUrl) {
        const modalImage = document.getElementById('modal-image');
        const modalDownload = document.getElementById('modal-download');
        const modalTitle = document.getElementById('imageModalLabel');
        
        if (modalImage) {
            modalImage.src = imageUrl;
            
            // Handle image loading
            modalImage.onload = () => {
                // Add loading indicator if needed
                modalImage.style.display = 'block';
            };
            
            modalImage.onerror = () => {
                console.error('Failed to load image in modal');
            };
        }

        if (modalDownload) {
            this.setupDownloadLink(modalDownload, imageUrl);
        }

        if (modalTitle) {
            const prompt = document.getElementById('prompt')?.value;
            modalTitle.textContent = prompt ? `Generated: ${prompt.slice(0, 50)}${prompt.length > 50 ? '...' : ''}` : 'Generated Image';
        }
        
        this.imageModal.show();
    }

    setupBackgroundButtons() {
        const setBackgroundButtons = ['set-background', 'modal-set-background'];
        
        setBackgroundButtons.forEach(buttonId => {
            const button = document.getElementById(buttonId);
            if (button) {
                button.addEventListener('click', () => this.setAsBackground());
            }
        });
    }

    setupPromptSuggestions() {
        // Setup click handlers for suggestion buttons
        const suggestionButtons = document.querySelectorAll('.suggestion-btn');
        suggestionButtons.forEach(button => {
            button.addEventListener('click', () => {
                const promptArea = document.getElementById('prompt');
                if (promptArea) {
                    promptArea.value = button.textContent.trim();
                    // Trigger height adjustment
                    promptArea.style.height = 'auto';
                    promptArea.style.height = promptArea.scrollHeight + 'px';
                    // Optional: add focus and scroll to view
                    promptArea.focus();
                }
            });
        });

        // Setup auto-resize for prompt textarea
        const promptArea = document.getElementById('prompt');
        if (promptArea) {
            // Initial height adjustment
            promptArea.style.height = 'auto';
            promptArea.style.height = promptArea.scrollHeight + 'px';

            // Adjust height on input
            promptArea.addEventListener('input', () => {
                promptArea.style.height = 'auto';
                promptArea.style.height = promptArea.scrollHeight + 'px';
            });
        }
    }

    setRandomPrompt() {
        const promptArea = document.getElementById('prompt');
        if (promptArea && this.prompts.length > 0) {
            const randomIndex = Math.floor(Math.random() * this.prompts.length);
            promptArea.value = this.prompts[randomIndex];
            // Trigger height adjustment
            promptArea.style.height = 'auto';
            promptArea.style.height = promptArea.scrollHeight + 'px';
            // Optional: add focus and scroll to view
            promptArea.focus();
        }
    }

    setAsBackground() {
        if (!this.currentImageUrl) return;

        // Update body background
        document.body.style.backgroundImage = `url('${this.currentImageUrl}')`;
        
        // Show success message
        const toast = document.createElement('div');
        toast.className = 'toast-message';
        toast.innerHTML = `
            <i class="fas fa-check-circle me-2"></i>
            Background updated successfully
        `;
        document.body.appendChild(toast);

        // Remove toast after animation
        setTimeout(() => {
            toast.remove();
        }, 3000);

        // Close modal if open
        if (this.imageModal._isShown) {
            this.imageModal.hide();
        }
    }

    handleError(error) {
        console.error('Generation error:', error);
        const errorMsg = document.getElementById('error-message');
        const progressBar = document.getElementById('progress-bar');
        const placeholder = document.getElementById('placeholder-message');

        if (errorMsg) {
            errorMsg.querySelector('.error-text').textContent = error.message || 'An unexpected error occurred';
            errorMsg.style.display = 'block';
        }
        if (progressBar) progressBar.style.display = 'none';
        if (placeholder) placeholder.style.display = 'block';
    }

    updateProgressUI(progress) {
        const progressBar = document.getElementById('progress-bar');
        const progressStatus = document.getElementById('progress-status');

        if (progressBar && progressStatus) {
            progressBar.style.width = `${progress.progress * 100}%`;

            let statusText = '';
            switch (progress.status) {
                case 'loading_model':
                    statusText = 'Loading model...';
                    break;
                case 'generating':
                    statusText = `Generating image... Step ${progress.current_step}/${progress.total_steps}`;
                    break;
                case 'completed':
                    statusText = `Generation completed in ${progress.generation_time.toFixed(2)}s`;
                    break;
                case 'failed':
                    statusText = `Error: ${progress.error}`;
                    break;
                default:
                    statusText = progress.status;
            }
            progressStatus.textContent = statusText;
        }
    }

    showProgress() {
        const progressContainer = document.getElementById('progress-container');
        if (progressContainer) {
            progressContainer.style.display = 'block';
        }
    }

    hideProgress() {
        const progressContainer = document.getElementById('progress-container');
        if (progressContainer) {
            progressContainer.style.display = 'none';
        }
    }

    displayResult(imageUrl, generationTime) {
        const resultImage = document.getElementById('result-image');
        const timeInfo = document.getElementById('generation-time');

        if (resultImage) {
            resultImage.src = imageUrl;
            resultImage.style.display = 'block';
        }

        if (timeInfo) {
            timeInfo.textContent = `Generation time: ${generationTime}s`;
        }
    }

    handleError(error) {
        const errorElement = document.getElementById('error-message');
        if (errorElement) {
            errorElement.textContent = error.error || 'An error occurred during image generation';
            errorElement.style.display = 'block';
        }
        this.hideProgress();
    }
}

// Initialize the generator when the page loads
window.addEventListener('DOMContentLoaded', () => {
    window.imageGenerator = new ImageGenerator();

    // Update form submission
    const form = document.getElementById('generate-form');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(form);
            window.imageGenerator.startGeneration(formData);
        });
    }
});
