/**
 * Manages all UI-related operations and state
 */
class UIManager {
    constructor() {
        this.models = [];
        this.activeModels = new Set();
        // Initialize all UI elements
        this.elements = {
            promptInput: document.getElementById('prompt-input'),
            negativePromptInput: document.getElementById('negative-prompt-input'),
            generateBtn: document.getElementById('generate-btn'),
            progressBar: document.getElementById('progress-bar'),
            progressContainer: document.getElementById('progress-container'),
            progressStatus: document.getElementById('progress-status'),
            progressTime: document.getElementById('progress-time'),
            progressDetails: document.getElementById('progress-details'),
            timeElapsed: document.getElementById('time-elapsed'),
            timeRemaining: document.getElementById('time-remaining'),
            errorMessage: document.getElementById('error-message'),
            resultImage: document.getElementById('result-image'),
            generationInfo: document.getElementById('generation-info'),
            downloadLink: document.getElementById('download-link'),
            setBackground: document.getElementById('set-background'),
            placeholderMessage: document.getElementById('placeholder-message'),
            modelList: document.getElementById('model-list'),
            saveModelChanges: document.getElementById('save-model-changes')
        };

        this.setupRangeInputs();
        this.setupModelControl();
        this.loadModelStates();
    }

    /**
     * Sets up range input listeners for real-time value updates
     */
    setupRangeInputs() {
        const rangeInputs = {
            'guidance_scale': 'guidance_scale_value',
            'num_inference_steps': 'steps_value'
        };

        Object.entries(rangeInputs).forEach(([inputId, valueId]) => {
            const input = document.getElementById(inputId);
            const value = document.getElementById(valueId);
            if (input && value) {
                input.addEventListener('input', () => {
                    value.textContent = input.value;
                });
            }
        });
    }

    /**
     * Gets all parameter values from UI controls
     * @returns {Object} Parameters object
     */
    getParameters() {
        // Debug: Check if we can find the negative prompt element
        const negativePromptEl = document.getElementById('negative-prompt-input');
        console.log('Found negative prompt element:', negativePromptEl);
        console.log('Negative prompt element value:', negativePromptEl?.value);

        // Get all form values with detailed logging
        const formValues = {};

        // Add prompt
        const promptEl = document.getElementById('prompt-input');
        formValues.prompt = promptEl?.value || '';
        console.log('Prompt value:', formValues.prompt);

        // Add negative prompt
        formValues.negative_prompt = negativePromptEl?.value || '';
        console.log('Negative prompt value:', formValues.negative_prompt);

        // Add other parameters
        formValues.width = parseInt(document.getElementById('width')?.value || '512');
        formValues.height = parseInt(document.getElementById('height')?.value || '512');
        formValues.guidance_scale = parseFloat(document.getElementById('guidance_scale')?.value || '7.5');
        formValues.num_inference_steps = parseInt(document.getElementById('num_inference_steps')?.value || '50');
        formValues.model_name = Array.from(this.activeModels)[0] || 'sd-v1.5';

        // Log the final object
        console.log('Complete form values object:', formValues);

        return formValues;
    }

    /**
     * Updates the progress bar and status
     * @param {number} percentage - Progress percentage (0-100)
     * @param {string} status - Status message
     */
    updateProgress(percentage, status = '') {
        this.elements.progressContainer.style.display = 'block';
        this.elements.progressBar.style.width = `${percentage}%`;
        this.elements.progressBar.setAttribute('aria-valuenow', percentage);

        if (status) {
            this.elements.progressStatus.textContent = status;
        }
    }

    /**
     * Updates the generation time information
     * @param {number} elapsed - Elapsed time in seconds
     * @param {number} remaining - Remaining time in seconds
     */
    updateTimeInfo(elapsed, remaining) {
        this.elements.timeElapsed.textContent = `${elapsed}s`;
        this.elements.timeRemaining.textContent = remaining ? `${remaining}s` : 'Calculating...';
    }

    /**
     * Toggles the generate button state
     * @param {boolean} disabled - Whether to disable the button
     */
    toggleGenerateButton(disabled) {
        this.elements.generateBtn.disabled = disabled;
        this.elements.generateBtn.innerHTML = disabled ?
            '<i class="fas fa-spinner fa-spin me-2"></i>Generating...' :
            'Generate Image';
    }

    /**
     * Displays an error message
     * @param {string} message - Error message to display
     */
    showError(message) {
        this.elements.errorMessage.style.display = 'block';
        this.elements.errorMessage.querySelector('.error-text').textContent = message;
    }

    /**
     * Hides the error message
     */
    hideError() {
        this.elements.errorMessage.style.display = 'none';
    }

    /**
     * Displays the generated image and related information
     * @param {string} imageUrl - URL of the generated image
     * @param {Object} generationInfo - Information about the generation
     */
    displayResult(imageUrl, generationInfo) {
        this.elements.placeholderMessage.style.display = 'none';
        this.elements.resultImage.src = imageUrl;
        this.elements.resultImage.style.display = 'block';
        this.elements.downloadLink.href = imageUrl;
        this.elements.downloadLink.style.display = 'block';
        this.elements.setBackground.style.display = 'block';

        // Update generation info
        this.elements.generationInfo.style.display = 'block';
        this.elements.generationInfo.querySelector('#generation-time').textContent =
            `Generated in ${generationInfo.totalTime.toFixed(1)}s`;
        this.elements.generationInfo.querySelector('#generation-params').textContent =
            `Parameters: ${JSON.stringify(generationInfo.parameters, null, 2)}`;
    }

    /**
     * Resets the UI to its initial state
     */
    reset() {
        this.elements.progressContainer.style.display = 'none';
        this.elements.progressBar.style.width = '0%';
        this.elements.progressBar.setAttribute('aria-valuenow', 0);
        this.elements.progressStatus.textContent = 'Initializing...';
        this.hideError();
    }

    /**
     * Sets up model control functionality
     */
    setupModelControl() {
        // Fetch available models
        fetch('/api/models')
            .then(response => response.json())
            .then(models => {
                this.models = models;
                this.renderModelList();
            })
            .catch(error => {
                console.error('Error fetching models:', error);
                this.showError('无法加载模型列表');
            });

        // Handle save changes
        this.elements.saveModelChanges.addEventListener('click', () => {
            const modelStates = {};
            this.models.forEach(model => {
                const checkbox = document.getElementById(`model-${model.id}`);
                if (checkbox) {
                    modelStates[model.id] = checkbox.checked;
                }
            });

            const payload = { states: modelStates };
            console.log('Sending model states:', payload);

            fetch('/api/models/state', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            })
            .then(response => {
                console.log('Response status:', response.status);
                return response.json();
            })
            .then(result => {
                console.log('Response result:', result);
                if (result.success) {
                    this.activeModels = new Set(
                        Object.entries(modelStates)
                            .filter(([_, active]) => active)
                            .map(([id]) => id)
                    );
                    this.saveModelStates();

                    // 正确关闭模态框并清理背景
                    const modalElement = document.getElementById('modelControlModal');
                    const modalInstance = bootstrap.Modal.getInstance(modalElement);
                    modalInstance.hide();

                    // 确保背景遮罩被移除
                    modalElement.addEventListener('hidden.bs.modal', () => {
                        const backdrop = document.querySelector('.modal-backdrop');
                        if (backdrop) {
                            backdrop.remove();
                        }
                        document.body.classList.remove('modal-open');
                        document.body.style.overflow = '';
                        document.body.style.paddingRight = '';
                    }, { once: true });
                } else {
                    throw new Error(result.message || '保存失败');
                }
            })
            .catch(error => {
                console.error('Error saving model states:', error);
                this.showError('保存模型状态失败');
            });
        });
    }

    /**
     * Renders the model list in the control panel
     */
    renderModelList() {
        this.elements.modelList.innerHTML = '';

        this.models.forEach(model => {
            const item = document.createElement('div');
            item.className = 'list-group-item d-flex justify-content-between align-items-center';

            const label = document.createElement('label');
            label.className = 'form-check-label flex-grow-1 ms-2';
            label.htmlFor = `model-${model.id}`;
            label.textContent = model.display_name;

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.className = 'form-check-input';
            checkbox.id = `model-${model.id}`;
            checkbox.checked = this.activeModels.has(model.id);

            item.appendChild(checkbox);
            item.appendChild(label);

            if (model.description) {
                const description = document.createElement('small');
                description.className = 'text-muted d-block ms-4';
                description.textContent = model.description;
                item.appendChild(description);
            }

            this.elements.modelList.appendChild(item);
        });
    }

    /**
     * Saves model states to localStorage
     */
    saveModelStates() {
        localStorage.setItem('activeModels', JSON.stringify(Array.from(this.activeModels)));
    }

    /**
     * Loads model states from localStorage
     */
    loadModelStates() {
        try {
            const savedStates = localStorage.getItem('activeModels');
            if (savedStates) {
                this.activeModels = new Set(JSON.parse(savedStates));
            }
        } catch (error) {
            console.error('Error loading model states:', error);
        }
    }
}

export default UIManager;
