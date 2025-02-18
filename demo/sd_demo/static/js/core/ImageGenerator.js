/**
 * Handles image generation requests and manages generation state
 */
class ImageGenerator {
    /**
     * @param {string} apiEndpoint - The endpoint for image generation API
     * @param {UIManager} uiManager - Instance of UIManager for UI updates
     */
    constructor(apiEndpoint, uiManager) {
        this.apiEndpoint = apiEndpoint;
        this.uiManager = uiManager;
        this.currentTaskId = null;
        this.isGenerating = false;
        this.startTime = null;
        this.progressInterval = null;
        this.lastParameters = null;
    }

    /**
     * Initiates image generation with the given parameters
     * @param {Object} parameters - Generation parameters including prompt
     * @returns {Promise<Object>} Generation result
     */
    async generateImage(parameters) {
        try {
            if (this.isGenerating) {
                throw new Error('Generation already in progress');
            }

            this.isGenerating = true;
            this.startTime = Date.now();
            this.uiManager.reset();
            this.uiManager.toggleGenerateButton(true);

            const formData = new FormData();
            Object.entries(parameters).forEach(([key, value]) => {
                formData.append(key, value);
                console.log(`Adding parameter: ${key} = ${value}`);
            });

            console.log('Sending request to /generate');
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                },
                body: formData,
                credentials: 'same-origin'
            });

            console.log('Response status:', response.status);
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Error response:', errorText);
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.currentTaskId = result.task_id;
            this.lastParameters = parameters;
            this.startProgressTracking();

            return result;
        } catch (error) {
            this.handleError(error);
            throw error;
        }
    }

    /**
     * Starts tracking generation progress
     */
    startProgressTracking() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }

        this.progressInterval = setInterval(async () => {
            try {
                const status = await this.checkProgress();

                if (!status) return;

                if (status.status === 'completed') {
                    // 停止进度检查
                    clearInterval(this.progressInterval);
                    this.progressInterval = null;

                    // 获取并显示图片
                    await this.fetchGeneratedImage();
                } else if (status.status === 'failed') {
                    // 停止进度检查
                    clearInterval(this.progressInterval);
                    this.progressInterval = null;

                    throw new Error(status.error || 'Generation failed');
                }
            } catch (error) {
                // 发生错误时也要停止进度检查
                clearInterval(this.progressInterval);
                this.progressInterval = null;

                this.handleError(error);
            }
        }, 1000);
    }

    /**
     * Checks the current progress of image generation
     * @returns {Promise<Object>} Progress status
     */
    async checkProgress() {
        if (!this.currentTaskId) return null;

        console.log('Checking progress for task:', this.currentTaskId);
        const response = await fetch(`/task/${this.currentTaskId}`);
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Error response:', errorText);
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const status = await response.json();
        console.log('Progress status:', status);

        // 更新进度条
        if (status.progress !== undefined) {
            this.uiManager.updateProgress(status.progress * 100, status.status_message);
        }

        return status;
    }

    async fetchGeneratedImage() {
        if (!this.currentTaskId) return;

        try {
            const response = await fetch(`/image/${this.currentTaskId}`);
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Error fetching image:', errorText);
                throw new Error(`Failed to fetch image: ${response.status}`);
            }

            const imageBlob = await response.blob();
            const imageUrl = URL.createObjectURL(imageBlob);

            // 更新UI显示图片
            this.uiManager.displayResult(imageUrl, {
                totalTime: (Date.now() - this.startTime) / 1000,
                parameters: this.lastParameters
            });

            // 清理状态
            this.isGenerating = false;
            this.uiManager.toggleGenerateButton(false);
        } catch (error) {
            console.error('Error fetching generated image:', error);
            this.handleError(error);
        }
    }

    /**
     * Handles successful generation completion
     * @param {Object} status - Final generation status
     */
    handleCompletion(status) {
        this.cleanup();

        const totalTime = (Date.now() - this.startTime) / 1000;
        this.uiManager.displayResult(status.image_url, {
            totalTime,
            parameters: status.parameters
        });
    }

    /**
     * Handles generation errors
     * @param {Error} error - Error object
     */
    handleError(error) {
        this.cleanup();
        this.uiManager.showError(error.message);
        console.error('Generation error:', error);
    }

    /**
     * Cleans up generation state
     */
    cleanup() {
        this.isGenerating = false;
        this.currentTaskId = null;
        this.startTime = null;
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
        this.uiManager.toggleGenerateButton(false);
    }

    /**
     * Retries the last failed generation
     */
    async retry() {
        this.uiManager.hideError();
        const parameters = this.uiManager.getParameters();
        await this.generateImage(parameters);
    }
}

export default ImageGenerator;
