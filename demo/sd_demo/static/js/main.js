import UIManager from './core/UIManager.js';
import ImageGenerator from './core/ImageGenerator.js';
import EventBus from './core/EventBus.js';

class App {
    constructor() {
        this.eventBus = new EventBus();
        this.uiManager = new UIManager();
        // 使用空字符串作为基础路径，因为后端路由直接从根路径开始
        this.imageGenerator = new ImageGenerator('', this.uiManager);

        this.setupEventListeners();
        this.setupModalHandling();
    }

    setupEventListeners() {
        // Generate button click
        document.getElementById('generate-btn').addEventListener('click', async (e) => {
            e.preventDefault();
            try {
                const parameters = this.uiManager.getParameters();
                await this.imageGenerator.generateImage(parameters);
            } catch (error) {
                console.error('Generation failed:', error);
            }
        });

        // Retry button click
        document.querySelector('[onclick="window.imageGenerator.retryGeneration()"]')
            ?.addEventListener('click', () => this.imageGenerator.retry());

        // Image click for fullscreen
        document.getElementById('result-image')?.addEventListener('click', () => {
            this.showImageModal();
        });

        // Set as background click
        document.getElementById('set-background')?.addEventListener('click', () => {
            const imageUrl = document.getElementById('result-image').src;
            document.body.style.backgroundImage = `url(${imageUrl})`;
        });
    }

    setupModalHandling() {
        const modal = document.getElementById('imageModal');
        if (!modal) return;

        const modalImage = document.getElementById('modal-image');
        const modalLoading = document.getElementById('modal-loading');
        let isZoomed = false;

        // Toggle zoom on modal image click
        modalImage?.addEventListener('click', () => {
            isZoomed = !isZoomed;
            modalImage.style.transform = isZoomed ? 'scale(1.5)' : 'scale(1)';
        });

        // Reset zoom when modal is closed
        modal.addEventListener('hidden.bs.modal', () => {
            isZoomed = false;
            modalImage.style.transform = 'scale(1)';
        });

        // Handle zoom button
        document.getElementById('modal-zoom-toggle')?.addEventListener('click', () => {
            isZoomed = !isZoomed;
            modalImage.style.transform = isZoomed ? 'scale(1.5)' : 'scale(1)';
        });
    }

    showImageModal() {
        const modal = document.getElementById('imageModal');
        const modalImage = document.getElementById('modal-image');
        const modalLoading = document.getElementById('modal-loading');
        const resultImage = document.getElementById('result-image');

        if (modal && modalImage && resultImage) {
            modalLoading.style.display = 'block';
            modalImage.src = resultImage.src;
            modalImage.onload = () => {
                modalLoading.style.display = 'none';
            };
            new bootstrap.Modal(modal).show();
        }
    }
}

// Initialize the app when the DOM is loaded
window.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
