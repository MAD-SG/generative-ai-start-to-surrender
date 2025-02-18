document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('generation-form');
    const loadingIndicator = document.getElementById('loading');
    const resultContainer = document.getElementById('result');
    const errorContainer = document.getElementById('error');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        resultContainer.innerHTML = '';
        errorContainer.style.display = 'none';

        try {
            const formData = new FormData(form);
            
            // Send request to generate image
            const response = await fetch('/generate', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to generate image');
            }

            // Create image element with the response
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            const img = document.createElement('img');
            img.src = imageUrl;
            resultContainer.appendChild(img);

        } catch (error) {
            // Show error message
            errorContainer.textContent = error.message;
            errorContainer.style.display = 'block';
        } finally {
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
        }
    });

    // Update dimensions based on model selection
    const modelSelect = document.getElementById('model_name');
    modelSelect.addEventListener('change', function() {
        const model = this.value;
        let defaultHeight = 512;
        let defaultWidth = 512;

        // Set different default dimensions for different models
        if (model === 'sd-v2.0') {
            defaultHeight = 768;
            defaultWidth = 768;
        }

        document.getElementById('height').value = defaultHeight;
        document.getElementById('width').value = defaultWidth;
    });
});
