# Latent Diffusion Model Demo

This guide provides instructions for running the Latent Diffusion Model demo using a Flask web application.

## Prerequisites

- Python 3.6 or higher
- Flask
- PyTorch
- Additional Python packages as specified in the `requirements.txt` (if available)

## Setup Instructions

1. **Clone the Repository**

   Clone the repository to your local machine using the following command:

   ```bash
   git clone https://github.com/CompVis/latent-diffusion.git
   git clone https://github.com/CompVis/taming-transformers.git

   ```

    Install taming transformer

    ```bash titile='Install‘
    cd taming-transformers
    pip install -e .
    ```

    Install clip

    ```bash
    pip install git+https://github.com/openai/CLIP.git
    ```

2. **Navigate to the Project Directory**

   Change into the project directory:

   ```bash
   cd experiment/repos/latent-diffusion/experiment/latent_diffusion
   ```

3. **Install Dependencies**

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**

   Start the Flask application:

   ```bash
   CUDA_VISIBLE_DEVICES=1 python app.py
   ```

5. **Access the Web Interface**

   Open a web browser and go to `http://localhost:5000` to access the demo interface.

6. **Generate Images**

   Enter a text prompt and adjust the optional parameters to generate an image. The generated image will be displayed and available for download.

## bugs

1. change

```
--from pytorch_lightning.utilities.distributed import rank_zero_only
++from pytorch_lightning.utilities import rank_zero_only
```

## Notes

- Ensure that the model checkpoint is downloaded automatically when the application is run.
- You can customize the image generation parameters such as `DDIM Eta`, `Height`, `Width`, and `Scale` using the web interface.

For further assistance or questions, please refer to the project's documentation or contact the project maintainers.
