prompt="A magical garden with giant mushrooms, fairy lights, flowing streams, rainbow flowers, butterflies, mystical atmosphere, ultra detailed, soft lighting, photorealistic textures"
# CUDA_VISIBLE_DEVICES=3 python3 app.py\
#     --model-id flux1-schnell\
#     --prompt "${prompt}" --output-path output.png

# CUDA_VISIBLE_DEVICES=3 python3 app.py\
#     --model-id sd-v3.5-quant\
#     --prompt "${prompt}" --output-path output.png

# CUDA_VISIBLE_DEVICES=3 python3 app.py\
#     --model-id animagine-xl-4\
#     --prompt "${prompt}" --output-path output.png


# CUDA_VISIBLE_DEVICES=3 python3 app.py\
#     --model-id lumina-2-image\
#     --prompt "${prompt}" --output-path output.png

# CUDA_VISIBLE_DEVICES=3 python3 app.py\
#     --model-id  sd-v3.0\
#     --prompt "${prompt}" --output-path output.png

CUDA_VISIBLE_DEVICES=3 python3 app.py --model-id sd-v2.0 \
    --prompt "${prompt}" --output-path output.png

# CUDA_VISIBLE_DEVICES=3 python3 app.py --model-id sd-xl-1.0 \
#     --prompt "${prompt}" --output-path output.png

# CUDA_VISIBLE_DEVICES=3 python3 app.py --model-id sd-v1.5 \
#     --prompt "${prompt}" --output-path output.png

# CUDA_VISIBLE_DEVICES=3 python3 app.py --model-id sd-v1.4 \
#     --prompt "${prompt}" --output-path output.png

# CUDA_VISIBLE_DEVICES=3 python3 app.py --model-id sd-v1.2 \
#     --prompt "${prompt}" --output-path output.png

# CUDA_VISIBLE_DEVICES=3 python3 app.py --model-id sd-v1.1 \
#     --prompt "${prompt}" --output-path output.png