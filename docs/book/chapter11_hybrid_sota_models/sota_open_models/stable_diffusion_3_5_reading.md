# Stable Diffusion 3.5

- repo: <https://github.com/Stability-AI/sd3.5>

## Summary of Key Architectural Differences

1. Architecture Enhancements
   1. Skip Layer Guidance (SLG):
         - SD3.5 introduces this novel technique that selectively skips specific transformer layers (typically 7-9)
         - Only active during 1-20% of the sampling process
         - Uses a separate guidance scale (2.5 in SD3.5 Medium) distinct from the main CFG scale
         - Significantly improves image coherence and reduces artifacts
   2. MMDiTX vs MMDiT:
        - SD3.5 uses the enhanced MMDiTX architecture with more flexible configuration options
        - Added support for cross-block attention with x_block_self_attn
        - Improved normalization with RMSNorm support as an alternative to LayerNorm
        - Better parameter management and more modular design
2. Sampling Improvements
   1. Default Samplers:
         - SD3: Uses euler sampler as default
         - SD3.5: Uses dpmpp_2m (DPM-Solver++) sampler for better quality
   2. Noise Scheduling:
         - SD3: Uses shift=1.0
         - SD3.5: Uses shift=3.0 for improved noise distribution
   3. Default Configurations:
         - SD3.5 Medium: 50 steps, CFG 5.0, with Skip Layer Guidance
         - SD3.5 Large: 40 steps, CFG 4.5
         - SD3.5 Large Turbo: 4 steps, CFG 1.0 (optimized for speed)
3. New Capabilities
    1. ControlNet Integration:
            - Native support for various ControlNet types (blur, canny, depth)
            - Dedicated ControlNetEmbedder class for processing control inputs
            - Support for 8-bit and 2-bit ControlNet variations
    2. Attention Mechanisms:
            - More configurable attention with qk_norm options
            - Enhanced cross-attention capabilities
            - Better handling of long-range dependencies

4. Technical Implementation

   1. Code Quality:
      - More modular design in SD3.5
      - Better type hinting and parameter validation
      - Enhanced error handling and debugging capabilities
   2. Performance:
      - More efficient attention mechanisms
      - Better memory management
      - Support for different precision modes

In this article, we will study the differences in architecture, such as skip layer guidance and MM-DiTX. We will also explore how ControlNet is implemented in SD3.5.

As for the elements that are similar to SD 3, including the VAE, prompt processing, and sampling scheme, the differences are not significant. Please refer to the previous article [stable diffusion 3 reading](./stable_diffusion_3_reading.md) for more information.

## Skip Layer Guidance

|![alt text](../../../images/image-95.png) | ![alt text](../../../images/image-96.png)|
|---|---|
| w/o SLG | w/ SLG |

Apparently, the fingers look better. This could be evidence that supports the claimed benefits (improved anatomy). However, other aspects of the image also change.

|![alt text](../../../images/image-98.png) | ![alt text](../../../images/image-97.png)|![alt text](../../../images/image-99.png)|
|---|---|---|
| vanilla diffusers which looks awful| skipping layers 6, 7, 8, 9 with SLG scale of 5.6| skipping 7, 8, 9 with SLG scale of 2.8|

See more comparisons of CFG and SLG in [here](https://sandner.art/sd-35-medium-skip-layer-guidance-and-fix-composition-hands-and-anatomy/)

    ```py3
    # From SkipLayerCFGDenoiser in SD3.5
    def forward(self, x, timestep, cond, uncond, cond_scale, **kwargs):
        # Run cond and uncond in a batch together
        batched = self.model.apply_model(
            torch.cat([x, x]),
            torch.cat([timestep, timestep]),
            c_crossattn=torch.cat([cond["c_crossattn"], uncond["c_crossattn"]]),
            y=torch.cat([cond["y"], uncond["y"]]),
            **kwargs,
        )
        # Then split and apply CFG Scaling
        pos_out, neg_out = batched.chunk(2)
        scaled = neg_out + (pos_out - neg_out) * cond_scale

        # Then run with skip layer
        if (self.slg > 0 and self.step > (self.skip_start * self.steps)
            and self.step < (self.skip_end * self.steps)):
            skip_layer_out = self.model.apply_model(
                x, timestep, c_crossattn=cond["c_crossattn"],
                y=cond["y"], skip_layers=self.skip_layers,
            )
            # Then scale acc to skip layer guidance
            scaled = scaled + (pos_out - skip_layer_out) * self.slg

        self.step += 1
        return scaled
    ```

Compared to CFG, SLG incorporates an additional direction correction term, which helps improve anatomical accuracy in generated images.

According to the configuration:

    ```json
    "sd3.5_medium": {
        "shift": 3.0,
        "steps": 50,
        "cfg": 5.0,
        "sampler": "dpmpp_2m",
        "skip_layer_config": {
            "scale": 2.5,
            "start": 0.01,  # skip_start value
            "end": 0.20,    # skip_end value
            "layers": [7, 8, 9],
            "cfg": 4.0,
        },
    }
    ```
The skip layer guidance is only active during the initial 1-20% of the sampling process, targeting layers [7, 8, 9], and scaling the CFG to 4.0.

In the MM-DiTX implementation, the skip layers are treated as identity functions:

    ```py3
    for i, block in enumerate(self.joint_blocks):
        if i in skip_layers:
            continue
        context, x = block(context, x, c=c_mod)
    ```

Both `pos_out` and `skip_layer_out` use the same positive condition but differ in their treatment of skip layers. If we consider the skipped layers as a negative condition, this effectively pushes the sample away from that negative influence. What does this negative influence represent when removing layers 7, 8, and 9 (or any specific layers)? If we assume that specific layers are responsible for different features in the image—for example, if layers 7, 8, and 9 handle finer details—then the negative condition would produce images with poor fine structure. Therefore, moving away from this negative influence results in images with enhanced fine details and better structural integrity.

## MM-DiTX

## ControlNet in SD 3.5
