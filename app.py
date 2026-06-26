import cv2
import numpy as np
import gradio as gr
from skimage.metrics import structural_similarity as ssim_metric


def calculate_mse(original, denoised):
    return np.mean(
        (original.astype(np.float32) - denoised.astype(np.float32)) ** 2
    )


def calculate_psnr(original, denoised):
    mse = calculate_mse(original, denoised)

    if mse == 0:
        return float("inf")

    return 20 * np.log10(255.0 / np.sqrt(mse))


def denoise_image(image, h_value):
    if image is None:
        return None, "N/A", "N/A", "N/A"

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    denoised = cv2.fastNlMeansDenoisingColored(
        image_bgr,
        None,
        int(h_value),
        int(h_value),
        7,
        21
    )

    gray_original = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    mse = calculate_mse(gray_original, gray_denoised)
    psnr = calculate_psnr(gray_original, gray_denoised)
    ssim, _ = ssim_metric(gray_original, gray_denoised, full=True)

    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

    return (
        denoised_rgb,
        f"{mse:.2f}",
        f"{psnr:.2f} dB",
        f"{ssim:.4f}"
    )


demo = gr.Interface(
    fn=denoise_image,
    inputs=[
        gr.Image(type="numpy"),
        gr.Slider(
            minimum=3,
            maximum=25,
            value=10,
            step=1,
            label="Denoising Strength"
        )
    ],
    outputs=[
        gr.Image(type="numpy", label="Denoised Image"),
        gr.Textbox(label="MSE"),
        gr.Textbox(label="PSNR"),
        gr.Textbox(label="SSIM")
    ],
    title="Satellite Image Denoising",
    description="Upload a satellite image and evaluate denoising quality."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
