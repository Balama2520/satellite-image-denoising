# Satellite Image Denoising Using Machine Learning

Published research on ML-based satellite image denoising using Non-Local Means (NLM) filtering. Compares 5 denoising algorithms across PSNR, SSIM, and MSE metrics.

## 🚀 Live Demo
Try it instantly — no setup needed:
👉 [huggingface.co/spaces/Balamaneesh2520/satellite](https://huggingface.co/spaces/Balamaneesh2520/satellite)

## 📄 Publication
ResearchGate — *"Satellite Image Denoising Using Machine Learning"*
👉 [View Paper](https://www.researchgate.net/profile/Bala-Ayanala)

## 📊 Results — Algorithm Comparison

| Algorithm | PSNR ↑ | MSE ↓ | SSIM ↑ |
|---|---|---|---|
| Gaussian Filtering | 24.30 | 118.42 | 0.95 |
| Median Filtering | 26.71 | 159.42 | 0.93 |
| Wavelet Transform | 27.39 | 96.96 | 0.96 |
| **NLM Filtering (Ours)** | **29.95** | **73.46** | **0.99** |
| Bilateral Filtering | 24.36 | 254.56 | 0.97 |

**NLM achieves the best results** — highest PSNR, lowest MSE, highest SSIM.

## 🔍 What This Does
- Uploads a noisy satellite image
- Applies Non-Local Means (NLM) denoising
- Outputs denoised image with real-time PSNR, SSIM, and MSE scores
- Outperforms Gaussian, Median, Wavelet, and Bilateral filtering

## 🛠️ Tech Stack
- Python · OpenCV · NumPy · scikit-image
- Gradio (interactive web UI)
- Google Colab (training environment)
- HuggingFace Spaces (deployment)

## ⚡ Quick Start

```bash
git clone https://github.com/Balama2520/satellite-image-denoising
cd satellite-image-denoising
pip install -r requirements.txt
python app.py
```

## 📁 Key Files
- `app.py` — Gradio web interface + NLM denoising pipeline
- `requirements.txt` — Dependencies

## 👨‍💻 Author
**Bala Maneesh Ayanala** — SRM Institute of Science and Technology

🌐 [Portfolio](https://abms-portfolio.netlify.app) · 
💼 [LinkedIn](https://linkedin.com/in/bala-maneesh-ayanala-702582266) · 
⌥ [GitHub](https://github.com/Balama2520)
