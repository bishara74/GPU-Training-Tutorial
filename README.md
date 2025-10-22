#  PyTorch GPU vs. CPU Training Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bishara74/GPU-Training-Tutorial/blob/main/GPU_vs_CPU_Benchmark.ipynb)

This project serves as a **Jupyter Notebook tutorial** and **shared resource** for developers looking to understand the real-world impact of **GPU resource handling** for deep learning.

The goal is to benchmark the training of a ResNet-18 model on a CPU versus a GPU. The project uses a custom dataset of 2,200+ sneaker images (Nike Air Force 1, Jordan 1, etc.) to provide a real-world example of an image classification task.

## 📊 Key Results

The benchmark was performed by training the *exact same model* for 3 epochs on the same dataset, once on a CPU and once on an NVIDIA T4 GPU.

| Metric | CPU (Intel Xeon) | GPU (NVIDIA T4) |
| :--- | :--- | :--- |
| **Total Training Time** | **1425.24 seconds** (~23.7 min) | **107.13 seconds** (~1.8 min) |
| **Result** | --- | **13.3x Speedup** |

This benchmark clearly demonstrates the critical importance of using GPU acceleration for modern machine learning workflows.

## Technologies Used

* **Python 3**
* **PyTorch**: The primary deep learning framework.
* **torchvision**: Used for the `ResNet-18` pre-trained model and image transformations.
* **Google Colab**: The **Jupyter Notebook** environment used for development and **GPU access**.
* **Matplotlib / NumPy**: Used for data handling and visualization (implied in the stack).

##  How to Run This Tutorial

1.  **Open in Colab:** Click the "Open in Colab" badge at the top of this `README`.
2.  **Upload Data:** This notebook requires a `.zip` file of image data. The code is set up to extract a file named `sneakers.zip` which should contain sub-folders for each class (e.g., `sneakers_dataset/Nike Air Force 1/`, `sneakers_dataset/Nike Dunk Low/`, etc.).
3.  **Enable the GPU:** In Colab, go to **Runtime -> Change runtime type** and select **"T4 GPU"** from the hardware accelerator dropdown.
4.  **Run All Cells:** You can run the cells sequentially to perform the data loading, CPU training, and GPU training to reproduce the benchmark for yourself.
