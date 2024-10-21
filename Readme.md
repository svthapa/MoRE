
# MoRE: MultiModal Contrastive Pretraining of X-ray, ECG, and Report

![MoRE Framework](./diagramMultimodal_final.png)

## Setting up the Environment

1. **Create a virtual environment**:
   ```bash
   python -m venv myenv
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Pre-Train MoRE

1. **Download the required datasets** from Physionet (datasets are not attached due to credential requirements for data signing).
   
2. **Add the datasets** to the appropriate folder.

3. **Preprocess the data** (preprocessing code is included).

4. **Run the pretraining script**:
   ```bash
   python pretrain_multimodel.py
   ```
   Add arguments as needed; default settings are provided.

## Fine-tune in Mimic/Chexpert

1. **Ensure that the pre-trained model is saved**.

2. **Run the fine-tuning script**:
   ```bash
   python multimodal_infer.py
   ```
   Make sure to change the data paths and model paths as needed.

## Zero-Shot Classification

1. **Run the zero-shot classification script**:
   ```bash
   python zero_shot_xray/ecg_more.py
   ```
   Update data paths or parameters as necessary.

## Retrieval Tasks

1. **Check the `xray_ecg_retrieval.ipynb` notebook** for an example of multimodal retrieval.

2. **Run the X-ray retrieval script**:
   ```bash
   python xray_retrieval.py
   ```

## t-SNE Plot

1. **Check the `tnse_plot.ipynb` notebook** for an example of a t-SNE plot of features.

