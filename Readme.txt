MoRE: MultiModal Contrastive Pretraining of Xray, ECG, and Report

Setting up environment:
1.Create a virtual ENV
2.pip install requirements.txt

Pre-Train MoRE:
1. Download Required Datasets from Physionet (we do not attach due to required credential for data signing)
2. Add data in appropriate folder.
3. Preprocess data (preprocess code included)
4. Run python pretrain_multimodel.py (add args as needed, default in place)

Fine-tune in Mimic/Chexpert:
1. Ensure model is saved.
2. Run multimodal_infer.py (change data paths, and model paths)

Zero-Shot classification:
1. Run zero_shot_xray/ecg_more.py to do zero-shot classification on Mimic/Chexpert
2. Change data as necessary/update path

Retrieval Tasks:
1. Look at xray_ecg_retrieval notebook for example of multimodal retrieval
2. Run xray_retrieval for text/image retrieval of X-ray

t-SNE plot:
1. Look at tnse_plot notebook for example of tnse plot of features