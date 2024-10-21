import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from skimage import exposure

import sys
sys.path.append('./utils')
from build_model import ViTModelXray, ProjectionHead, MultiModal
from metrics import calculate_multilabel_map, plot_precision_recall_curve, plot_roc_curve, calc_roc_auc
from sklearn.metrics.pairwise import cosine_similarity


CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    # "Consolidation": {
    #     "severity": ["", "increased", "improved", "apperance of"],
    #     "subtype": [
    #         "bilateral consolidation",
    #         "reticular consolidation",
    #         "retrocardiac consolidation",
    #         "patchy consolidation",
    #         "airspace consolidation",
    #         "partial consolidation",
    #     ],
    #     "location": [
    #         "at the lower lung zone",
    #         "at the upper lung zone",
    #         "at the left lower lobe",
    #         "at the right lower lobe",
    #         "at the left upper lobe",
    #         "at the right uppper lobe",
    #         "at the right lung base",
    #         "at the left lung base",
    #     ],
    # },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}

label_index_mapping = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Consolidation": 2,
    "Edema": 3,
    "Enlarged Cardiomediastinum": 4,
    "Fracture": 5,
    "Lung Lesion": 6,
    "Lung Opacity": 7,
    "No Finding": 8,
    "Pleural Effusion": 9,
    "Pleural Other": 10,
    "Pneumonia": 11,
    "Pneumothorax": 12,
    "Support Devices": 13
}

class PreprocessNote:
    def __init__(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        
    def encode_note(self,x):
        return self.tokenizer(x, return_tensors="pt", padding="max_length",  max_length=512, \
                                      truncation=True, return_attention_mask=True)
        
    def encode(self, x):
        return self.encode_note(x)
    
def generate_chexpert_class_prompts(n: int = 5):
    """Generate text prompts for each CheXpert classification task

    Parameters
    ----------
    n:  int
        number of prompts per class

    Returns
    -------
    class prompts : dict
        dictionary of class to prompts
    """

    prompts = {}
    for k, v in CHEXPERT_CLASS_PROMPTS.items():
        cls_prompts = []
        keys = list(v.keys())

        # severity
        for k0 in v[keys[0]]:
            # subtype
            for k1 in v[keys[1]]:
                # location
                for k2 in v[keys[2]]:
                    cls_prompts.append(f"{k0} {k1} {k2}")

        prompts[k] = random.sample(cls_prompts, n)
    return prompts

def preprocess_xray(x):
    x = Image.open(x)
    x = x.convert('L')
    x = x.resize((224,224))
    x = np.array(exposure.equalize_adapthist(x/np.max(x)))
    x = x *255 
    x = x.astype(np.uint8)
    x = Image.fromarray(x)
    x = transforms.ToTensor()(x)
    x = transforms.Normalize([0.499], [0.293])(x)

    return torch.cat([x,x,x], dim = 0)
         
    
def get_embeddings_similarity(model, imgs, texts, device):
    model.eval()

    imgs = imgs.cuda()

    text_ids = torch.stack([item['input_ids'].squeeze(0) for item in texts])
    masks = torch.stack([item['attention_mask'].squeeze(0) for item in texts])

    text_ids = text_ids.to(device)
    masks = masks.to(device)
    
    with torch.no_grad():
        img_feature = model.projetor_xray_text(model.xray_model(imgs))
        text_feature = model.text_model(text_ids, attention_mask = masks).last_hidden_state.mean(dim=1)
        text_feature = model.projector_text(text_feature)
        

    image_embeddings = F.normalize(img_feature, dim = -1).cpu().numpy()
    text_embeddings = F.normalize(text_feature, dim = -1).cpu().numpy()
    # dot_similarity = image_embeddings @ text_embeddings.T
    cos_sim = cosine_similarity(image_embeddings, text_embeddings)
    return torch.tensor(cos_sim, device='cuda')


def normalize(similarities, method="norm"):
    if method == "norm":
        return (similarities - similarities.mean(axis=0)) / (similarities.std(axis=0))
    elif method == "standardize":
        return (similarities - similarities.min(axis=0)) / (
            similarities.max(axis=0) - similarities.min(axis=0)
        )
    else:
        raise Exception("normalizing method not implemented")


def evaluate_performance(predictions, ground_truth):
    auc_scores = {}
    auprc_scores = {}
    for label in ground_truth.columns:
        auc = roc_auc_score(ground_truth[label], predictions[label])
        auprc = average_precision_score(ground_truth[label], predictions[label])
        auc_scores[label] = auc
        auprc_scores[label] = auprc
    return auc_scores, auprc_scores


def process_prompts(prompts):
    processed_prompts = {}
    process_note = PreprocessNote()
    for key, notes in prompts.items():
        for note in notes:
            if key in processed_prompts.keys():
                processed_prompts[key].append(process_note.encode(note))
            else:
                processed_prompts[key] = [process_note.encode(note)]
    return processed_prompts


def get_similarity(processed_prompts, device):
    class_similarities = []
    for key, note in processed_prompts.items():
        similarities = get_embeddings_similarity(model, img_tensor, note, device)
        cls_similarity, _ = torch.max(similarities, dim =1)  # average between class prompts
        class_similarities.append(cls_similarity.cpu())

    class_similarities = np.stack(class_similarities, axis=1)

    class_similarities = normalize(class_similarities, method='norm')
    class_similarities = pd.DataFrame(
            class_similarities, columns=processed_prompts.keys()
        )
    return class_similarities

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    chexpert_data = load_data('/path/to/chexpert_data.csv')
    model = MultiModal()
    model.load_state_dict(model_dict)
    prompts = generate_chexpert_class_prompts()
    processed_prompts = process_prompt(prompts)
    class_similarities = get_similarity(processed_prompts, device)
    ground_truth = chexpert[chexpert.columns[[14, 8, 11, 16]]]
    auc_scores, auprc_scores = evaluate_performance(class_similarities, ground_truth)
    
    for label, auc in auc_scores.items():
        print(f"{label}: AUC = {auc:.4f}")
    
    for label, auprc in auprc_scores.items():
        print(f"{label}: AUPRC = {auprc:.4f}")

    macro_auc = sum(auc_scores.values())/len(auc_scores)
    print(f"Macro-average AUC: {macro_auc:.4f}")

    average_auprc = sum(auprc_scores.values()) / len(auprc_scores)
    print(f"Average AUPRC: {average_auprc:.4f}")
    

if __name__ == '__main__':
    main()