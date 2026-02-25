import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
#%% Paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
default_model = os.path.join(project_root, "vnct", "best_model")
default_label_map = os.path.join(project_root, "vnct", "label_map.json")
max_len = 256


#%% Helpers
def load_label_map(label_map_path):
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map: dict[str, int] = json.load(f)
    return {v: k for k, v in label_map.items()}


def load_model(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict(text, tokenizer, model, id2label, device, top_k=1):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]

    top_k = min(top_k, len(id2label))
    top_probs, top_ids = torch.topk(probs, k=top_k)

    results = []
    for rank, (prob, label_id) in enumerate(zip(top_probs.tolist(), top_ids.tolist()), start=1):
        results.append(
            {
                "rank": rank,
                "label": id2label[label_id],
                "label_id": label_id,
                "confidence": prob,
            }
        )
    return results

#%% Main
text = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else input("Enter news text: ").strip()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
id2label = load_label_map(default_label_map)
tokenizer, model = load_model(default_model, device)
results = predict(text, tokenizer, model, id2label, device, top_k=3)

print(f"\nInput : {text}")
print("=" * 100)
for r in results:
    bar = "█" * int(r["confidence"] * 30) + "░" * (30 - int(r["confidence"] * 30))
    print(f"  [{r['rank']}] {r['label']:<25} {r['confidence']*100:6.2f}%  |{bar}|")
print("=" * 100)
print(f"→ {results[0]['label']}  ({results[0]['confidence']*100:.2f}%)")
