import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.lavdf_dataset import LAVDFDataset
from models.visual_encoder import VisualEncoder
from models.temporal_model import TemporalModel

class VideoOnlyDataset(LAVDFDataset):
    def __getitem__(self, index):
        sample = self.samples[index]
        video = self._load_video(sample["video_path"], sample["start"], sample["end"])
        return video, torch.tensor(int(sample["video_label"]), dtype=torch.float32)

class VideoOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = VisualEncoder(pretrained=False, freeze_backbone=True)
        self.temporal = TemporalModel(input_dim=2048, hidden_dim=256)
        self.classifier = nn.Sequential(nn.Linear(512,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,1))
    def forward(self, video):
        self.visual.eval()
        b,t,c,h,w = video.shape
        flat = video.reshape(b*t,c,h,w)
        with torch.no_grad():
            feats = [self.visual(flat[i:i+4]) for i in range(0, flat.shape[0], 4)]
        feats = torch.cat(feats).reshape(b,t,2048)
        return self.classifier(self.temporal(feats)).squeeze(1)

def choose(dataset, limit, seed):
    labels = [int(s["video_label"]) for s in dataset.samples]
    groups = {k:[i for i,y in enumerate(labels) if y==k] for k in (0,1)}
    if not groups[0] or not groups[1]:
        raise RuntimeError(f"Both video classes required: real={len(groups[0])}, video_fake={len(groups[1])}")
    if limit <= 0 or limit >= len(dataset):
        return list(range(len(dataset)))
    rng=random.Random(seed)
    for v in groups.values(): rng.shuffle(v)
    n=limit//2
    chosen=groups[0][:n]+groups[1][:n]
    if len(chosen)<limit:
        used=set(chosen); rest=[i for i in range(len(dataset)) if i not in used]
        rng.shuffle(rest); chosen += rest[:limit-len(chosen)]
    rng.shuffle(chosen)
    return chosen

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--split",choices=["dev","test"],default="test")
    p.add_argument("--samples",type=int,default=256,help="0 = full split")
    p.add_argument("--batch-size",type=int,default=1)
    a=p.parse_args()
    if a.batch_size<1: p.error("--batch-size must be >=1")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    if device.type=="cuda": print("GPU:",torch.cuda.get_device_name(0))
    ds=VideoOnlyDataset(a.split)
    ids=choose(ds,a.samples,44 if a.split=="test" else 43)
    ys=[int(ds.samples[i]["video_label"]) for i in ids]
    print(f"Split: {a.split}; selected: {len(ids)}")
    print({"real":ys.count(0),"video_fake":ys.count(1)})
    loader=DataLoader(Subset(ds,ids),batch_size=a.batch_size,shuffle=False,num_workers=0,pin_memory=device.type=="cuda")
    ckpt=ROOT/"checkpoints"/"video_only_v2_best.pt"
    if not ckpt.exists(): raise FileNotFoundError(ckpt)
    model=VideoOnlyModel().to(device)
    model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True),strict=True)
    model.eval(); criterion=nn.BCEWithLogitsLoss()
    labels_all=[]; probs_all=[]; loss_sum=0.0
    with torch.inference_mode():
        for j,(videos,labels) in enumerate(loader,1):
            videos=videos.to(device,non_blocking=True); labels=labels.to(device,non_blocking=True)
            logits=model(videos); loss=criterion(logits,labels); probs=torch.sigmoid(logits)
            loss_sum += loss.item()*labels.size(0)
            labels_all.extend(labels.int().cpu().tolist()); probs_all.extend(probs.cpu().tolist())
            if j%50==0 or j==len(loader): print(f"Evaluated {j}/{len(loader)}")
    preds=[int(x>=0.5) for x in probs_all]; both=len(set(labels_all))==2
    metrics={
      "model":"Video-only ResNet50 + BiLSTM","checkpoint":str(ckpt),"split":a.split,
      "samples":len(labels_all),"threshold":0.5,"loss":loss_sum/max(1,len(labels_all)),
      "accuracy":accuracy_score(labels_all,preds),"balanced_accuracy":balanced_accuracy_score(labels_all,preds),
      "precision_fake":precision_score(labels_all,preds,zero_division=0),"recall_fake":recall_score(labels_all,preds,zero_division=0),
      "f1_fake":f1_score(labels_all,preds,zero_division=0),"macro_f1":f1_score(labels_all,preds,average="macro",zero_division=0),
      "roc_auc":roc_auc_score(labels_all,probs_all) if both else None,
      "confusion_matrix_labels_0_1":confusion_matrix(labels_all,preds,labels=[0,1]).tolist(),
      "warning":"Subset evaluation." if len(ids)<len(ds) else "Full split evaluation."
    }
    suffix="full" if len(ids)==len(ds) else str(len(ids))
    out=ROOT/"results"/f"video_only_v2_{a.split}_{suffix}_metrics.json"
    if out.exists(): raise FileExistsError(f"Will not overwrite: {out}")
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print("\nVIDEO-ONLY CHECKPOINT TEST:\n",json.dumps(metrics,indent=2),"\nSaved:",out)

if __name__=="__main__": main()
