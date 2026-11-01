"""Same-subset multimodal comparison and feature-level ablation.
Run: python evaluation/model_comparison_ablation.py --split test --samples 256 --batch-size 1
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from datasets.lavdf_dataset import LAVDFDataset
from models.multimodal_model import MultimodalDeepfakeDetector

def choose_indices(ds, limit, seed):
    if limit <= 0 or limit >= len(ds): return list(range(len(ds)))
    groups = {0: [], 1: []}
    for i, row in enumerate(ds.samples): groups[int(row["label"])].append(i)
    rng = random.Random(seed)
    for v in groups.values(): rng.shuffle(v)
    a, b = min(len(groups[0]), limit//2), min(len(groups[1]), limit-limit//2)
    out = groups[0][:a] + groups[1][:b]
    if len(out) < limit:
        used=set(out); rest=[i for i in range(len(ds)) if i not in used]; rng.shuffle(rest)
        out += rest[:limit-len(out)]
    rng.shuffle(out)
    return out

def metrics(y, p):
    y=np.asarray(y,dtype=np.int64); p=np.asarray(p,dtype=np.float64); pred=(p>=0.5).astype(np.int64)
    both=len(np.unique(y))==2
    return {"samples":len(y),"threshold":0.5,"accuracy":float(accuracy_score(y,pred)),
      "balanced_accuracy":float(balanced_accuracy_score(y,pred)) if both else None,
      "precision_fake":float(precision_score(y,pred,zero_division=0)),
      "recall_fake":float(recall_score(y,pred,zero_division=0)),
      "f1_fake":float(f1_score(y,pred,zero_division=0)),
      "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
      "roc_auc":float(roc_auc_score(y,p)) if both else None,
      "confusion_matrix_labels_0_1":confusion_matrix(y,pred,labels=[0,1]).tolist()}

def find_ckpt(mode, supplied):
    if supplied: return Path(supplied)
    for p in [ROOT/"checkpoints"/f"multimodal_{mode}_best.pt", ROOT/"checkpoints"/f"multimodal_{mode}_candidate.pt"]:
        if p.is_file(): return p
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--split",choices=["dev","test"],default="test")
    ap.add_argument("--samples",type=int,default=256,help="0 means full split")
    ap.add_argument("--batch-size",type=int,default=1)
    ap.add_argument("--seed",type=int,default=123)
    ap.add_argument("--concat-checkpoint",default="")
    ap.add_argument("--reliability-checkpoint",default="")
    ap.add_argument("--output",default="")
    a=ap.parse_args()
    if a.samples<0 or a.batch_size<1: ap.error("samples must be >=0 and batch-size >=1")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    if device.type=="cuda": print("GPU:",torch.cuda.get_device_name(0))
    ds=LAVDFDataset(split=a.split); ids=choose_indices(ds,a.samples,a.seed)
    print("Selected windows:",len(ids))
    print("Class counts:",{"real":sum(int(ds.samples[i]["label"])==0 for i in ids),"fake":sum(int(ds.samples[i]["label"])==1 for i in ids)})
    loader=DataLoader(Subset(ds,ids),batch_size=a.batch_size,shuffle=False,num_workers=0,pin_memory=device.type=="cuda")
    report={"split":a.split,"seed":a.seed,"selected_dataset_indices":ids,
      "target":"LAV-DF window-level label; identical samples and labels across all evaluated variants",
      "ablation_note":"Feature-level diagnostic: one modality embedding is replaced by zeros before the trained fusion head. This is not a separately trained audio-only/video-only model.",
      "models":{}}
    for mode,supplied in [("concat",a.concat_checkpoint),("reliability",a.reliability_checkpoint)]:
        cp=find_ckpt(mode,supplied)
        if cp is None: print("Skipping",mode,"checkpoint not found"); continue
        if not cp.is_absolute(): cp=ROOT/cp
        ck=torch.load(cp,map_location=device,weights_only=False)
        ck_mode=ck.get("fusion_mode",mode)
        if ck_mode!=mode: raise ValueError(f"{cp} mode is {ck_mode}, expected {mode}")
        model=MultimodalDeepfakeDetector(pretrained_visual=False,freeze_visual=True,frame_batch_size=4,fusion_mode=mode).to(device)
        model.load_state_dict(ck["model_state_dict"],strict=True); model.eval()
        vals={k:{"y":[],"p":[]} for k in ["full","video_ablated","audio_ablated"]}
        t0=time.perf_counter(); done=0
        with torch.inference_mode():
            for batch in loader:
                v=batch["video"].to(device); au=batch["audio"].to(device); y=batch["label"].long().reshape(-1)
                out=model(v,au); ve,ae,q=out["video_embedding"],out["audio_embedding"],out["quality_features"]
                _,lv,_=model.fusion(torch.zeros_like(ve),ae,q)
                _,la,_=model.fusion(ve,torch.zeros_like(ae),q)
                for key,prob in [("full",torch.sigmoid(out["logits"])),("video_ablated",torch.sigmoid(lv)),("audio_ablated",torch.sigmoid(la))]:
                    vals[key]["y"].extend(y.tolist()); vals[key]["p"].extend(prob.cpu().tolist())
                done+=len(y)
                if done%50==0 or done==len(ids): print(f"{mode}: {done}/{len(ids)}",flush=True)
        report["models"][mode]={"checkpoint":str(cp.resolve()),"checkpoint_epoch":ck.get("epoch"),
          "elapsed_seconds":round(time.perf_counter()-t0,2),
          "results":{k:metrics(v["y"],v["p"]) for k,v in vals.items()}}
        del model
        if device.type=="cuda": torch.cuda.empty_cache()
    if not report["models"]: raise RuntimeError("No matching multimodal checkpoints found.")
    out=Path(a.output) if a.output else ROOT/"results"/f"model_comparison_ablation_{a.split}_{len(ids)}.json"
    if not out.is_absolute(): out=ROOT/out
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise FileExistsError(f"Refusing to overwrite {out}; choose --output with a new filename.")
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("\nSUMMARY")
    for mode,data in report["models"].items():
        print("\n"+mode)
        for name,m in data["results"].items(): print(name,json.dumps(m))
    print("\nSaved:",out)

if __name__=="__main__": main()
