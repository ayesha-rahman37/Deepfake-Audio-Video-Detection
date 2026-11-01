"""Quick robustness and inference benchmark for multimodal LAV-DF."""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from datasets.lavdf_dataset import LAVDFDataset
from models.multimodal_model import MultimodalDeepfakeDetector

def transform(v,a,name):
    if name=="baseline": return v,a
    if name=="low_resolution":
        b,t,c,h,w=v.shape; x=v.reshape(b*t,c,h,w)
        x=F.interpolate(x,size=(112,112),mode="bilinear",align_corners=False)
        return F.interpolate(x,size=(h,w),mode="bilinear",align_corners=False).reshape(b,t,c,h,w),a
    if name=="blur":
        b,t,c,h,w=v.shape
        return F.avg_pool2d(v.reshape(b*t,c,h,w),5,1,2).reshape(b,t,c,h,w),a
    if name=="frame_drop":
        x=v.clone()
        if x.shape[1]>2: x[:,1::4]=x[:,0:1].expand(-1,x[:,1::4].shape[1],-1,-1,-1)
        return x,a
    if name=="audio_noise": return v,a+torch.randn_like(a)*a.std().clamp_min(1e-6)*0.1
    if name=="audio_quantization":
        lo,hi=a.amin(),a.amax(); z=((a-lo)/(hi-lo).clamp_min(1e-6)).clamp(0,1)
        return v,torch.round(z*31)/31*(hi-lo)+lo
    if name=="av_shift": return torch.roll(v,2,dims=1),torch.roll(a,max(1,a.shape[-1]//10),dims=-1)
    if name=="combined":
        v,a=transform(v,a,"low_resolution"); v,a=transform(v,a,"blur")
        v,a=transform(v,a,"audio_noise"); return transform(v,a,"av_shift")
    raise ValueError(name)

def pack(y,p,t):
    pred=(p>=t).astype(np.int8); both=len(np.unique(y))==2
    return {"accuracy":float(accuracy_score(y,pred)),
      "balanced_accuracy":float(balanced_accuracy_score(y,pred)) if both else None,
      "precision_fake":float(precision_score(y,pred,zero_division=0)),
      "recall_fake":float(recall_score(y,pred,zero_division=0)),
      "f1_fake":float(f1_score(y,pred,zero_division=0)),
      "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
      "roc_auc":float(roc_auc_score(y,p)) if both else None,
      "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",default="checkpoints/multimodal_reliability_best.pt")
    ap.add_argument("--fusion-mode",choices=["concat","reliability"],default="reliability")
    ap.add_argument("--split",choices=["dev","test"],default="test")
    ap.add_argument("--samples",type=int,default=256); ap.add_argument("--batch-size",type=int,default=1)
    ap.add_argument("--threshold",type=float,default=0.5); ap.add_argument("--seed",type=int,default=123)
    a=ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp=Path(a.checkpoint); cp=cp if cp.is_absolute() else ROOT/cp
    ck=torch.load(cp,map_location=dev,weights_only=False); mode=ck.get("fusion_mode",a.fusion_mode)
    if mode!=a.fusion_mode: raise ValueError(f"Checkpoint mode {mode} != requested {a.fusion_mode}")
    model=MultimodalDeepfakeDetector(pretrained_visual=False,freeze_visual=True,frame_batch_size=4,fusion_mode=mode).to(dev)
    model.load_state_dict(ck["model_state_dict"],strict=True); model.eval()
    ds=LAVDFDataset(split=a.split); n=min(a.samples,len(ds))
    ids=np.sort(np.random.default_rng(a.seed).choice(len(ds),n,replace=False)).tolist()
    dl=DataLoader(Subset(ds,ids),batch_size=a.batch_size,shuffle=False,num_workers=0,pin_memory=dev.type=="cuda")
    names=["baseline","low_resolution","blur","frame_drop","audio_noise","audio_quantization","av_shift","combined"]
    out={k:{"y":[],"p":[],"seconds":0.0} for k in names}
    with torch.inference_mode():
      for batch in dl:
        v=batch["video"].to(dev,non_blocking=True); au=batch["audio"].to(dev,non_blocking=True)
        y=batch["label"].cpu().numpy().astype(np.int8).reshape(-1)
        for name in names:
          vv,aa=transform(v,au,name)
          if dev.type=="cuda": torch.cuda.synchronize()
          st=time.perf_counter(); logits=model(vv,aa)["logits"].reshape(-1)
          if dev.type=="cuda": torch.cuda.synchronize()
          out[name]["seconds"]+=time.perf_counter()-st
          out[name]["y"].extend(y.tolist()); out[name]["p"].extend(torch.sigmoid(logits).cpu().numpy().reshape(-1).tolist())
    results={}
    for name,d in out.items():
      yy=np.asarray(d["y"],dtype=np.int8); pp=np.asarray(d["p"],dtype=np.float32); sec=d["seconds"]
      results[name]={**pack(yy,pp,a.threshold),"samples":len(yy),"total_inference_seconds":sec,
                     "mean_inference_ms_per_sample":sec*1000/max(1,len(yy))}
    report={"split":a.split,"checkpoint":str(cp.resolve()),"checkpoint_size_mb":cp.stat().st_size/1048576,
      "device":str(dev),"fusion_mode":mode,"threshold":a.threshold,"sample_count":n,"seed":a.seed,
      "note":"Tensor-level degradation simulations, not real codec re-encoding.","results":results}
    dest=ROOT/"results"/f"robustness_benchmark_{mode}_{a.split}.json"; dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2)); print(f"Saved: {dest}")
if __name__=="__main__": main()
