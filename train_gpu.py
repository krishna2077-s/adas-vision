# train_gpu.py -- GPU fine-tuning for RTX 3050 + IDD Lite
import argparse, glob, json, sys, os, random, time
if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8",errors="replace")
import cv2, numpy as np, torch, torch.nn as nn
from PIL import Image, ImageDraw
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
import adverse_aug

IDD_TAR      = r"C:\Users\laksh\Downloads\idd-lite.tar.gz"
DATA_DIR     = r"C:\Users\laksh\Downloads\idd_extracted"
INIT_WEIGHTS = "drivable_idd_full_best.pth"
OUT_WEIGHTS  = "drivable_idd_lraspp_gpu_best.pth"
CKPT         = "train_gpu_ckpt.pth"
IN_W, IN_H   = 768, 432
MEAN = np.array([0.485,0.456,0.406],np.float32)
STD  = np.array([0.229,0.224,0.225],np.float32)

def get_device():
    if torch.cuda.is_available():
        dev=torch.device("cuda"); p=torch.cuda.get_device_properties(dev)
        print(f"[device] CUDA OK -- {p.name} ({p.total_memory/1e9:.1f} GB VRAM)")
        print(f"[device] CUDA version: {torch.version.cuda}")
        torch.backends.cudnn.benchmark=True; return dev
    print("[device] CPU fallback"); return torch.device("cpu")

def build_model():
    m=lraspp_mobilenet_v3_large(weights=None,weights_backbone=None)
    m.classifier.low_classifier=nn.Conv2d(40,2,1)
    m.classifier.high_classifier=nn.Conv2d(128,2,1); return m

def ensure_extracted():
    hit=glob.glob(os.path.join(DATA_DIR,"**","leftImg8bit"),recursive=True)
    if hit: root=os.path.dirname(hit[0]); print(f"[data] already extracted: {root}"); return root
    if not os.path.exists(IDD_TAR): raise SystemExit(f"[data] Not found: {IDD_TAR}")
    print(f"[data] extracting {os.path.basename(IDD_TAR)}...")
    import tarfile, shutil
    print(f"[data] free disk: {shutil.disk_usage(os.path.dirname(DATA_DIR) or '.').free/1e9:.1f} GB")
    os.makedirs(DATA_DIR,exist_ok=True); t0=time.time()
    with tarfile.open(IDD_TAR) as t: t.extractall(DATA_DIR)
    print(f"[data] extracted in {time.time()-t0:.0f}s")
    hit=glob.glob(os.path.join(DATA_DIR,"**","leftImg8bit"),recursive=True)
    if not hit: raise SystemExit("[data] no leftImg8bit/ found after extraction")
    return os.path.dirname(hit[0])

def rasterize_masks(root):
    print("[masks] IDD Lite uses pre-rasterized PNG label maps -- skipping.")

def pairs(root, split):
    out=[]
    for ip in sorted(glob.glob(os.path.join(root,"leftImg8bit",split,"*","*_image.jpg"))):
        seq=os.path.basename(os.path.dirname(ip))
        base=os.path.splitext(os.path.basename(ip))[0].replace("_image","")
        lp=os.path.join(root,"gtFine",split,seq,base+"_label.png")
        if os.path.exists(lp): out.append((ip,lp))
    return out

class IDDAug(Dataset):
    def __init__(self,pairs,train=True,hard=False,rich=False):
        self.pairs,self.train,self.hard,self.rich=pairs,train,hard,rich
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        ip,mp=self.pairs[i]
        img=cv2.cvtColor(cv2.imread(ip),cv2.COLOR_BGR2RGB)
        lbl=cv2.imread(mp,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(IN_W,IN_H)).astype(np.float32)
        lbl=cv2.resize(lbl,(IN_W,IN_H),interpolation=cv2.INTER_NEAREST)
        if self.train:
            if random.random()<0.5: img,lbl=img[:,::-1].copy(),lbl[:,::-1].copy()
            if self.rich:
                img=adverse_aug.random_adverse(img,random)
            else:
                if random.random()<0.25: img=np.clip(img*random.uniform(0.35,0.60),0,255)
                if random.random()<0.20:
                    h=img.shape[0]; g=np.linspace(1.0,0.3,h).reshape(h,1,1)
                    img=np.clip(img*(1-0.35*g)+np.full_like(img,200.0)*(0.35*g),0,255)
                if random.random()<0.30: img=np.clip(img*random.uniform(0.7,1.3),0,255)
        elif self.hard:
            img=adverse_aug.hard_adverse(img,10000+i) if self.rich else np.clip(img*0.45,0,255)
        x=(img/255.0-MEAN)/STD
        x=torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).float()
        b=(lbl==0).astype(np.int64) if lbl.max()<20 else (lbl>127).astype(np.int64)
        return x,torch.from_numpy(np.ascontiguousarray(b))

def dice_loss(logits,target,eps=1.0):
    prob=torch.softmax(logits,1)[:,1]; t=(target==1).float()
    inter=(prob*t).sum((1,2)); union=prob.sum((1,2))+t.sum((1,2))
    return (1-(2*inter+eps)/(union+eps)).mean()

def seg_loss(out,y,ce):
    loss=ce(out["out"],y)+dice_loss(out["out"],y)
    if "aux" in out: loss=loss+0.4*ce(out["aux"],y)
    return loss

@torch.no_grad()
def val_iou(model,dl,device):
    model.eval(); inter=union=0
    for x,y in dl:
        x,y=x.to(device),y.to(device)
        with autocast(device_type=device.type): p=model(x)["out"].argmax(1)
        inter+=((p==1)&(y==1)).sum().item(); union+=((p==1)|(y==1)).sum().item()
    return inter/max(union,1)

def smoke_test(device):
    print("[smoke] building model + loading weights ...")
    model=build_model().to(device)
    if os.path.exists(INIT_WEIGHTS):
        s=torch.load(INIT_WEIGHTS,map_location=device,weights_only=True)
        m,u=model.load_state_dict(s,strict=False)
        print(f"[smoke] loaded -- missing {len(m)}, unexpected {len(u)}")
    model.eval()
    ce=nn.CrossEntropyLoss()
    x=torch.randn(1,3,IN_H,IN_W,device=device); y=torch.randint(0,2,(1,IN_H,IN_W),device=device)
    with autocast(device_type=device.type): _=model(x)
    if device.type=="cuda": torch.cuda.synchronize()
    t0=time.time()
    with autocast(device_type=device.type): out=model(x)
    if device.type=="cuda": torch.cuda.synchronize()
    fwd=time.time()-t0; vram=torch.cuda.memory_allocated()/1e6 if device.type=="cuda" else 0
    print(f"[smoke] out {tuple(out['out'].shape)} | {fwd*1000:.0f}ms | VRAM {vram:.0f}MB")
    print(f"[smoke] PASS -- GPU + model + AMP all working on {device}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--subset",type=int,default=200)
    ap.add_argument("--val-cap",type=int,default=200)
    ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--lr",type=float,default=4e-4)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--adverse",action="store_true")
    args=ap.parse_args(); device=get_device()
    if args.smoke: smoke_test(device); return

    global OUT_WEIGHTS,CKPT
    if args.adverse:
        OUT_WEIGHTS="drivable_idd_lraspp_gpu_adv_best.pth"; CKPT="train_gpu_adv_ckpt.pth"
        print("[adverse] Phase 12 rich augmentation ON")

    if not os.path.exists(INIT_WEIGHTS): raise SystemExit(f"[init] {INIT_WEIGHTS} not found")
    root=ensure_extracted(); rasterize_masks(root)
    tp=pairs(root,"train"); vp=pairs(root,"val")
    print(f"[data] pairs -- train {len(tp)} | val {len(vp)}")
    if len(tp)<10: raise SystemExit("[data] too few pairs -- check dataset structure")
    if args.val_cap and len(vp)>args.val_cap: random.seed(0); vp=random.sample(vp,args.val_cap)

    kw=dict(batch_size=args.batch,num_workers=args.workers,
            pin_memory=(device.type=="cuda"),persistent_workers=(args.workers>0))
    val_dl=DataLoader(IDDAug(vp,train=False),shuffle=False,**kw)
    hard_dl=DataLoader(IDDAug(vp,train=False,hard=True,rich=args.adverse),shuffle=False,**kw)

    model=build_model().to(device)
    opt=torch.optim.AdamW([
        {"params":model.backbone.parameters(),"lr":args.lr*0.1},
        {"params":model.classifier.parameters(),"lr":args.lr}],weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    ce=nn.CrossEntropyLoss()
    scaler=GradScaler(device="cuda" if device.type=="cuda" else "cpu")

    start_ep,best_iou=0,None
    if os.path.exists(CKPT):
        ck=torch.load(CKPT,map_location=device,weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_ep,best_iou=ck["epoch"]+1,ck["best_iou"]
        print(f"[resume] epoch {start_ep+1}, best IoU {best_iou:.3f}")
    else:
        model.load_state_dict(torch.load(INIT_WEIGHTS,map_location=device,weights_only=True))
        print(f"[init] fine-tuning from {INIT_WEIGHTS}")

    if best_iou is None:
        print("[eval] baseline ..."); c=val_iou(model,val_dl,device); best_iou=val_iou(model,hard_dl,device)
        print(f"[eval] clean {c:.3f} | hard {best_iou:.3f}"); torch.save(model.state_dict(),OUT_WEIGHTS)

    print(f"\n{'='*55}\n  Training: {args.epochs}ep batch={args.batch} subset={args.subset} AMP=ON\n{'='*55}\n")
    global_t0=time.time()
    for ep in range(start_ep,args.epochs):
        ep_pairs=(random.sample(tp,args.subset) if args.subset and len(tp)>args.subset else tp)
        train_dl=DataLoader(IDDAug(ep_pairs,train=True,rich=args.adverse),shuffle=True,**kw)
        model.train(); running=seen=0; t0=time.time()
        for bi,(x,y) in enumerate(train_dl):
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type): loss=seg_loss(model(x),y,ce)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update()
            running+=loss.item()*x.size(0); seen+=x.size(0)
            if bi%5==0:
                vram=torch.cuda.memory_allocated()/1e6 if device.type=="cuda" else 0
                eta=(len(ep_pairs)-seen)/max(1e-6,seen/max(1e-6,time.time()-t0))/60
                print(f"  ep{ep+1} {seen}/{len(ep_pairs)} loss={running/max(1,seen):.4f} VRAM={vram:.0f}MB ETA={eta:.1f}m",flush=True)
        sched.step()
        ci=val_iou(model,val_dl,device); hi=val_iou(model,hard_dl,device)
        flag=""
        if hi>best_iou: best_iou=hi; torch.save(model.state_dict(),OUT_WEIGHTS); flag="  <-- NEW BEST"
        torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"sched":sched.state_dict(),
                    "scaler":scaler.state_dict(),"epoch":ep,"best_iou":best_iou},CKPT)
        print(f"[ep {ep+1}/{args.epochs}] loss={running/max(1,seen):.4f} clean={ci:.3f} hard={hi:.3f} best={best_iou:.3f} {(time.time()-t0)/60:.1f}m{flag}\n")
    print(f"DONE in {(time.time()-global_t0)/60:.1f}min | best hard IoU {best_iou:.4f} -> {OUT_WEIGHTS}")

if __name__=="__main__":
    main()
