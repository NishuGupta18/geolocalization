#!/usr/bin/env python3
"""Create representative Grad-CAM explanations for pretrained ConvNeXt-Tiny outputs."""
from pathlib import Path
import csv
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WEIGHT_PATH = Path('/tmp/thermal_torch_cache/hub/checkpoints/convnext_tiny-983f1562.pth')
if not WEIGHT_PATH.exists():
    raise FileNotFoundError(f'Pretrained checkpoint not found: {WEIGHT_PATH}')
torch.set_num_threads(min(8, torch.get_num_threads()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = ConvNeXt_Tiny_Weights.DEFAULT
model = convnext_tiny(weights=None)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location='cpu', weights_only=True))
model = model.to(device).eval()
categories = weights.meta['categories']

# Choose four source frames spanning the collection's observed per-frame standard deviation.
stat_rows = list(csv.DictReader(open(ROOT / 'thermal_preprocessing_comparison' / 'original_statistics.csv', newline='')))
ranked = sorted(stat_rows, key=lambda r: float(r['std']))
selected = [ranked[round(p/100*(len(ranked)-1))] for p in (10,35,65,90)]
mean = torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
std = torch.tensor([0.229,0.224,0.225],device=device).view(1,3,1,1)

results=[]
for row in selected:
    name=row['filename']
    raw=np.asarray(Image.open(ROOT/name))
    lo,hi=np.percentile(raw,[1,99])
    gray=np.clip((raw.astype(np.float32)-lo)*255.0/(hi-lo),0,255).round().astype(np.uint8)
    rgb=np.repeat(gray[:,:,None],3,axis=2)
    x=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float().to(device)/255.0
    x=(x-mean)/std
    activations={}
    def save_activation(_module,_inputs,output):
        activations['value']=output
        output.retain_grad()
    hook=model.features[-1].register_forward_hook(save_activation)
    model.zero_grad(set_to_none=True)
    logits=model(x)
    probs=logits.softmax(dim=1)
    confidence,index=probs[0].max(dim=0)
    target_score=logits[0,index]
    target_score.backward()
    hook.remove()
    activation=activations['value'].detach()
    gradient=activations['value'].grad.detach()
    weights_spatial=gradient.mean(dim=(2,3),keepdim=True)
    cam=torch.relu((weights_spatial*activation).sum(dim=1,keepdim=True))[0,0].cpu().numpy()
    if float(cam.max()) > float(cam.min()):
        cam=(cam-cam.min())/(cam.max()-cam.min())
    else:
        cam=np.zeros_like(cam)
    cam=cv2.resize(cam,(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_CUBIC)
    cam=np.clip(cam,0,1)
    heat=cv2.applyColorMap(np.uint8(cam*255),cv2.COLORMAP_JET)
    heat=cv2.cvtColor(heat,cv2.COLOR_BGR2RGB)
    overlay=np.clip(0.55*rgb+0.45*heat,0,255).astype(np.uint8)
    label=categories[int(index)]
    confidence_value=float(confidence)
    Image.fromarray(overlay).save(OUT/f'{Path(name).stem}__gradcam_overlay.png')
    Image.fromarray(heat).save(OUT/f'{Path(name).stem}__gradcam_heatmap.png')
    results.append({'filename':name,'gray':gray,'heat':heat,'overlay':overlay,'label':label,'confidence':confidence_value,'map_size':tuple(activation.shape[-2:]),'lo':float(lo),'hi':float(hi)})

# Readable contact sheet: original, activation heatmap, overlay for each source.
W,H,HEADER,ROW,GAP=320,256,54,286,8
LEFT=8
sheet=Image.new('RGB',(LEFT+3*W+3*GAP,HEADER+len(results)*(ROW+GAP)),'white')
d=ImageDraw.Draw(sheet)
d.text((8,8),'ConvNeXt-Tiny Grad-CAM examples | ImageNet-pretrained, not thermal-trained',fill='black')
for j,title in enumerate(['Input (1–99% stretch)','Grad-CAM activation','Overlay on input']):
    d.text((LEFT+j*(W+GAP)+2,32),title,fill='black')
for i,r in enumerate(results):
    y=HEADER+i*(ROW+GAP)
    d.text((LEFT,y+2),f"{r['filename']} | top-1 ImageNet: {r['label']} ({r['confidence']:.1%})",fill='black')
    yimg=y+22
    sheet.paste(Image.fromarray(r['gray']).convert('RGB'),(LEFT,yimg))
    sheet.paste(Image.fromarray(r['heat']),(LEFT+(W+GAP),yimg))
    sheet.paste(Image.fromarray(r['overlay']),(LEFT+2*(W+GAP),yimg))
sheet.save(OUT/'gradcam_examples.png')

report='''# Grad-CAM examples for ConvNeXt-Tiny features

The contact sheet `gradcam_examples.png` shows four representative inputs, the Grad-CAM maps, and overlays. Each map targets that image's top-1 class from the pretrained ImageNet-1K classifier. The activation map is computed from the final ConvNeXt stage, then resized from its 8×10 feature grid to the input size for display.

**Interpretation:** warm colors indicate areas with stronger positive contribution to the selected ImageNet class score under Grad-CAM. This is a rough model-attention explanation, not a learned thermal importance map, a segmentation, a temperature map, or a scientific causal explanation. ConvNeXt-Tiny was pretrained on ImageNet natural images and was not trained or fine-tuned on these UAV thermal-like files. Its predicted class may be irrelevant. Treat the overlays as a way to inspect where this particular pretrained model responds, not as evidence that those areas matter to the real-world scene.

The input uses the same per-image 1–99% grayscale stretch as feature extraction, repeated across RGB. Original source arrays remain unchanged. Per-image overlays and color heatmaps are saved beside the comparison sheet.
'''
(OUT/'gradcam_report.md').write_text(report)
print('saved',len(results),'Grad-CAM examples')
for r in results: print(r['filename'],r['label'],f"{r['confidence']:.3f}",r['map_size'])
