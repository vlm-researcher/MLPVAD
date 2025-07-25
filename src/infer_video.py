from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch
from src.model import CLIPVAD
from utils.tools import get_batch_mask, get_prompt_text
import src.ucf_option
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Static HTML frontend
@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Load args, model, and prompt once
args = src.ucf_option.parser.parse_args([])
device = "cuda" if torch.cuda.is_available() else "cpu"
args.model_path = "model_ucf.pth"

model = CLIPVAD(args.classes_num, args.embed_dim, args.snippets, args.visual_width,
                args.visual_head, args.visual_layers, args.attn_window,
                args.prompt_prefix, args.prompt_postfix, args, device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()

label_map = {
    'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson',
    'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion',
    'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery',
    'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'
}
prompt_text = get_prompt_text(label_map)

def pad(feat, min_len):
    if feat.shape[0] <= min_len:
        return np.pad(feat, ((0, min_len - feat.shape[0]), (0, 0)), mode='constant')
    return feat

def process_split(feat, snippets):
    clip_length = feat.shape[0]
    if clip_length < snippets:
        return pad(feat, snippets).reshape(1, snippets, feat.shape[1]), clip_length
    split_num = int(clip_length / snippets) + 1
    result = []
    for i in range(split_num):
        chunk = feat[i * snippets : i * snippets + snippets, :]
        result.append(pad(chunk, snippets))
    return np.stack(result), clip_length

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        file_path = f"temp_feat.npy"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        clip_feat = np.load(file_path)
        clip_feat=clip_feat.squeeze()
        os.remove(file_path)

        clip_feat, clip_length = process_split(clip_feat, args.snippets)
        clip_feat = torch.tensor(clip_feat).float().to(device)

        maxlen = args.snippets
        lengths = torch.zeros(int(clip_length / maxlen) + 1)
        l = clip_length
        for j in range(len(lengths)):
            if j == 0 and l < maxlen:
                lengths[j] = l
            elif j == 0:
                lengths[j] = maxlen
                l -= maxlen
            elif l > maxlen:
                lengths[j] = maxlen
                l -= maxlen
            else:
                lengths[j] = l

        lengths = lengths.to(int)
        padding_mask = get_batch_mask(lengths, maxlen).to(device)

        with torch.no_grad():
            _, logits1, _, _, _ = model(clip_feat, padding_mask, prompt_text, lengths, True)
            logits1 = logits1.reshape(-1, logits1.shape[2])[:clip_length]
            prob1 = torch.sigmoid(logits1.squeeze(-1))
            avg_score = prob1.mean().item()

        result = "Anomaly" if avg_score > 0.5 else "Normal"

        # Generate plot
        fig, ax = plt.subplots()
        ax.plot(prob1.cpu().numpy(), label="Anomaly Score")
        ax.axhline(0.5, color='r', linestyle='--', label='Threshold')
        ax.set_title("Frame-wise Anomaly Scores")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Score")
        ax.legend()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return templates.TemplateResponse("result.html", {
                    "request": request,
                    "result": result,
                    "score": f"{avg_score:.4f}",
                    "plot": img_base64
                })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    