import torch
from PIL import Image
from lavis.models import load_model_and_preprocess  

if __name__=='__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    raw_image = Image.open("./fighting.png").convert("RGB")
    caption = "two men fighting in a cage during a boxing match"
    model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    txt = text_processors["eval"](caption)
    # print(model)
    itm_output = model({"image": img, "text_input": txt}, match_head="itm", precomputed_features=False)
    itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
    print(f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')
    # itc_score = model({"image": img, "text_input": txt}, match_head='itc')
    # print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)