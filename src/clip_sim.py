import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModel
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


vgg = models.vgg16(pretrained=True)
# 분류 레이어 없이 feature extractor 역할만 사용
feature_extractor = torch.nn.Sequential(*list(vgg.features.children()), torch.nn.AdaptiveAvgPool2d((7,7)))
feature_extractor.eval()
for p in feature_extractor.parameters():
    p.requires_grad = False

# 2) 전처리 정의
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])


def cosine_sim(a, b):
    return F.cosine_similarity(a, b).item()
    
def get_embedding(path, device='cpu'):
    img = Image.open(path).convert('RGB')
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feature_extractor(x)
    emb = torch.flatten(feat, start_dim=1)
    emb = F.normalize(emb, p=2, dim=1)
    return emb




#with dinov2
#processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
#model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
#model.eval()


#def get_embedding(path):
#    img = Image.open(path).convert("RGB")
#    inputs = processor(images=img, return_tensors="pt").to(device)
#    with torch.no_grad():
#        outputs = model(**inputs)
    # 마지막 hidden state를 평균풀링하여 임베딩 생성
 #   features = outputs.last_hidden_state.mean(dim=1)[0]
 #   return features / features.norm(p=2)

#def cosine_sim(a, b):
 #   return nn.CosineSimilarity(dim=0)(a, b).item()


# with clip
#model_name="openai/clip-vit-base-patch32"
#device = "cuda" if torch.cuda.is_available() else "cpu"
#model = CLIPModel.from_pretrained(model_name).to(device)
#processor = CLIPProcessor.from_pretrained(model_name)

#def get_embedding(path):
#    image = Image.open(path).convert("RGB")
#    inputs = processor(images=image, return_tensors="pt").to(device)
#    with torch.no_grad():
#        feats = model.get_image_features(**inputs)
#    return feats / feats.norm(dim=-1, keepdim=True)

#def cos_distance(emb1, emb2):
#    cos = torch.nn.functional.cosine_similarity(emb1, emb2)
#    return cos.item()

# 사용 예
#sim_score = compare_images_hf("image1.jpg", "image2.jpg")
#print(f"코사인 유사도: {sim_score:.4f}")

original_path = '/mnt/DAS2/avatar_studio/variation_test_images/all'
original_list = os.listdir(original_path)

origin_embds = []

print('get original embd...')
for p in original_list:
  pp = os.path.join(original_path,p)
  #img = Image.open(pp)
  emb = get_embedding(pp)
  origin_embds.append(emb)

print(len(origin_embds))

  


modelpath = ['SDXL_base','realvisXL_V5','JuggernautXL_Ragnarok', 'cyber_realistic','ipadapter']
vary = ['weakout','0.9','0.8','0.7','0.6','0.5','flat']

for model_vary in modelpath:
    prefix = '/mnt/DAS2/avatar_studio/variation_test_images/result/'
    mid = model_vary
    lpath = os.path.join(prefix,mid)
    imglist = os.listdir(lpath)
    vscore = [[],[],[],[],[],[],[],[]]
    
    print('get genereted embd... for ', lpath)
    
    for img in imglist:
        fpath = os.path.join(lpath,img)
        for idx,v in enumerate(vary):
            if v in fpath:
                break
        now = get_embedding(fpath)
        coslist = []
        for oe in origin_embds:
        
          coslist.append(cosine_sim(oe,now))
        #print(coslist)
        #print(np.max(coslist))
        vscore[idx].append(np.max(coslist))
          
        
    #print(vscore)
    print('avg')
    for vs in vscore:
        print(np.around(np.mean(vs), decimals=3) )

        
