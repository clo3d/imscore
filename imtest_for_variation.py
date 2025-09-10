from imscore.aesthetic.model import ShadowAesthetic, LAIONAestheticScorer
from imscore.hps.model import HPSv2
from imscore.mps.model import MPS
from imscore.preference.model import SiglipPreferenceScorer, CLIPScore
from imscore.pickscore.model import PickScorer
from imscore.imreward.model import ImageReward
from imscore.vqascore.model import VQAScore
from imscore.cyclereward.model import CycleReward
from imscore.evalmuse.model import EvalMuse

import os
import torch
import numpy as np
from PIL import Image
from einops import rearrange

# popular aesthetic/preference scorers
#model = ShadowAesthetic.from_pretrained("RE-N-Y/aesthetic-shadow-v2") # ShadowAesthetic aesthetic scorer (my favorite)
#model = CLIPScore.from_pretrained("RE-N-Y/clipscore-vit-large-patch14") # CLIPScore
#model = PickScorer.from_pretrained("RE-N-Y/pickscore") # PickScore preference scorer
#model = MPS.from_pretrained("RE-N-Y/mpsv1") # MPS (ovreall) preference scorer
#model = HPSv2.from_pretrained("RE-N-Y/hpsv21") # HPSv2.1 preference scorer
#model = ImageReward.from_pretrained("RE-N-Y/ImageReward") # ImageReward aesthetic scorer
model = LAIONAestheticScorer.from_pretrained("RE-N-Y/laion-aesthetic") # LAION aesthetic scorer
#model = CycleReward.from_pretrained('NagaSaiAbhinay/CycleReward-Combo') # CycleReward preference scorer.
#model = VQAScore.from_pretrained("RE-N-Y/clip-t5-xxl")
model=model.to('cuda')
#model = EvalMuse.from_pretrained("RE-N-Y/evalmuse")

# multimodal (pixels + text) preference scorers trained on PickaPicv2 dataset 
#model = SiglipPreferenceScorer.from_pretrained("RE-N-Y/pickscore-siglip")

prompts = "photo of garment in the online shop, realistic film, photo in fashion magazin"





modelpath = ['SDXL_base','realvisXL_V5','JuggernautXL_Ragnarok','cyber_realistic','ipadapter']
modelpath = ['flux_schnell/20steps']
vary = ['weakout','0.9','0.8','0.7','0.6','0.5','0.4','0.3','flat']

for model_vary in modelpath:
    prefix = '/mnt/DAS2/avatar_studio/variation_test_images/result/'
    mid = model_vary
    lpath = os.path.join(prefix,mid)
    imglist = os.listdir(lpath)
    vscore = [[],[],[],[],[],[],[],[],[],[],[]]
    
    for img in imglist:
        fpath = os.path.join(lpath,img)
        for idx,v in enumerate(vary):
            if v in fpath:
                break
        pixels = Image.open(fpath)
        pixels = np.array(pixels)
        pixels = rearrange(torch.tensor(pixels), "h w c -> 1 c h w") / 255.0
        pixels = pixels.to('cuda')
        score = model.score(pixels, prompts).detach().cpu().numpy()
       # print(fpath,score)
        vscore[idx].append(score[0])
 #   print(vscore)
    print(model_vary,' avg -----')
    for vs in vscore:
        print(np.mean(vs))

        



