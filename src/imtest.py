from imscore.aesthetic.model import ShadowAesthetic, LAIONAestheticScorer
from imscore.hps.model import HPSv2
from imscore.mps.model import MPS
from imscore.preference.model import SiglipPreferenceScorer, CLIPScore
from imscore.pickscore.model import PickScorer
from imscore.imreward.model import ImageReward
from imscore.vqascore.model import VQAScore
from imscore.cyclereward.model import CycleReward
from imscore.evalmuse.model import EvalMuse

import argparse
import os
import torch
import numpy as np
from PIL import Image
from einops import rearrange

import t2v_metrics.t2v_metrics as t2v_metrics



realism_prompts = [
                # Physics and common sense
    "Are the directions of the shadows from the objects consistent?", #0
    "Does the reflection in the image look natural?",  #1
    "Are the fingers on the person's hands all present and complete?", #2
    "Is this object floating in the air or touching a surface?", #3

    # Texture and detail evaluation
    "Does the surface of the person look smooth?", #4
    "Is the depiction of the person's hair detailed?", #5
    "Is the background sharp?" #6
]

artifacts_prompts = [
    # Coherence and distortion
    "Are there any awkward or distorted parts in the image?", #7 # False : positive 
    "Do the person's face or body parts look unnatural?", #8 # False : positive
    "Can you see the artifact in the background?", #9 # False : positive

    # Boundary and integrity assessment
    "Are the boundaries of the objects clear, or do they unnaturally blend with other objects?", #10
    "Is the overall color tone or lighting of the image consistent?" #11
]

overall_quality_prompts = [
    # Composition and aesthetics
    "Is the composition of this image stable?", #12
    "Is the color combination in the image harmonious?", #13
    "Is this image visually appealing?", #14

    # Contextual understanding
    "Does this image look like a scene from a story?" #15
]
        

# popular aesthetic/preference scorers
#model = ShadowAesthetic.from_pretrained("RE-N-Y/aesthetic-shadow-v2") # ShadowAesthetic aesthetic scorer (my favorite)
#model = CLIPScore.from_pretrained("RE-N-Y/clipscore-vit-large-patch14") # CLIPScore
#model = PickScorer.from_pretrained("RE-N-Y/pickscore") # PickScore preference scorer
#model = MPS.from_pretrained("RE-N-Y/mpsv1") # MPS (ovreall) preference scorer
#model = HPSv2.from_pretrained("RE-N-Y/hpsv21") # HPSv2.1 preference scorer
#model = ImageReward.from_pretrained("RE-N-Y/ImageReward") # ImageReward aesthetic scorer
# model = LAIONAestheticScorer.from_pretrained("RE-N-Y/laion-aesthetic") # LAION aesthetic scorer
#model = CycleReward.from_pretrained('NagaSaiAbhinay/CycleReward-Combo') # CycleReward preference scorer.
#model = VQAScore.from_pretrained("RE-N-Y/clip-t5-xxl")
# model=model.to('cuda')
#model = EvalMuse.from_pretrained("RE-N-Y/evalmuse")

# multimodal (pixels + text) preference scorers trained on PickaPicv2 dataset 
#model = SiglipPreferenceScorer.from_pretrained("RE-N-Y/pickscore-siglip")


def score_image(model_name, image_path, prompts):
    if model_name.lower() == "shadow":
        model = ShadowAesthetic.from_pretrained("RE-N-Y/aesthetic-shadow-v2")
    elif model_name.lower() == "hpsv2":
        model = HPSv2.from_pretrained("RE-N-Y/hpsv21")
    elif model_name.lower() == "laion":
        model = LAIONAestheticScorer.from_pretrained("RE-N-Y/laion-aesthetic")
    elif model_name.lower() == "vqascore":
        # model = VQAScore().to('cuda')
        clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    scorelist = []    
    if os.path.isdir( image_path ):
        imglist = os.listdir(image_path)
        if model_name.lower() == "vqascore":
             for img in imglist:
                # for p in prompts:
                fpath = os.path.join(args.dir,img)
                image =  Image.open(fpath)
                pixels = Image.open(fpath).convert("RGB")#.resize((384,680))
                pixels = np.array(pixels)
                pixels = rearrange(torch.tensor(pixels), "h w c -> 1 c h w") / 255.0
                score = clip_flant5_score(images=[fpath], texts=prompts)
                print(score)

        else:
            model = model.to('cuda')
            for img in imglist:
                for p in prompts:
                    fpath = os.path.join(args.dir,img)
                    pixels = Image.open(fpath).convert("RGB")#.resize((384,680))
                    pixels = np.array(pixels)
                    pixels = rearrange(torch.tensor(pixels), "h w c -> 1 c h w") / 255.0
                    pixels = pixels.to('cuda')
                    score = model.score(pixels, p).detach().cpu().numpy()
                    if model_name.lower()== 'vqascore':
                        score[7] = 1-score[7]  # reverse the score for prompt 7
                        score[8] = 1-score[8]  # reverse the score for prompt 7
                        score[9] = 1-score[9]  # reverse the score for prompt 7
                    print(fpath,score)
                    scorelist.append(score)
                print(scores)
            
    return scorelist,np.mean(scorelist)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image scoring script")
    parser.add_argument('--dir', type=str, required=True, help='Directory path containing images')
    parser.add_argument('--scorer', type=str, default='all', required=True, help='Scoring method (e.g., laion, shadow, etc.)')
    args = parser.parse_args()

    promptlist = ["protrait of person, realistic film, 8k, high quality, detailed, winner photography"]

    # # Select model based on input scorer
    # if args.scorer.lower() == "shadow":
    #     model = ShadowAesthetic.from_pretrained("RE-N-Y/aesthetic-shadow-v2")
    # elif args.scorer.lower() == "hpsv2":
    #     model = HPSv2.from_pretrained("RE-N-Y/hpsv21")
    # elif args.scorer.lower() == "laion":
    #     model = LAIONAestheticScorer.from_pretrained("RE-N-Y/laion-aesthetic")
    # elif args.scorer.lower() == "vqascore":
    #     model = VQAScore.from_pretrained("RE-N-Y/clip-t5-xxl")
    # else:
    #     raise ValueError(f"Unknown scorer: {args.scorer}")


    if args.scorer.lower() == 'all':
        args.scorerlist = ['hpsv2','laion','vqascore']
        for scorer in args.scorerlist:
            scolrelist, avg_score = score_image(args.scorer, args.dir, promptlist)
            print(f"Average score across images: {args.scorer} : {avg_score}")
        
    
    else:
        if args.scorer.lower() == 'vqascore' :
            promptlist = realism_prompts
            promptlist.extend(artifacts_prompts)
            promptlist.extend(overall_quality_prompts)
            print(promptlist)
        else:
            promptlist = ["protrait of person, realistic film, 8k, high quality, detailed, winner photography"]
        scolrelist, avg_score = score_image(args.scorer, args.dir, promptlist)
        print(f"Average score across images: {args.scorer} : {avg_score}")
