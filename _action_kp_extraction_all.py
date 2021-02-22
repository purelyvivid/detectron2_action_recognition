import os
from video_human_kp_detect import *
import numpy as np
import glob
import joblib
import time

SAVE_KPS_PTH = "predicted_kps" 
BASE_PTH = "../"*(len(os.getcwd().split("/"))-1)  #"../../../../../"
DATASET_PTH = BASE_PTH+"datasets/public/HMDB51"

filenames = sorted(glob.glob(DATASET_PTH+"/*/*.avi"))

predictor, cfg = prepare_predictor(THRESH=0.9)
visualizer = prepare_video_visualizer(cfg)

import subprocess 
if not os.path.isdir(SAVE_KPS_PTH):
    subprocess.run("mkdir "+SAVE_KPS_PTH, shell=True)#若沒有此路徑 則新增
#else:
    #subprocess.run("rm "+SAVE_KPS_PTH+"/*", shell=True)#若有此路徑 清空路徑下所有檔案
    
st = time.time()
for j, fn in enumerate(filenames):

    st_ = time.time()
    class_ = filenames[0].split("/")[-2] #video_info_df.loc[j,"class"]
    id_ = str(j).zfill(4)
    INFO_OUTPUT_PTH=f"{SAVE_KPS_PTH}/[{id_}]{class_}.joblib"
    txt = f"[{(j+1)}/{len(filenames)}] {INFO_OUTPUT_PTH}"
    if not os.path.isfile(INFO_OUTPUT_PTH):
        try:
            predict_single_video(predictor, visualizer, 
                                 VIDEO_FILE_PTH=fn, 
                                 INFO_OUTPUT_PTH=INFO_OUTPUT_PTH, 
                                 save_video=False, #VIDEO_OUTPUT_PTH="out.avi", 
                                 top_k_selected_roles=1,
                                 save_video_selected_roles=False, #VIDEO_OUTPUT_PTH_SELECTED_ROLES="out_s.avi",
                                 tracking_max_distance_limit=[27,50],
                                 print_status = False,
                                ) 
            print("{}  Spent Time [{:.2f}] sec".format(txt, time.time()-st_)) 
        except:
            print(f"{txt}  Error!")
    else:
        print(f"{txt}  Already Existed!")
print("Total Spent Time [{:.2f}] sec".format(time.time()-st))