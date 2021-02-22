import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import tqdm
import cv2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
import time
from munkres import munkres_algo_w_match
import joblib
import torch
from detectron2.structures import Instances, Boxes

def load_image(IMAGE_FILE_PTH="./dogs.jpg"):
    imgarr = cv2.imread(IMAGE_FILE_PTH)
    return imgarr

def load_video(VIDEO_FILE_PTH='video-input.avi'):
    # Extract video properties
    video = cv2.VideoCapture(VIDEO_FILE_PTH)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_dict = {
        'width' : width,
        'height' : height,
        'frames_per_second' : frames_per_second,
        'num_frames' : num_frames,
    }
    return video, video_dict


def prepare_predictor(THRESH=0.5):
    cfg = get_cfg()
    CFG_FILE_PTH = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"
    #CFG_FILE_PTH = "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(CFG_FILE_PTH))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESH  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CFG_FILE_PTH)
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

def predict_single_image(predictor, imgarr):
    outputs = predictor(imgarr)
    return outputs

def prepare_video_writer(video_dict, VIDEO_OUTPUT_PTH='out.avi'):
    # Initialize video writer
    video_writer = cv2.VideoWriter(VIDEO_OUTPUT_PTH, 
                                   fourcc=cv2.VideoWriter_fourcc(*'XVID'), 
                                   fps=float(video_dict['frames_per_second']), 
                                   frameSize=(video_dict['width'], video_dict['height']), 
                                   isColor=True)
    return video_writer


def prepare_video_visualizer(cfg):
    # Initialize visualizer
    visualizer = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)
    return visualizer

def visualize_video_frame(visualizer, frame, outputs):
    # Draw a visualization of the predictions using the video visualizer
    visualization = visualizer.draw_instance_predictions(frame, outputs["instances"].to("cpu")).get_image()
    
    # Convert Matplotlib RGB format to OpenCV BGR format
    #visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

    return visualization

def runOnVideo(video, maxFrames):
    """ Runs  on every frame in the video (unless maxFrames is given)
    """
    
    readFrames = 0
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break

        yield frame

        readFrames += 1
        if readFrames > maxFrames:
            break
            
def collect_outputs_info(outputs, video_dict, tracking_max_distance_limit, frame_i=0, allow_detect_delay_frame=5, tracking_cost_matrix_weight_list=[1.0, 0.5]):
    
    # Get keypoints/boxes information 
    kps = outputs['instances'].get('pred_keypoints').to("cpu").numpy() #shape = (#person, 17, 3)
    boxes = outputs['instances'].get('pred_boxes').tensor.to("cpu").numpy() #shape = (#person, 4)
        
    # Write to cut (frames_split_pos)
    n = boxes.shape[0] #目前這個frame出現的角色數目
    video_dict["frames_split_pos"].append(video_dict["frames_split_pos"][-1]+n)
        
    # Write to role_buffer
    if video_dict["roles_buffer"]==[]:
        video_dict["roles_buffer"].append(np.array(range(n))) #如果是第一個frame, 所有出現的人給予新角色
        video_dict["num_role_register"] += n
    else:
        
        #容忍短時間內沒偵測到
        j = 1
        last_kps = video_dict["kps_buffer"][-j][:,:,:2].reshape((-1,17*2))
        last_boxes = video_dict["boxes_buffer"][-j]
        last_roles = video_dict["roles_buffer"][-j]
        while len(last_boxes)==0 and (j<=(1+allow_detect_delay_frame)):# -1~-5
            last_kps = video_dict["kps_buffer"][-j][:,:,:2].reshape((-1,17*2))
            last_boxes = video_dict["boxes_buffer"][-j]
            last_roles = video_dict["roles_buffer"][-j]
            j+=1
                
        if len(boxes)==0:
            role = np.array([])
        elif len(last_boxes)==0: # len(boxes)>0
            n, add_n = video_dict["num_role_register"], len(boxes)
            role = np.array(range(n, n+add_n))
            video_dict["num_role_register"]=n+add_n
        else: # len(boxes)>0 and len(last_boxes)>0
            """
            kps and boxes both take into consideration
            """
            kps_ = kps[:,:,:2].reshape((-1,17*2))   
            get_center = lambda bs : np.concatenate([(bs[:,2:3]+bs[:,0:1])/2, (bs[:,3:4]+bs[:,1:2])/2], 1)
            boxes_center = get_center(boxes)
            last_boxes_center = get_center(last_boxes)
            feat_list = [boxes_center, kps_]
            last_feat_list = [last_boxes_center, last_kps ] 
            
            role, video_dict["num_role_register"], cost_matrix = \
            munkres_algo_w_match(feat_list, last_feat_list, tracking_max_distance_limit, tracking_cost_matrix_weight_list, last_roles, video_dict["num_role_register"]) #物件追蹤
            video_dict["cost_matrix_collection"][frame_i] = cost_matrix 
            
        video_dict["roles_buffer"].append(role) 

    # Write to boxes buffer
    video_dict["boxes_buffer"].append(boxes)
        
    # Write to kps buffer
    #kps[:,:,0] /= video_dict['width']
    #kps[:,:,1] /= video_dict['height']
    video_dict["kps_buffer"].append(kps)
    
    return video_dict

def collect_video_info(predictor, video, video_dict, tracking_max_distance_limit):
    
    video_dict.update({
        "num_role_register": 0,  #init #新角色由這個號碼開始編碼
        "boxes_buffer": [] ,     #init #每個frame出現的 人 bounding boxes座標 #np.array shape:(sum of #person, 4)
        "kps_buffer": [] ,       #init #每個frame出現的 人 肢體關鍵點座標      #np.array shape:(sum of #person, 17, 3)
        "roles_buffer": [] ,     #init #每個frame出現的 人 角色編碼(不切)      #np.array shape:(sum of #person, )
        "frames_split_pos": [0], #init #For以上三個buffer, 每個frame的切分位置 #np.array shape:(num_frames-1, )
        "cost_matrix_collection": [None]*video_dict['num_frames'], #init 
    })
    
    # Enumerate the frames of the video
    for i, (frame) in enumerate(runOnVideo(video, video_dict['num_frames'])):
        
        # Get prediction results for this frame
        outputs = predict_single_image(predictor, frame)
        if i==0:
            video_dict.update({
                "image_size": outputs["instances"].image_size
            })
            
        # Make sure the frame is colored
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Update information dict
        video_dict = collect_outputs_info(outputs, video_dict, tracking_max_distance_limit, frame_i=i)
        
    video_dict["roles_buffer"] = np.concatenate(video_dict["roles_buffer"]).astype(int)
    video_dict["boxes_buffer"] = np.concatenate(video_dict["boxes_buffer"], 0)
    video_dict["kps_buffer"] = np.concatenate(video_dict["kps_buffer"], 0)
    video_dict["frames_split_pos"] = video_dict["frames_split_pos"][1:-1]#切分點 去除頭尾 才能直接套用 np.split
    return video_dict

def select_top_k_role(video_dict, top_k=1, print_status=False ):#選出角色 k 個
    num_role_total = video_dict['num_role_register']

    is_show_by_role = np.zeros(num_role_total,)#每個角色出現過了沒
    bbox_area_by_role = np.zeros(num_role_total,)#每個角色出現的bbox總面積
    num_frames_by_role = np.zeros(num_role_total,)#每個角色出現的頻率（frame數目）
    #motion_distance_by_role = np.zeros(num_role_total,)#每個角色出現的bbox center總移動距離
    motion_distance_by_role = np.zeros(num_role_total,)#每個角色出現的kps總移動距離
    frames_by_role = [[]]*num_role_total #每個角色出現在哪些frame
    completeness_by_role = np.zeros(num_role_total,)#每個角色是否完整（沒碰到邊緣） 預設為不完整
    
    #last_pos_by_role = np.full((num_role_total,2), np.nan)#每個角色上次出現的bbox center位置 
    last_pos_by_role = np.full((num_role_total,17,2), np.nan)#每個角色上次出現的kps位置 
    
    kps_split = np.split(video_dict["kps_buffer"], video_dict["frames_split_pos"])
    boxes_split = np.split(video_dict["boxes_buffer"], video_dict["frames_split_pos"])
    roles_split = np.split(video_dict["roles_buffer"], video_dict["frames_split_pos"])
    
    for frame_i, (frame_roles, frame_boxes, frame_kps) in enumerate(zip(roles_split,boxes_split,kps_split)):# for each frame
        for (role, box, kps) in zip(frame_roles,frame_boxes, frame_kps): # for each role in this frame
            """
            video_dict['image_size'] 
                eg. (240, 320)
            
            bbox: (4,)
                x1,  y1,  x2, y2
                eg. (221.4248,  35.1949, 271.2917, 155.6825)
                eg. (94.5497,  33.5201, 133.6174, 146.7651)
                eg. (283.49512  ,   0.       , 320.       , 235.78056  )
            
            kps: (17,3)
            """
            frames_by_role[role].append(frame_i)
            x1,  y1, x2, y2 = box
            area = (x2-x1)*(y2-y1)
            #center = np.array([np.mean([x1, x2]), np.mean([y1, y2])])
            
            H, W = video_dict['image_size'] # eg. (240, 320)
            x_low, y_low, x_high, y_high = int(W*0.05), int(H*0.05), int(W*0.95), int(H*0.95)
            completeness_x = (x1>x_low) and (x2<x_high)
            completeness_y = (y1>y_low) and (y2<y_high)
            full_x = (x2-x1)>int(W*0.8)
            full_y = (y2-y1)>int(H*0.8)
            if  (completeness_x or full_x) and (completeness_y or full_y):
                completeness_by_role[role] += 1
            
            if is_show_by_role[role]==0:#如果是第一次創立這個角色
                motion_distance = 0
                is_show_by_role[role] = 1
                if print_status: print(f"  角色[{role}]出現在 frame [{frame_i}]")
            else: #如果非第一次創立這個角色,　last_pos_by_role[role]的值已非　np.nan
                #motion_distance = np.sum((center-last_pos_by_role[role])**2)**0.5
                motion_distance = np.mean(np.sum((kps[:,:2]-last_pos_by_role[role])**2, 1)**0.5)
            bbox_area_by_role[role] += area
            num_frames_by_role[role] += 1
            motion_distance_by_role[role] += motion_distance
            #last_pos_by_role[role] = center #update
            last_pos_by_role[role] = kps[:,:2]
    
    video_dict.update({
        "bbox_area_by_role" : bbox_area_by_role, 
        "num_frames_by_role" : num_frames_by_role, 
        "motion_distance_by_role" : motion_distance_by_role,
        "frames_by_role":frames_by_role,
        "completeness_by_role":completeness_by_role,
    })    
    
    # 選擇 移動最多(o)/最頻繁出現(o)/bbox最大(x)       
    frequency_score = num_frames_by_role / video_dict["num_frames"]
    completeness_score = completeness_by_role / video_dict["num_frames"]
    motion_score = (motion_distance_by_role/(bbox_area_by_role/num_frames_by_role)**0.5)*completeness_score
    
    if print_status: 
        print("所有角色 frequency_score :", frequency_score )
        print("所有角色 completeness_score :", completeness_score )
        print("所有角色 motion_score :",  motion_score)   

    
    score =  frequency_score + 1.5*motion_score + completeness_score #比重可調
    selected_roles = np.flip(np.argsort(score)[-top_k:])
    
    if print_status: 
        print("selected_roles(由分數大到小): ", selected_roles)
        print("  selected角色 frequency_score :", frequency_score[selected_roles] )
        print("  selected角色 completeness_score :", completeness_score[selected_roles] )
        print("  selected角色 motion_score :",  motion_score[selected_roles]) 

    return selected_roles, video_dict

def get_kps_by_role(kps, roles, selected_role=0):
    if sum(roles==selected_role)>0:
        idx = np.argmax(roles==selected_role)
        return torch.tensor(kps[idx]) #(17,3)
    else:
        return torch.zeros((17,3))-1
        
def get_box_by_role(boxes, roles, selected_role=0):
    if sum(roles==selected_role)>0:
        idx = np.argmax(roles==selected_role)
        return torch.tensor(boxes[idx])#(4,)
    else:
        return torch.zeros((4))-1

def collect_video_info_by_roles(video_dict, selected_roles=[0]):
    
    video_dict.update({
        "selected_roles": selected_roles,  #選出角色 k 個
        "boxes_buffer_selected": [] ,  #init #每個角色在 frame出現的bounding boxes座標 #np.array shape:(#frame, k, 4)
        "kps_buffer_selected": [] ,    #init #每個角色在每個frame出現的肢體關鍵點座標   #np.array shape:(#frame, k, 17, 3)
    })
    
    kps_split = np.split(video_dict["kps_buffer"], video_dict["frames_split_pos"])
    boxes_split = np.split(video_dict["boxes_buffer"], video_dict["frames_split_pos"])
    roles_split = np.split(video_dict["roles_buffer"], video_dict["frames_split_pos"]) 
    
    for kps, boxes, roles in zip(kps_split, boxes_split, roles_split): # for each frame
        kps_selected = np.stack([ get_kps_by_role(kps, roles, selected_role=role) for role in selected_roles ], 0).astype(np.float32)#shape:(k, 17, 3)
        boxes_selected = np.stack([ get_box_by_role(boxes, roles, selected_role=role) for role in selected_roles ], 0).astype(np.float32)#shape:(k, 4)
        video_dict["kps_buffer_selected"].append(kps_selected)
        video_dict["boxes_buffer_selected"].append(boxes_selected)
    video_dict["boxes_buffer_selected"] = np.stack(video_dict["boxes_buffer_selected"], 0)#shape:(#frame, k, 4)
    video_dict["kps_buffer_selected"] = np.stack(video_dict["kps_buffer_selected"], 0)#shape:(#frame, k, 17, 3)       
    return video_dict

def visualize_video(visualizer, video, video_dict, video_writer, use_selected=False):
    
    if use_selected:
        kps_split = video_dict["kps_buffer_selected"]
        boxes_split = video_dict["boxes_buffer_selected"]
    else:
        kps_split = np.split(video_dict["kps_buffer"], video_dict["frames_split_pos"])
        boxes_split = np.split(video_dict["boxes_buffer"], video_dict["frames_split_pos"])        
     
    # Enumerate the frames of the video
    for i, (frame) in enumerate(tqdm.tqdm(runOnVideo(video, video_dict['num_frames']), total=video_dict['num_frames'])):
        
        # Reconstruct outputs
        ins = Instances(image_size=video_dict["image_size"])#create new instances for all frames by one specific role
        kps_new = torch.tensor(kps_split[i]).to(torch.float32)
        boxes_new = torch.tensor(boxes_split[i]).to(torch.float32)
        num_person = kps_new.shape[0]
        ins.set('pred_keypoints', kps_new)
        ins.set('pred_boxes', Boxes(boxes_new))
        ins.set('pred_classes', torch.tensor([0]*num_person).to(torch.int64) )    
        outputs = {'instances':ins}

        # Draw a visualization of the predictions using the video visualizer
        visualization = visualize_video_frame(visualizer, frame, outputs)
         
        # Write to video file
        video_writer.write(visualization)    
  
            
def predict_single_video(predictor, visualizer, 
                         VIDEO_FILE_PTH, INFO_OUTPUT_PTH, 
                         save_video=False, VIDEO_OUTPUT_PTH="out.avi", 
                         top_k_selected_roles=2,
                         save_video_selected_roles=False, VIDEO_OUTPUT_PTH_SELECTED_ROLES="out.avi", 
                         tracking_max_distance_limit=[27,50],
                         print_status = False,
                        ):
    
    if print_status: print("收集 video 內角色的關鍵點..", flush=True)
    video, video_dict = load_video(VIDEO_FILE_PTH=VIDEO_FILE_PTH)
    video_dict = collect_video_info(predictor, video, video_dict, tracking_max_distance_limit)
    video.release();cv2.destroyAllWindows()
    
    if save_video:
        if print_status: print("視覺化關鍵點 並輸出video..", flush=True)
        video_writer = prepare_video_writer(video_dict, VIDEO_OUTPUT_PTH=VIDEO_OUTPUT_PTH)
        video, _ = load_video(VIDEO_FILE_PTH=VIDEO_FILE_PTH)#須重load一次video
        visualize_video(visualizer, video, video_dict, video_writer, use_selected=False)
        video.release(); video_writer.release();cv2.destroyAllWindows()
        if print_status: print(f"save video [{VIDEO_OUTPUT_PTH}]")
        
    if print_status: print("選出 移動最多/最頻繁出現/最完整 的角色 k 個..")
    selected_roles, video_dict = select_top_k_role(video_dict, top_k_selected_roles, print_status=print_status) 
    
    if print_status: print("重新篩選出這 k 個角色的關鍵點資訊..")
    video_dict = collect_video_info_by_roles(video_dict, selected_roles=selected_roles)
    
    if save_video:
        if print_status: print("視覺化k 個角色的關鍵點 並輸出video..", flush=True)
        video_writer = prepare_video_writer(video_dict, VIDEO_OUTPUT_PTH=VIDEO_OUTPUT_PTH_SELECTED_ROLES)
        video, _ = load_video(VIDEO_FILE_PTH=VIDEO_FILE_PTH)#須重load一次video
        visualize_video(visualizer, video, video_dict, video_writer, use_selected=True)
        video.release(); video_writer.release();cv2.destroyAllWindows()
        if print_status: print(f"save video [{VIDEO_OUTPUT_PTH_SELECTED_ROLES}]")
   
    # video_dict 存檔
    joblib.dump(video_dict, INFO_OUTPUT_PTH)
    if print_status: print(f"save video_dict [{INFO_OUTPUT_PTH}]")
    
    
    