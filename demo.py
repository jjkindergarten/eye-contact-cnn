import dlib
import cv2
import argparse, os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from model import model_static
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from colour import Color
from time import time
from pathlib import Path
import json


parser = argparse.ArgumentParser()

parser.add_argument('--video', type=str, help='input video path. live cam is used when not specified')
parser.add_argument('--face', type=str, help='face detection file path. dlib face detector is used when not specified')
parser.add_argument('--model_weight', type=str, help='path to model weights file', default='data/model_weights.pkl')
parser.add_argument('--jitter', type=int, help='jitter bbox n times, and average results', default=0)
parser.add_argument('-save_vis', help='saves output as video', action='store_true')
parser.add_argument('-save_text', help='saves output as text', action='store_true')
parser.add_argument('-display_off', help='do not display frames', action='store_true')

args = parser.parse_args()

CNN_FACE_MODEL = 'data/mmod_human_face_detector.dat' # from http://dlib.net/files/mmod_human_face_detector.dat.bz2


def bbox_jitter(bbox_left, bbox_top, bbox_right, bbox_bottom):
    cx = (bbox_right+bbox_left)/2.0
    cy = (bbox_bottom+bbox_top)/2.0
    scale = random.uniform(0.8, 1.2)
    bbox_right = (bbox_right-cx)*scale + cx
    bbox_left = (bbox_left-cx)*scale + cx
    bbox_top = (bbox_top-cy)*scale + cy
    bbox_bottom = (bbox_bottom-cy)*scale + cy
    return bbox_left, bbox_top, bbox_right, bbox_bottom


def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)


def run(video_path, face_path, model_weight, jitter, vis, display_off, save_text):
    # set up vis settings
    red = Color("red")
    colors = list(red.range_to(Color("green"),10))
    font = ImageFont.truetype("data/arial.ttf", 40)

    # set up video source
    if video_path is None:
        cap = cv2.VideoCapture(0)
        video_path = 'live.avi'
    else:
        cap = cv2.VideoCapture(video_path)
        ori_fps = cap.get(cv2.CAP_PROP_FPS)

    tar_fps = 2
    fps_ratio = int(ori_fps//tar_fps)
    print('fps ratio {}'.format(fps_ratio))

    # set up output file
    if save_text:
        outtext_name = os.path.basename(video_path).replace('.avi','_output.txt')
        f = open(outtext_name, "w")
    if vis:
        outvis_name = os.path.basename(video_path).replace('.avi','_output.avi')
        imwidth = int(cap.get(3)); imheight = int(cap.get(4))
        outvid = cv2.VideoWriter(outvis_name,cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(5), (imwidth,imheight))

    # set up face detection mode
    if face_path is None:
        facemode = 'DLIB'
    else:
        facemode = 'GIVEN'
        column_names = ['frame', 'left', 'top', 'right', 'bottom']
        df = pd.read_csv(face_path, names=column_names, index_col=0)
        df['left'] -= (df['right']-df['left'])*0.2
        df['right'] += (df['right']-df['left'])*0.2
        df['top'] -= (df['bottom']-df['top'])*0.1
        df['bottom'] += (df['bottom']-df['top'])*0.1
        df['left'] = df['left'].astype('int')
        df['top'] = df['top'].astype('int')
        df['right'] = df['right'].astype('int')
        df['bottom'] = df['bottom'].astype('int')

    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        exit()

    if facemode == 'DLIB':
        cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)
    frame_cnt = 0

    # set up data transformation
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # load model weights
    model = model_static(model_weight)
    model_dict = model.state_dict()
    # snapshot = torch.load(model_weight, map_location='cpu')
    snapshot = torch.load(model_weight)
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

    model.cuda()
    model.train(False)

    # score_list
    score_list = []

    # video reading loop
    count = 0
    timing = time()
    while(cap.isOpened()):
        count += 1
        if count % fps_ratio == 0:
            print('frame {} is done, duration {}'.format(count, time()-timing))
            timing = time()
            ret, frame = cap.read()
            if ret == True:
                height, width, channels = frame.shape
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame_cnt += 1
                bbox = []
                if facemode == 'DLIB':
                    dets = cnn_face_detector(frame, 1)
                    for d in dets:
                        l = d.rect.left()
                        r = d.rect.right()
                        t = d.rect.top()
                        b = d.rect.bottom()
                        # expand a bit
                        l -= (r-l)*0.2
                        r += (r-l)*0.2
                        t -= (b-t)*0.2
                        b += (b-t)*0.2
                        bbox.append([l,t,r,b])
                elif facemode == 'GIVEN':
                    if frame_cnt in df.index:
                        bbox.append([df.loc[frame_cnt,'left'],df.loc[frame_cnt,'top'],df.loc[frame_cnt,'right'],df.loc[frame_cnt,'bottom']])

                if len(bbox) == 0:
                    score_list.append(0)
                frame = Image.fromarray(frame)
                for b in bbox[:1]:
                    face = frame.crop((b))
                    img = test_transforms(face)
                    img.unsqueeze_(0)
                    if jitter > 0:
                        for i in range(jitter):
                            bj_left, bj_top, bj_right, bj_bottom = bbox_jitter(b[0], b[1], b[2], b[3])
                            bj = [bj_left, bj_top, bj_right, bj_bottom]
                            facej = frame.crop((bj))
                            img_jittered = test_transforms(facej)
                            img_jittered.unsqueeze_(0)
                            img = torch.cat([img, img_jittered])

                    # forward pass
                    output = model(img.cuda())
                    # output = model(img)
                    if jitter > 0:
                        output = torch.mean(output, 0)
                    score = F.sigmoid(output).item()
                    score_list.append(score)

                    coloridx = min(int(round(score*10)),9)
                    draw = ImageDraw.Draw(frame)
                    drawrect(draw, [(b[0], b[1]), (b[2], b[3])], outline=colors[coloridx].hex, width=5)
                    draw.text((b[0],b[3]), str(round(score,2)), fill=(255,255,255,128), font=font)
                    if save_text:
                        f.write("%d,%f\n"%(frame_cnt,score))

                if not display_off:
                    frame = np.asarray(frame) # convert PIL image back to opencv format for faster display
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # cv2.imshow('',frame)
                    if vis:
                        outvid.write(frame)
                        # cv2.imwrite('test_' + '{0:04d}'.format(count) + '.jpg', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
            else:
                break
        else:
            ret = cap.grab()

    # videoname = os.path.basename(video_path)
    # pre, ext = os.path.splitext(videoname)
    # jsonname = pre + '.json'
    # print(score_list)
    # with open(jsonname, 'w') as f:
    #     json.dump(score_list, f)


    if vis:
        outvid.release()
    if save_text:
        f.close()
    cap.release()
    return score_list


if __name__ == "__main__":

    target_dir = './'

    if os.path.isdir(args.video):
        path_list = args.video.rglob('*.mp4')
        for f_path in path_list:
            f_ele = str(f_path).split('/')
            fn = f_ele[-1]
            f_score = f_ele[-2]
            score_list = run(str(f_path), args.face, args.model_weight, args.jitter, args.save_vis, args.display_off, args.save_text)
            target_path = os.path.join(target_dir, f_score, fn.replace('mp4', 'json'))
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, 'w') as f:
                json.dump(score_list, f)
    elif os.path.isfile(args.video):
        score_list = run(args.video, args.face, args.model_weight, args.jitter, args.save_vis, args.display_off,
                         args.save_text)
        f_ele = args.video.split('/')
        fn = f_ele[-1]
        f_score = f_ele[-2]
        target_path = os.path.join(target_dir, f_score, fn.replace('mp4', 'json'))
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, 'w') as f:
            json.dump(score_list, f)