import urllib.request
import json
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as trans
import torch
import argparse
import sys
import os
import io
import json
from torchvision import models
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
from dtaidistance import dtw

"""

Run Alpha Pose

"""

red = (0, 0, 255)
green = (0, 255, 0)


# 비디오를 넣어 알파포즈 돌리기 // video_sk: path, video_json: [dict,...]
def alphapose_vid(video):
    out_img = "/".join(video.split('/')[:-1]) + '/result'
    scr = 'python3 scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --video ' + \
        video + ' --save_video --outdir ' + out_img
    os.system(scr)
    video_json = [out_img + '/' + video.split('/')[-1].split('.')[0] + '.json']
    video_sk = [out_img + '/' + video.split('/')[-1]]
    return video_sk, video_json


# 이미지를 넣어 알파포즈 돌리기 // cap_sk: [path,...], cap_json: [dict,...]
def alphapose_img(img_list):
    cap_json, cap_sk = [], []
    img_list = list(
        map(lambda x: x.strip(), img_list.rstrip(']').lstrip('[').split(',')))
    for img in img_list:
        out_img = "/".join(img.split('/')[:-1]) + '/result'
        scr = 'python3 scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --image ' + \
            img + ' --save_img --outdir ' + out_img
        os.system(scr)
        cap_json.append(out_img + '/' + img.split('/')
                        [-1].split('.')[0] + '.json')
        cap_sk.append(out_img + '/vis/' + img.split('/')[-1])
    return cap_sk, cap_json


def l2_normalize(jsonfile):  # 알파포즈에서 리턴된 json파일에서 l2_norm을 계산한다.
    ret = []
    for _ in jsonfile:
        with open(_) as kps:
            json_kps = json.load(kps)

        for frame in range(len(json_kps)):
            keypoints = json_kps[frame]['keypoints']
            box = json_kps[frame]['box']
            temp_x = np.abs(box[0] - box[2]) / 2
            temp_y = np.abs(box[1] - box[3]) / 2

            if temp_x <= temp_y:
                if box[0] <= box[2]:
                    sub_x = box[0] - (temp_y - temp_x)
                else:
                    sub_x = box[2] - (temp_y - temp_x)

                if box[1] <= box[3]:
                    sub_y = box[1]
                else:
                    sub_y = box[3]
            else:
                if box[1] <= box[3]:
                    sub_y = box[1] - (temp_x - temp_y)
                else:
                    sub_y = box[3] - (temp_x - temp_y)

                if box[0] <= box[2]:
                    sub_x = box[0]
                else:
                    sub_x = box[2]

            temp = []
            for _ in range(17):
                temp.append(keypoints[_ * 3] - sub_x)
                temp.append(keypoints[_ * 3 + 1] - sub_y)

            norm = np.linalg.norm(temp)
            for _ in range(17):
                keypoints[_ * 3] = (keypoints[_ * 3] - sub_x) / norm
                keypoints[_ * 3 + 1] = (keypoints[_ * 3 + 1] - sub_y) / norm
                json_kps[frame]['keypoints'] = keypoints
            ret.append(json_kps)
    return ret


def cos_sim(cap_json_l2, video_json_l2, video, video_json, cap_score):
    capscore_list, cap_list = [], []
    ip_data = video_json_l2

    for idx, label in enumerate(cap_json_l2):

        high_score = 0
        highlight = ''
        score_list = []

        for frame in range(len(ip_data)):
            ip_kpt = ip_data[frame]['keypoints']
            ip_list = []
            label_list = []

            for _ in range(17):
                temp_ip = []
                temp_la = []
                ip_list.append(ip_kpt[_ * 3])
                ip_list.append(ip_kpt[_ * 3 + 1])
                label_list.append(label[_ * 3])
                label_list.append(label[_ * 3 + 1])

                temp_ip.append(ip_kpt[_ * 3])
                temp_ip.append(ip_kpt[_ * 3 + 1])
                temp_la.append(label[_ * 3])
                temp_la.append(label[_ * 3 + 1])

                cs_temp = np.dot(temp_ip, temp_la) / \
                    (np.linalg.norm(temp_ip)*np.linalg.norm(temp_la))
                score_list.append(((2 - np.sqrt(2 * (1 - cs_temp))) / 2) * 100)

            cs_temp = np.dot(ip_list, label_list) / \
                (np.linalg.norm(ip_list)*np.linalg.norm(label_list))
            score = ((2 - np.sqrt(2 * (1 - cs_temp))) / 2) * 100

            if high_score <= score:
                high_score = score
                highlight = ip_data[frame]['image_id']
                sk_kp = list(
                    map(lambda x: round(x), video_json[frame]['keypoints']))
                score_detail = score_list

        cap = cv2.VideoCapture(video)
        fr_iter = 0

        score_json = {}
        score_json['Frame_Num'] = highlight.split('.')[0]
        score_json['Total_Score'] = high_score
        score_json['Head'] = np.mean(score_detail[0:5])
        score_json['LShoulder'] = score_detail[5]
        score_json['RShoulder'] = score_detail[6]
        score_json['LElbow'] = score_detail[7]
        score_json['RElbow'] = score_detail[8]
        score_json['LWrist'] = score_detail[9]
        score_json['RWrist'] = score_detail[10]
        score_json['LHip'] = score_detail[11]
        score_json['RHip'] = score_detail[12]
        score_json['LKnee'] = score_detail[13]
        score_json['Rknee'] = score_detail[14]
        score_json['LAnkle'] = score_detail[15]
        score_json['RAnkle'] = score_detail[16]
        capscore_list.append(score_json)

        while(cap.isOpened()):
            try:
                ret, frame = cap.read()
                if fr_iter == int(highlight.split('.')[0]):  # 점수에 따라 색칠하기
                    sk_sc = cap_score[idx]
                    ck = [0, 0, 0, 0]
                    x, y = 0, 0
                    for _ in range(4):  # head
                        x += sk_kp[_*3]
                        y += sk_kp[_*3+1]
                    x //= 4
                    y //= 4
                    if score_json['Head'] >= sk_sc:  # 초록색
                        cv2.line(frame, (x, y), (x, y), green, thickness=15)
                    else:
                        cv2.line(frame, (x, y), (x, y),
                                 red, thickness=15)  # 빨간색

                    for i, v in enumerate(score_json.values()):
                        if i < 2:
                            continue
                        if v >= sk_sc:
                            ck.append(green)
                        else:
                            ck.append(red)

                    if ck[5] == green and ck[6] == green:
                        cv2.line(
                            frame, (sk_kp[5*3], sk_kp[5*3+1]), (sk_kp[6*3], sk_kp[6*3+1]), green, thickness=5)
                    else:
                        cv2.line(
                            frame, (sk_kp[5*3], sk_kp[5*3+1]), (sk_kp[6*3], sk_kp[6*3+1]), red, thickness=5)
                    if ck[5] == green and ck[7] == green:
                        cv2.line(
                            frame, (sk_kp[5*3], sk_kp[5*3+1]), (sk_kp[7*3], sk_kp[7*3+1]), green, thickness=5)
                    else:
                        cv2.line(
                            frame, (sk_kp[5*3], sk_kp[5*3+1]), (sk_kp[7*3], sk_kp[7*3+1]), red, thickness=5)
                    if ck[5] == green and ck[11] == green:
                        cv2.line(
                            frame, (sk_kp[5*3], sk_kp[5*3+1]), (sk_kp[11*3], sk_kp[11*3+1]), green, thickness=5)
                    else:
                        cv2.line(
                            frame, (sk_kp[5*3], sk_kp[5*3+1]), (sk_kp[11*3], sk_kp[11*3+1]), red, thickness=5)
                    if ck[6] == green and ck[8] == green:
                        cv2.line(
                            frame, (sk_kp[6*3], sk_kp[6*3+1]), (sk_kp[8*3], sk_kp[8*3+1]), green, thickness=5)
                    else:
                        cv2.line(
                            frame, (sk_kp[6*3], sk_kp[6*3+1]), (sk_kp[8*3], sk_kp[8*3+1]), red, thickness=5)
                    if ck[6] == green and ck[12] == green:
                        cv2.line(
                            frame, (sk_kp[6*3], sk_kp[6*3+1]), (sk_kp[12*3], sk_kp[12*3+1]), green, thickness=5)
                    else:
                        cv2.line(
                            frame, (sk_kp[6*3], sk_kp[6*3+1]), (sk_kp[12*3], sk_kp[12*3+1]), red, thickness=5)
                    if ck[7] == green and ck[9] == green:
                        cv2.line(
                            frame, (sk_kp[7*3], sk_kp[7*3+1]), (sk_kp[9*3], sk_kp[9*3+1]), green, thickness=5)
                    else:
                        cv2.line(
                            frame, (sk_kp[7*3], sk_kp[7*3+1]), (sk_kp[9*3], sk_kp[9*3+1]), red, thickness=5)
                    if ck[8] == green and ck[10] == green:
                        cv2.line(
                            frame, (sk_kp[8*3], sk_kp[8*3+1]), (sk_kp[10*3], sk_kp[10*3+1]), green, thickness=5)
                    else:
                        cv2.line(
                            frame, (sk_kp[8*3], sk_kp[8*3+1]), (sk_kp[10*3], sk_kp[10*3+1]), red, thickness=5)
                    if ck[11] == green and ck[12] == green:
                        cv2.line(
                            frame, (sk_kp[11*3], sk_kp[11*3+1]), (sk_kp[12*3], sk_kp[12*3+1]), green, thickness=5)
                    else:
                        cv2.line(
                            frame, (sk_kp[11*3], sk_kp[11*3+1]), (sk_kp[12*3], sk_kp[12*3+1]), red, thickness=5)
                    if ck[11] == green and ck[13] == green:
                        cv2.line(
                            frame, (sk_kp[11*3], sk_kp[11*3+1]), (sk_kp[13*3], sk_kp[13*3+1]), green, thickness=5)
                    else:
                        cv2.line(
                            frame, (sk_kp[11*3], sk_kp[11*3+1]), (sk_kp[13*3], sk_kp[13*3+1]), red, thickness=5)
                    if ck[12] == green and ck[14] == green:
                        cv2.line(
                            frame, (sk_kp[12*3], sk_kp[12*3+1]), (sk_kp[14*3], sk_kp[14*3+1]), green, thickness=5)
                    else:
                        cv2.line(
                            frame, (sk_kp[12*3], sk_kp[12*3+1]), (sk_kp[14*3], sk_kp[14*3+1]), red, thickness=5)
                    if ck[13] == green and ck[15] == green:
                        cv2.line(
                            frame, (sk_kp[13*3], sk_kp[13*3+1]), (sk_kp[15*3], sk_kp[15*3+1]), green, thickness=5)
                    else:
                        cv2.line(
                            frame, (sk_kp[13*3], sk_kp[13*3+1]), (sk_kp[15*3], sk_kp[15*3+1]), red, thickness=5)
                    if ck[14] == green and ck[16] == green:
                        cv2.line(
                            frame, (sk_kp[14*3], sk_kp[14*3+1]), (sk_kp[16*3], sk_kp[16*3+1]), green, thickness=5)
                    else:
                        cv2.line(
                            frame, (sk_kp[14*3], sk_kp[14*3+1]), (sk_kp[16*3], sk_kp[16*3+1]), red, thickness=5)
                    for _ in range(5, 17):
                        cv2.line(
                            frame, (sk_kp[_*3], sk_kp[_*3+1]), (sk_kp[_*3], sk_kp[_*3+1]), ck[_], thickness=15)

                    cap_path = str(idx) + '.png'
                    cap_list.append(cap_path)
                    cv2.imwrite(cap_path, frame)
                    break
                fr_iter += 1
            except:
                continue

    return cap_list, capscore_list


def dtw_compare(pro_json_l2, stu_json_l2):  # 두 비디오의 l2_json 파일을 비교한다.

    label = pro_json_l2
    ip_data = stu_json_l2

    label_list = []
    ip_list = []
    score = []
    score_json = {}

    for _ in range(17):
        temp_x = []
        temp_y = []
        for frame in range(len(label)):
            temp = label[frame]['keypoints']
            temp_x.append(temp[_ * 3])
            temp_y.append(temp[_ * 3 + 1])
        label_list.append(temp_x)
        label_list.append(temp_y)

    for _ in range(17):
        temp_x = []
        temp_y = []
        for frame in range(len(ip_data)):
            temp = ip_data[frame]['keypoints']
            temp_x.append(temp[_ * 3])
            temp_y.append(temp[_ * 3 + 1])
        ip_list.append(temp_x)
        ip_list.append(temp_y)

    for _ in range(17):
        score_x = dtw.distance(label_list[_ * 2], ip_list[_ * 2])
        score_y = dtw.distance(label_list[_ * 2 + 1], ip_list[_ * 2 + 1])
        score_temp = []
        score_temp.append(100 - (score_x * 100))
        score_temp.append(100 - (score_y * 100))
        score.append(np.mean(score_temp))

    score_json['Total_Score'] = np.mean(score)
    score_json['Head'] = np.mean(score[0:5])
    score_json['LShoulder'] = score[5]
    score_json['RShoulder'] = score[6]
    score_json['LElbow'] = score[7]
    score_json['RElbow'] = score[8]
    score_json['LWrist'] = score[9]
    score_json['RWrist'] = score[10]
    score_json['LHip'] = score[11]
    score_json['RHip'] = score[12]
    score_json['LKnee'] = score[13]
    score_json['Rknee'] = score[14]
    score_json['LAnkle'] = score[15]
    score_json['RAnkle'] = score[16]

    return score_json


app = Flask(__name__)
CORS(app)


@app.route('/professor', methods=['GET'])
def professor():
    if request.method == 'GET':
        video = request.args.get('video', '')
        cap_list = request.args.get('capture', '')
        cap_score = request.args.get('score', '')
        video_sk, video_json = alphapose_vid(video)
        cap_sk, cap_json = alphapose_img(cap_list)
        video_json_l2 = l2_normalize(video_json)
        cap_json_l2 = l2_normalize(cap_json)
        prof_json = {'video_json': video_json, 'video_json_l2': video_json_l2, 'video_sk': video_sk,
                     'cap_json': cap_json, 'cap_json_l2': cap_json_l2, 'cap_sk': cap_sk, 'cap_score': cap_score}
        return prof_json


@app.route('/student', methods=['GET'])
def student():
    if request.method == 'GET':
        video = request.args.get('video', '')
        prof_json = request.args.get('prof_json', '')
        video_sk, video_json = alphapose_vid(video)
        video_json_l2 = l2_normalize(video_json)
        cap_list, capscore_list = cos_sim(
            prof_json['cap_json_l2'], video_json_l2, video, video_json, prof_json['cap_score'])
        dtw_score = dtw_compare(prof_json['video_json_l2'], video_json_l2)
        stu_json = {'capscore_list': capscore_list,
                    'dtw_score': dtw_score, 'cap_list': cap_list, 'video_sk': video_sk}
        return stu_json


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1219)
