import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dtaidistance import dtw


def l2_normalize(jsonfile):

    with open(jsonfile) as kps:
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

    with open(jsonfile.replace('.json', '_l2norm.json'), 'w') as f:
        json.dump(json_kps, f)
        print('Write l2_norm keypoints')


def weightmatch(label_json, input_json, label_img, input_video):

    with open(label_json) as f:
        label = json.load(f)[0]['keypoints']

    with open(input_json) as f:
        ip_data = json.load(f)

    high_score = 0
    highlight = ''
    score_list = []

    for frame in range(len(ip_data)):
        ip_kpt = ip_data[frame]['keypoints']
        summation_1 = 0
        summation_2 = 0

        for _ in range(17):
            x = np.abs(ip_kpt[_ * 3] - label[_ * 3])
            y = np.abs(ip_kpt[_ * 3 + 1] - label[_ * 3 + 1])
            temp = (2 - (x + y))*50
            summation_1 += ip_kpt[_ * 3 + 2]
            summation_2 += ip_kpt[_ * 3 + 2] * temp
            score_list.append(temp)

        score = summation_2 / summation_1

        if high_score <= score:
            high_score = score
            highlight = ip_data[frame]['image_id']
            score_detail = score_list

    ip_img = cv2.imread(label_img)
    ip_img = cv2.cvtColor(ip_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(18, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(ip_img)
    plt.title('label img')

    cap = cv2.VideoCapture(input_video)
    fr_iter = 0

    while(cap.isOpened()):
        try:
            ret, frame = cap.read()
            if fr_iter == int(highlight.split('.')[0]):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                plt.subplot(1, 2, 2)
                plt.imshow(frame)
                plt.title('capture img')
                plt.show()
                break
            fr_iter += 1
        except:
            continue

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

    for (k, v) in score_json.items():
        print(k, ' : ', v, '\n')

    return score_json


def l2_weightmatch(label_json, input_json, label_img, input_video):

    with open(label_json) as f:
        label = json.load(f)[0]['keypoints']

    with open(input_json) as f:
        ip_data = json.load(f)

    high_score = 0
    highlight = ''
    score_list = []

    for frame in range(len(ip_data)):
        ip_kpt = ip_data[frame]['keypoints']
        summation_1 = 0
        summation_2 = 0

        for _ in range(17):
            x = (ip_kpt[_ * 3] - label[_ * 3])**2
            y = (ip_kpt[_ * 3 + 1] - label[_ * 3 + 1])**2
            temp = (np.sqrt(2) - np.sqrt(x + y))*(100/np.sqrt(2))
            summation_1 += ip_kpt[_ * 3 + 2]
            summation_2 += ip_kpt[_ * 3 + 2] * temp
            score_list.append(temp)

        score = summation_2 / summation_1

        if high_score <= score:
            high_score = score
            highlight = ip_data[frame]['image_id']
            score_detail = score_list

    ip_img = cv2.imread(label_img)
    ip_img = cv2.cvtColor(ip_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(18, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(ip_img)
    plt.title('label img')

    cap = cv2.VideoCapture(input_video)
    fr_iter = 0

    while(cap.isOpened()):
        try:
            ret, frame = cap.read()
            if fr_iter == int(highlight.split('.')[0]):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                plt.subplot(1, 2, 2)
                plt.imshow(frame)
                plt.title('capture img')
                plt.show()
                break
            fr_iter += 1
        except:
            continue

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

    for (k, v) in score_json.items():
        print(k, ' : ', v, '\n')

    return score_json


def cos_sim(label_json, input_json, label_img, input_video):

    with open(label_json) as f:
        label = json.load(f)[0]['keypoints']

    with open(input_json) as f:
        ip_data = json.load(f)

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
            score_detail = score_list

    ip_img = cv2.imread(label_img)
    ip_img = cv2.cvtColor(ip_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(18, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(ip_img)
    plt.title('label img')

    cap = cv2.VideoCapture(input_video)
    fr_iter = 0

    while(cap.isOpened()):
        try:
            ret, frame = cap.read()
            if fr_iter == int(highlight.split('.')[0]):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                plt.subplot(1, 2, 2)
                plt.imshow(frame)
                plt.title('capture img')
                plt.show()
                break
            fr_iter += 1
        except:
            continue

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

    for (k, v) in score_json.items():
        print(k, ' : ', v, '\n')

    return score_json


def dtw_compare(label_json, input_json):

    with open(label_json) as f:
        label = json.load(f)

    with open(input_json) as f:
        ip_data = json.load(f)

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

    for (k, v) in score_json.items():
        print(k, ' : ', v, '\n')

    return score_json
