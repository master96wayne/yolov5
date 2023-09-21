import pandas as pd
import cv2
import os
from imutils import build_montages
import tqdm
from utils.general import non_max_suppression
import numpy as np

def bestByImg(df):
    df['conf'] = df.obj_conf * df.cls_conf

    # Sort the DataFrame by 'conf' column in descending order
    # Use groupby to get the rows with the maximum 'conf' value for each unique 'img'
    result = (df[df.cls_conf!=1]
        .sort_values(by='conf', ascending=False)
        .groupby('img').first().reset_index())

    return result

# def truePositives(df):


def buildMontage(df, imgRoot, show=False):
    images = []

    for imgPath in tqdm.tqdm(df.img):
        try:
            # print(imgPath)
            image = cv2.imread(os.path.join(imgRoot, imgPath.replace('txt', 'jpg')))
            # print(image)
            imgRow = df[df['img'] == imgPath]
            xmin = int(imgRow.xmin)
            ymin = int(imgRow.ymin)
            xmax = int(imgRow.xmax)
            ymax = int(imgRow.ymax)
            image = cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
            images.append(image)
        except:
            image = cv2.imread(os.path.join(imgRoot, imgPath.replace('txt', 'jpg')))
            # print(image)
            imgRow = df[df['img'] == imgPath]
            imgRows = imgRow.values.tolist()
            for imgRow in imgRows:
                # print(imgRow)
                xmin = int(imgRow[4])
                xmax = int(imgRow[5])
                ymin = int(imgRow[6])
                ymax = int(imgRow[7])
                image = cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
            images.append(image)
            # pass

    montages = build_montages(images, (640//3, 640//4), (5,5))

    if show:
        showMontages(montages)

    return montages

def showMontages(montages):
    for montage in montages:
        cv2.imshow('montage', montage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def box_iou(box1,box2):
    inter_w = abs((min(box1[2], box2[2]) - max(box1[0], box2[0])))
    inter_h = abs((min(box1[3], box2[3]) - max(box1[1], box2[1])))

    inter = inter_w*inter_h

    a1 = abs((box1[2]-box1[0])*(box1[3]-box1[1]))
    a2 = abs((box2[2]-box2[0])*(box2[3]-box2[1]))
    union = a1 + a2 - inter

    return inter/union if inter/union>0 else 0

bg_hydrid_csv = 'bg_df_hard_hybrid_sep-conf.csv'
fg_hydrid_csv = 'fg_df_hard_hybrid_sep-conf.csv'
hard3_csv = 'df_hard3.csv'

fg_hybrid_df = pd.read_csv(fg_hydrid_csv)
fg_hybrid_df['conf'] = fg_hybrid_df.cls_conf * fg_hybrid_df.obj_conf
label_df = fg_hybrid_df[fg_hybrid_df.conf == 1].reset_index()
# print(fg_hybrid)

bg_hydrid_df = pd.read_csv(bg_hydrid_csv)

pred_df = pd.read_csv(hard3_csv)
# print(pred_df)

tp_df = pred_df.merge(label_df, how='inner', on=['img', 'class'], suffixes=['_pred', '_label'])
tp_df['conf'] = tp_df.cls_conf_pred * tp_df.obj_conf_pred
# print(tp_df.sort_values(by='conf', ascending=True))
fp_df = pred_df.merge(bg_hydrid_df, how='inner', on=['img', 'class'], suffixes=['_pred', '_label'])

fp_df['conf'] = fp_df.cls_conf_pred * fp_df.obj_conf_pred
# print(fp_df[fp_df.conf>0.5].reset_index())
# fp_df = fp_df[fp_df.conf>0.2].reset_index()

imgRoot = "C:\\Users\\leofi\OneDrive - Universidade de Lisboa\Documents\GitHub\masters\Data\\fc2015_yolo_uc_hard/test/images"

# label_df_s = bestByImg(label_df)

conf_th = tp_df.conf.mean()
# tp_df = tp_df[tp_df.conf<conf_th*0.01]
# buildMontage(tp_df, imgRoot=imgRoot, show=True)
# buildMontage(fp_df, imgRoot=imgRoot, show=True)

iou_th = 0.25
# print(tp_df.shape)
for img in tp_df.img.unique():
    tp_sub_df = tp_df[tp_df.img==img]

    for idx, row in tp_sub_df.iterrows():
        box_pred = [row.xmin_pred, row.ymin_pred, row.xmax_pred, row.ymax_pred]
        box_label = [row.xmin_label, row.ymin_label, row.xmax_label, row.ymax_label]

        # print(box_pred)
        # print(box_iou(box_pred, box_label))
        iou = box_iou(box1=box_pred, box2=box_label)
        if iou < iou_th:
            tp_df.drop(idx, inplace=True)

# print(tp_df.shape)
buildMontage(tp_df, imgRoot=imgRoot, show=True)