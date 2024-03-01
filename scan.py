import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel
# from transformers import AutoImageProcessor, AutoModel
import urllib
import ssl
import torch
import json
import os
import string
import csv
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import sys

# model_ckpt = "openai/clip-vit-base-patch32"
# model_ckpt = "openai/clip-vit-large-patch14-336"
# model_ckpt = "openai/clip-vit-large-patch14"
# model_ckpt = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_ckpt = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" 
# model_ckpt = "facebook/metaclip-b32-400m"
image_processor = CLIPProcessor.from_pretrained(model_ckpt)
model = CLIPModel.from_pretrained(model_ckpt)

# model_ckpt = "google/vit-base-patch16-224" all wrong
# image_processor = AutoImageProcessor.from_pretrained(model_ckpt)
# model = AutoModel.from_pretrained(model_ckpt)

model.eval()
model.to('cuda')

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = .5
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2

def save(name, num, prices, foil, csvWriter):
    csvWriter.writerow([num, name, prices, foil])

def openCsv():
    if len(sys.argv) != 2:
        print(sys.argv[0]+' CSVFILE')
        exit()
    name = sys.argv[1]
    if not name.endswith('.csv'):
        name += '.csv'
    onedrive = os.environ["OneDrive"]

    # the directory to write cards to.
    mtgDocDir = os.path.join(onedrive, "Documents\\Real World\\Collections\\mtg")
    name = os.path.join(mtgDocDir, name)

    if os.path.exists(name):
        print('file already exists, appending')
        csvwriter = csv.writer(open(name,'a', newline=''))
    else:
        csvwriter = csv.writer(open(name,'w', newline=''))
    return(csvwriter)


def getImage(collection, csvWriter):
    cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cv2.namedWindow('test')
    count = 0
    previous = ''
    found = False
    # imNum = 0
    sname = ''
    while True:
        ret,frame = cam.read()
        if not ret:
            print('failed to grab frame')
        if found:
            k = cv2.waitKey(1)
            if k%256 == 13:
                print('accepted')
                foil = False
                save(name, num, prices['usd'], foil, csvWriter)
                sname=''
                found=False
            elif k%256 == 27:
                print('reset')
                sname=''
                found=False
            elif k%256 == 9:
                print('accepted foil')
                foil = True
                save(name, num, prices['usd_foil'], foil, csvWriter)
                sname=''
                found=False
            elif chr(k%256) in string.printable:
                # print(k)
                sname+=chr(k%256)
                print(sname)
                found=False
            # if k%256 == 0:
            #     imNum+=1
            #     found=False
            #     if imNum >= len(rets):
            #         print('nope')
            #         imNum = 0
            continue
        img, rets = findBoundingBox(collection, frame, sname)
        if len(rets) == 0:
            sname = ''
            print('invalid name')
            continue
        name, num, prices, score = rets[0]
        boundImg = img.copy()
        if score > .80 or sname != '':
            print(num, score)
            for ret in rets[1:]:
                if ret[0] == name:
                    print('\t',ret[1], ret[3])
            bottomLeftCornerOfText = (5,60)
            cv2.putText(boundImg,name, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
            bottomLeftCornerOfText = (5,100)
            cv2.putText(boundImg,num, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
            bottomLeftCornerOfText = (5,140)
            try:
                cv2.putText(boundImg,'non foil: $'+prices['usd'], 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
            except:
                pass
            bottomLeftCornerOfText = (5,180)
            try:
                cv2.putText(boundImg,'foil: $'+prices['usd_foil'], 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
            except:
                pass
            found=True
        cv2.imshow('test', boundImg)
        
        k = cv2.waitKey(1)
        if k%256 == 27:
            print('leaving')
            break
        if k%256 == 32:
            print('leaving')
            break

    cam.release()
    cv2.destroyAllWindows()
    return frame

def findBoundingBox(collection, frame, name):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,70,255,cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    ### Step #3 - Draw the points as individual circles in the image
    img1 = frame.copy()
    foundContours = []
    for i in range(len(contours)):
        if hierarchy[0][i][3] in foundContours:
            foundContours.append(i)
            continue
        x,y,w,h = cv2.boundingRect(contours[i])
        moments = cv2.moments(contours[i])
        if h/w > 1 and h/w < 3 and moments['m00'] > 10000:
            foundContours.append(i)
            crop_img = frame[y:y+h, x:x+w]
            embeding = computeEmbedding(crop_img)
            ret = compareEmbedding(collection, embeding, name)
            lowerleftCorner = crop_img[8*h//9:h,0:w//3]
            lowerleftCorner = cv2.resize(lowerleftCorner, (w,h//3),interpolation = cv2.INTER_LINEAR)
            shape = lowerleftCorner.shape
            crop_img[h-shape[0]:h,0:shape[1]] = lowerleftCorner
            return crop_img, ret
    return img1,[[None, None, None, -1.0]]

def computeEmbedding(frame):
    new_batch = image_processor(text=[''],images=frame, return_tensors="pt")
    new_batch.to('cuda')
    output = model(**new_batch)
    embeddings = output.image_embeds.cpu().detach()
    return embeddings

def compareEmbedding(collection, embeding, name):
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 512},
    }
    
    sset,_,scnum = name.partition('-')
    result = collection.search(embeding.numpy().tolist(), "embedding", search_params, expr=f'searchName like "{name}%" or (set like "{sset}%" and collector_number like "{scnum}%")',limit=10, output_fields=["id", "set","collector_number","prices","name"])
    ret = []
    for i in range(len(result[0])):
        hit = result[0][i]
        num = f"{hit.entity.get('set')}-{hit.entity.get('collector_number')}"
        name = hit.entity.get('name')
        prices = hit.entity.get('prices')
        ret.append((name, num, prices, hit.distance))
    return ret

def connectDB():
    connections.connect("default", host="localhost", port="19530")

    if utility.has_collection('mtgCards', using='default'):
        return Collection('mtgCards')
    raise(Exception("db doesn't exist"))

if __name__ == '__main__':
    collection = connectDB()
    collection.load()
    csvWriter = openCsv()
    frame = getImage(collection, csvWriter)