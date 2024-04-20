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

cardSystem = "mtg"
# cardSystem = "lorcana"

# model_ckpt = "openai/clip-vit-base-patch32"
# model_ckpt = "openai/clip-vit-large-patch14-336"
# model_ckpt = "openai/clip-vit-large-patch14"
# model_ckpt = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_ckpt = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" 
# model_ckpt = "facebook/metaclip-b32-400m"
image_processor = None
model = None


font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = .5
fontColor              = (255,255,255)
fontBorder             = (0,0,0)
lineType               = 2

def loadModel():
    global image_processor, model
    image_processor = CLIPProcessor.from_pretrained(model_ckpt)
    model = CLIPModel.from_pretrained(model_ckpt)
    model.eval()
    model.to('cuda')


def save(name, num, prices, foil, csvWriter):
    try:
        cv2.destroyWindow("important")
    except:
        pass
    if prices == None:
        prices = "0.01"
    if float(prices) > 2:
        img = np.zeros([220, 400, 3])
        img[:,:,2]+=255
        writeText(img,name+" $"+prices+" greater than $2", 
                (10,100))
        cv2.imshow("important", img)
    csvWriter.writerow([num, name, prices, foil])

def openCsv():
    if len(sys.argv) < 3:
        print(sys.argv[0]+' CSV_FILE_NAME DESIRED_CARDS_PER_FILE [set]')
        exit()
    s = ''
    if len(sys.argv) >= 4:
        s = sys.argv[3]
    name = sys.argv[1]
    desriedLines = int(sys.argv[2])
    if not name.endswith('.csv'):
        name += '.csv'
    onedrive = os.environ["OneDrive"]

    # the directory to write cards to.
    docDir = os.path.join(onedrive, "Documents\\Real World\\Collections\\"+cardSystem)
    name = os.path.join(docDir, name)

    if os.path.exists(name):
        with open(name) as f:
            lines = sum(1 for _ in f)
        print('file already exists, appending')
        csvwriter = csv.writer(open(name,'a', newline=''))
    else:
        csvwriter = csv.writer(open(name,'w', newline=''))
        lines = 0
    if lines >= desriedLines:
        print(name,'already has',lines,'entries')
        exit()
    return(csvwriter, lines, desriedLines, s)


def writeText(boundImg,text, 
                bottomLeftCornerOfText):
    thickness=3
    cv2.putText(boundImg,text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontBorder,
        thickness,
        lineType)
    thickness=1
    cv2.putText(boundImg,text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)

def getImage(collection, csvWriter, lines, desiredLines, s):
    boudingScore = 0.5
    cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cv2.namedWindow('test')
    count = 0
    previous = ''
    found = False
    imNum = 0
    sname = ''
    while True:
        ret,frame = cam.read()
        if not ret:
            print('failed to grab frame')
        if found:
            k = cv2.waitKey(2)
            if k%256 == 13:
                lines+=1
                print('accepted',lines)
                foil = False
                try:
                    p = prices['usd']
                except:
                    p = "0.01"
                save(name, num, p, foil, csvWriter)
                if lines >= desiredLines:
                    print('complete')
                    break
                sname=''
                imNum = 0
                found=False
            elif k%256 == 27:
                print('reset')
                sname=''
                imNum = 0
                found=False
            elif k%256 == 9:
                lines+=1
                print('accepted foil',lines)
                foil = True
                try:
                    p = prices['usd_foil']
                except:
                    p = prices['usd']
                save(name, num, p, foil, csvWriter)
                if lines >= desiredLines:
                    print('complete')
                    break
                sname=''
                imNum = 0
                found=False
            elif chr(k%256) in string.printable:
                # print(k)
                sname+=chr(k%256)
                imNum = 0
                print(sname)
                found=False
            if k%256 == 0:
                imNum+=1
                found=False
                if imNum >= len(rets) or rets[imNum][3] <= boudingScore:
                    print('nope')
                    imNum = 0
            continue
        if imNum == 0:
            if s != '' and s[-1]!='-':
                s+='-'
            img, rets = findBoundingBox(collection, frame, s+sname)
        if len(rets) == 0:
            if sname != '':
                sname = ''
                print('invalid name')
            
            cv2.imshow('test', img)
            continue
        name, num, prices, score = rets[imNum]
        boundImg = img.copy()
        if score > boudingScore or sname != '':
            # for ret in rets[1:]:
            #     if ret[0] == name:
            #         print('\t',ret[1], ret[3])
            bottomLeftCornerOfText = (5,60)
            writeText(boundImg,name, 
                bottomLeftCornerOfText)
            for i in range(len(rets)):
                if i!= imNum and rets[i][3] <= boudingScore:
                    break
                x = i%2
                y = i//2
                bottomLeftCornerOfText = (5+x*100,100+y*20)
                writeText(boundImg,rets[i][1], 
                    bottomLeftCornerOfText)
                if i == imNum:
                    bottomLeftCornerOfText = (5+x*100,105+y*20)
                    underline = '_'*len(num)
                    writeText(boundImg,underline, 
                        bottomLeftCornerOfText)
            bottomLeftCornerOfText = (5,140)
            try:
                writeText(boundImg,'non foil: $'+str(prices['usd']), 
                bottomLeftCornerOfText)
            except Exception as e:
                pass
            bottomLeftCornerOfText = (5,160)
            try:
                writeText(boundImg,'foil: $'+str(prices['usd_foil']), 
                bottomLeftCornerOfText)
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
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #         cv2.THRESH_BINARY,3,2)
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
            crop_img = cv2.rotate(crop_img, cv2.ROTATE_180)
            embeding = computeEmbedding(crop_img)
            ret = compareEmbedding(collection, embeding, name)
            # lowerleftCorner = crop_img[8*h//9:h,0:w//3]
            # lowerleftCorner = cv2.resize(lowerleftCorner, (w,h//3),interpolation = cv2.INTER_LINEAR)
            # shape = lowerleftCorner.shape
            # crop_img[h-shape[0]:h,0:shape[1]] = lowerleftCorner
            return crop_img, ret
    return img1,[]

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
    result = collection.search(embeding.numpy().tolist(), "embedding", search_params, expr=f'searchName like "{name}%" or (set like "{sset}%" and collector_number like "{scnum}%")',limit=4, output_fields=["id", "set","collector_number","prices","name"])
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

    if utility.has_collection(cardSystem+'Cards', using='default'):
        return Collection(cardSystem+'Cards')
    raise(Exception("db doesn't exist"))

if __name__ == '__main__':
    collection = connectDB()
    collection.load()
    csvWriter, lines, desiredLines, s = openCsv()
    loadModel()
    frame = getImage(collection, csvWriter, lines, desiredLines, s)