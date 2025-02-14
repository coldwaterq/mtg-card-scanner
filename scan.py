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
import time
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import base64
import io
import sys
from sklearn.metrics.pairwise import cosine_similarity
import util

cos = torch.nn.CosineSimilarity(dim=1)


font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = .5
fontColor              = (255,255,255)
fontBorder             = (0,0,0)
lineType               = 2


def save(name, num, setName, setCode, prodId, prices, foil, csvWriter):
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
    printing = "Normal"
    if foil:
        printing = "Foil"
    csvWriter.writerow([1, name, setName, num, setCode, printing, "Near Mint","English", prodId])

def openCsv(config):
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
    

    # the directory to write cards to.
    docDir = os.path.join("G:\\My Drive\\", "Documents\\Real World\\Collections\\"+config['type'])
    name = os.path.join(docDir, name)

    if os.path.exists(name):
        with open(name) as f:
            lines = sum(1 for _ in f)
        print('file already exists, appending')
        csvwriter = csv.writer(open(name,'a', newline=''))
    else:
        csvwriter = csv.writer(open(name,'w', newline=''))
        csvwriter.writerow(['Quantity','Name','Set','Card Number','Set Code','Printing','Condition','Language','Product ID'])
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

def getImage(collection, csvWriter, lines, desiredLines, s, config, model):
    cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cv2.namedWindow('test')
    count = 0
    previous = ''
    bwThresh = 25
    found = False
    imNum = 0
    sname = ''
    rets = []
    while True:
        ret,frame = cam.read()
        if not ret:
            print('failed to grab frame')
        if found:
            k = cv2.waitKey(2)
            if k%256 == 13:
                lines+=len(rets)
                print('accepted',lines)
                foil = False
                prices = None
                for ret in rets:
                    save(ret['name'], ret['num'], ret['set'], ret['setCode'], ret['prodId'], prices, foil, csvWriter)
                if lines >= desiredLines:
                    print('complete')
                    break
                sname=''
                found=False
            elif k%256 == 27:
                print('reset')
                sname=''
                imNum = 0
                found=False
            elif k%256 == 9:
                lines+=len(rets)
                print('accepted foil',lines)
                foil = True
                prices = None
                for ret in rets:
                    save(ret['name'], ret['num'], ret['set'], ret['setCode'], ret['prodId'], prices, foil, csvWriter)
                if lines >= desiredLines:
                    print('complete')
                    break
                sname=''
                found=False
            elif chr(k%256) in string.printable:
                # print(k)
                sname+=chr(k%256)
                imNum = 0
                print(sname)
                found=False
            continue
        img, rets = findBoundingBox(collection, frame, s+sname, model,bwThresh)
        
        boundImg = img.copy()
        for i in range(len(rets)):
            ret = rets[i]
            bottomLeftCornerOfText = (5,60+i*80)
            writeText(boundImg,ret['name'], 
                bottomLeftCornerOfText)
            
        
            bottomLeftCornerOfText = (5,80+i*80)
            writeText(boundImg,ret['setCode']+'-'+str(ret['num']), 
                bottomLeftCornerOfText)
            
            bottomLeftCornerOfText = (5,100+i*80)
            writeText(boundImg,str(ret['cos']), 
                bottomLeftCornerOfText)
                
            # bottomLeftCornerOfText = (5,140)
            # try:
            #     writeText(boundImg,'non foil: $'+str(prices['usd']), 
            #     bottomLeftCornerOfText)
            # except Exception as e:
            #     pass
            # bottomLeftCornerOfText = (5,160)
            # try:
            #     writeText(boundImg,'foil: $'+str(prices['usd_foil']), 
            #     bottomLeftCornerOfText)
            # except:
            #     pass
            found=True
        cv2.imshow('test', boundImg)
        
        k = cv2.waitKey(1)
        if k%256 == 27:
            print('leaving')
            break
        if k%256 == 32:
            print('leaving')
            break
        if found!=True:#k%256 == 0:
            print(bwThresh)
            bwThresh+=25
            bwThresh%=250
            print(bwThresh)

    cam.release()
    cv2.destroyAllWindows()
    return frame

def findBoundingBox(collection, frame, setPNum, model, bwThresh):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ret,thresh = cv2.threshold(gray,bwThresh,255,cv2.THRESH_BINARY)
    cv2.imshow("bw", thresh)
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #         cv2.THRESH_BINARY,3,2)
    # contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # foundContours = []
    rets = []
    embeddings = []
    boxs = []
    usedContours = []
    for i in range(len(contours)):
        
        if hierarchy[0][i][3] in usedContours: # parent is already used
            usedContours.append(i)
            continue
        rotatedRect = cv2.minAreaRect(contours[i])
        if rotatedRect[2] > 45:
            rotation = rotatedRect[2]-90
            h,w = rotatedRect[1]
        else:
            rotation = rotatedRect[2] 
            w,h = rotatedRect[1]
        if h <100 or w <100:
            continue
        heightWidthRatio = h/w
        moments = cv2.moments(contours[i])
        if heightWidthRatio > 1.1 and heightWidthRatio < 1.5 and  moments['m00'] > 10000:
            usedContours.append(i)
            rot_mat = cv2.getRotationMatrix2D(rotatedRect[0], rotation, 1.0)
            img1 = cv2.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv2.INTER_LINEAR)
            w = int(w)
            h = int(h)
            x = int(rotatedRect[0][0]-w//2)
            y = int(rotatedRect[0][1]-h//2)
            if y < 0 or x < 0 or h < 0 or w < 0:
                print(y,x,h,w)
                continue
            # x,y,w,h = cv2.boundingRect(countour)
            crop_img = img1[y:y+h, x:x+w]
            crop_img = cv2.resize(crop_img, (745,1040), interpolation = cv2.INTER_AREA)
            # cv2.imshow('card',crop_img)
            crop_img = cv2.rotate(crop_img, cv2.ROTATE_180)
            embeding = computeEmbedding(crop_img, model)[0]
            setPart = crop_img[int(crop_img.shape[0]/6*5):crop_img.shape[0],:]
            # cv2.imshow('cards',setPart)
            setEmbeding = computeEmbedding(setPart, model)[0]
            embeddings.append((embeding,setEmbeding))
            box = cv2.boxPoints(rotatedRect) 
            box = np.intp(box)
            boxs.append(box)
    if len(embeddings) != 0:
        rets, boxs = compareEmbedding(collection, embeddings,boxs,setPNum)
        for box in boxs:
            frame = cv2.drawContours(frame,[box],0,(0,0,255),2)
    return frame,rets

def computeEmbedding(frame, model):
    new_batch = model[0](text=[''],images=frame, return_tensors="pt")
    new_batch.to('cuda')
    output = model[1](**new_batch)
    embeddings = output.image_embeds.cpu().detach()
    return embeddings

def compareEmbedding(collection, embeding,boxs,setPNum):
    maxCos = 0.0
    url = ''
    minBlob = {}
    source = ''
    embeddings = collection['embeddings']
    partialEmbeddings = collection['partialEmbeddings']
    t = time.time()
    rets = []
    nboxs = []
    prodIds = []
    for j in range(len(embeding)):
        similarity = cos(embeding[j][0].to('cuda'), embeddings).cpu()
        
        # setSimilarity = cos(embeding[j][1].to('cuda'), partialEmbeddings).cpu()
        i = np.argmax(similarity)
        blob = collection['blobs'][i]
        maxCos = similarity[i]
        minSetCos = 0
   
        if maxCos < 0.7:
            continue
        
        print(maxCos,blob['set'])
        tmaxCos = maxCos
        blob = collection['blobs'][i]    
        while not (blob['setCode']+'-'+blob['num']).startswith(setPNum) and similarity[i]>0.7:
            print(similarity[i],collection['blobs'][i]['setCode'],collection['blobs'][i]["name"],blob['num'])
            similarity[i] = 0
            # if minSetCos < .4 or (minSetCos < setSimilarity[i] and collection['blobs'][i]["name"]==blob['name']) or :
            blob = collection['blobs'][i]
                # minSetCos = setSimilarity[i]
            tmaxCos = maxCos
            i = np.argmax(similarity)
            maxCos = similarity[i]

        print()
            
        maxCos = tmaxCos 
        
        # if blob['prodId'] in prodIds:
        #     continue
        prodIds.append(blob['prodId'])
        minBlob = blob
        print(time.time()-t)
        minBlob['cos']=maxCos
        rets.append(minBlob)
        nboxs.append(boxs[j])
    return rets, nboxs
    return []
    result = collection.search(embeding.numpy().tolist(), "embedding", search_params, expr=f'searchName like "{name}%" or (set like "{sset}%" and collector_number like "{scnum}%")',limit=4, output_fields=["id", "set","collector_number","prices","name"])
    ret = []
    for i in range(len(result[0])):
        hit = result[0][i]
        num = f"{hit.entity.get('set')}-{hit.entity.get('collector_number')}"
        name = hit.entity.get('name')
        prices = hit.entity.get('prices')
        ret.append((name, num, prices, hit.distance))
    return ret

def loadEmbeddings(config):
    collection = {'partialEmbeddings':[],'embeddings':[],'blobs':[]}
    count = 0
    for fname in os.listdir(os.path.join('embeddings',config["type"])):
        print(fname)
        embeddingPath = os.path.join('embeddings',config["type"],fname)
        content = open(embeddingPath).read()
        for line in content.split('\n'):
            blob = json.loads(line)
            embedding = blob['embedding']
            embedding = base64.b64decode(embedding)
            f = io.BytesIO()
            f.write(embedding)
            f.seek(0)
            embedding = np.load(f, allow_pickle=False).tolist()
            if len(embedding) != 1:
                count +=1
                embedding = [embedding]
            collection['embeddings'].append(embedding[0])

            partialEmbedding = blob['partialEmbedding']
            partialEmbedding = base64.b64decode(partialEmbedding)
            f = io.BytesIO()
            f.write(partialEmbedding)
            f.seek(0)
            partialEmbedding = np.load(f, allow_pickle=False).tolist()
            if len(partialEmbedding) != 1:
                count +=1
                partialEmbedding = [partialEmbedding]
            collection['partialEmbeddings'].append(partialEmbedding[0])
            collection['blobs'].append(blob)
    collection['embeddings'] = torch.tensor(collection['embeddings']).to('cuda')
    collection['partialEmbeddings'] = torch.tensor(collection['partialEmbeddings']).to('cuda')
    print(count,'are saved wrong')
    return collection

if __name__ == '__main__':
    config = util.loadConfig()
    csvWriter, lines, desiredLines, s = openCsv(config)
    model = util.loadModel(config)
    collection = loadEmbeddings(config)
    
    frame = getImage(collection, csvWriter, lines, desiredLines, s, config, model)