import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import urllib
import ssl
import torch
import os

model_ckpt = "openai/clip-vit-large-patch14-336"
image_processor = CLIPProcessor.from_pretrained(model_ckpt)
model = CLIPModel.from_pretrained(model_ckpt)
model.eval()

def getImage(referenceEmbeddings, names):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('test')
    while True:
        ret,frame = cam.read()
        if not ret:
            print('failed to grab frame')
        boundImg, name, score = findBoundingBox(frame, referenceEmbeddings, names)
        cv2.imshow('test', boundImg)
        if score > .7:
            print(name)
            break
        k = cv2.waitKey(1)
        if k%256 == 27:
            print('escape hit')
            break
        if k%256 == 32:
            print('space hit')
            break

    cam.release()
    cv2.destroyAllWindows()
    return frame

def findBoundingBox(frame, referenceEmbeddings, names):
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
            embeding = computeEmbedding(frame)
            name, score = compareEmbedding(embeding, referenceEmbeddings,names)
            return(crop_img, name, score)
    return img1, None, 0

def computeEmbedding(frame):
    new_batch = image_processor(text=[''],images=frame, return_tensors="pt")
    output = model(**new_batch)
    embeddings = output.image_embeds
    return embeddings

def compareEmbedding(embeding1, embeding2, names):
    scores = torch.nn.functional.cosine_similarity(embeding1, embeding2, dim=1)
    npScores = scores.detach().numpy()
    i = np.argmax(npScores)
    return(names[i],npScores[i])

def createEmbeddingsFromDir(d):
    images = []
    names = []
    i = 0
    for im in os.listdir(d):
        i+=1
        print(im)
        if i>50:
            break
        names.append(im)
        path = os.path.join(d,im)
        img = cv2.imread(path)
        images.append(img)
    embeddings = computeEmbedding(images)
    return embeddings,names

def testAgainstLnk(lnk, embedding):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.urlopen(lnk, context=ctx)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    srcimg = cv2.imdecode(arr, cv2.IMREAD_COLOR) # 'Load it as it is'
    srcembeding = computeEmbedding(srcimg)
    compareEmbedding(srcembeding, embedding)

srclinks = [
    'https://gatherer.wizards.com/Handlers/Image.ashx?multiverseid=522215&type=card'
]

if __name__ == '__main__':
    referenceEmbeddings,names = createEmbeddingsFromDir(r'C:\Users\coldw\OneDrive\Pictures\cards\2018-rix')
    frame = getImage(referenceEmbeddings, names)
    # https://milvus.io/docs/example_code.md