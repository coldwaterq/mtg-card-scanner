import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import urllib
import ssl
import torch

model_ckpt = "google/vit-base-patch16-224"
image_processor = AutoImageProcessor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
model.eval()

def getImage():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('test')
    while True:
        ret,frame = cam.read()
        if not ret:
            print('failed to grab frame')
        boundImg = findBoundingBox(frame)
        cv2.imshow('test', boundImg)
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

def findBoundingBox(frame):
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
            return(crop_img)
            img1 = cv2.drawContours(img1, contours, i, (255,255,0),2)

    return img1

def computeEmbedding(frame):
    new_batch = image_processor(images=frame, return_tensors="pt")
    embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
    return embeddings

def compareEmbedding(embeding1, embeding2):
    scores = torch.nn.functional.cosine_similarity(embeding1, embeding2, dim=1)
    print(scores)
    print(scores.detach().numpy().tolist())

def testAgainstLnk(lnk, embedding):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.urlopen(lnk, context=ctx)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    srcimg = cv2.imdecode(arr, cv2.IMREAD_COLOR) # 'Load it as it is'
    srcembeding = computeEmbedding(srcimg)
    compareEmbedding(embeding, srcembeding)

if __name__ == '__main__':
    frame = getImage()
    embeding = computeEmbedding(frame)
    testAgainstLnk('https://cards.scryfall.io/large/front/e/a/ea54760c-2cd3-43eb-bc45-adc0997b34b0.jpg?1562442605',embeding)
    testAgainstLnk('https://cards.scryfall.io/large/front/1/8/18f8ac7a-a68a-4adf-995f-1cd96ee3d295.jpg?1562783088',embeding)
    testAgainstLnk('https://cards.scryfall.io/large/front/d/b/dba1cf83-e13d-401e-b76f-b12a51b307f9.jpg?1677149962',embeding)