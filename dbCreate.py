import os
import requests
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy
import io
import json
import base64

# Tested using Commander's Sphere
# model_ckpt = "openai/clip-vit-base-patch32" # correct set between 3-8
# model_ckpt = "openai/clip-vit-large-patch14-336" # can't identify correct set in top 10
# model_ckpt = "openai/clip-vit-large-patch14" # can't identify correct set in top 10
# model_ckpt = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" # correct set between 4-6
model_ckpt = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" # correct set between 3-5
# model_ckpt = "facebook/metaclip-b32-400m" # can't identify correct set in top 10
image_processor = CLIPProcessor.from_pretrained(model_ckpt)
model = CLIPModel.from_pretrained(model_ckpt)
model.eval()
model.to('cuda')

def addToDb(collection, embedding, cset, collector_number, prices, name):
    
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 512},
    }
    result = collection.search(embedding, "embedding", search_params, limit=1, output_fields=["set"])
    if len(result[0])>0 and result[0][0].distance > 0.99999:
        return
    entity = [
        [cset],
        [collector_number],
        [prices],
        [name],
        [name.lower()],
        [embedding[0]]
    ]
    insert_result = collection.insert(entity)
    print(insert_result)
    

def save(url, name,cset, embeddingId):
    print('\tcomputing embedding')
    embeddingId = embeddingId.replace('/','').partition('?')[0]
    name = clean(name)
    if not os.path.exists(name):
        if not os.path.exists(os.path.dirname(name)):
            os.mkdir(os.path.dirname(name))
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(name, 'wb') as f:
                for chunk in r:
                    f.write(chunk)
        print("\tdownloaded")
    img = cv2.imread(name)
    new_batch = image_processor(text=[''],images=img, return_tensors="pt")
    new_batch.to('cuda')
    output = model(**new_batch)
    embedding = output.image_embeds.cpu().detach().numpy()
    f = io.BytesIO()
    numpy.save(f, embedding, allow_pickle=False)
    f.seek(0)
    blob = {
        'embeddingId':embeddingId,
        'embedding':base64.b64encode(f.read()).decode('ascii')
    }
    jblob = json.dumps(blob)
    embeddingPath = os.path.join('embeddings','s-'+cset+'.jsonl')
    if os.path.exists(embeddingPath):
        f = open(embeddingPath, 'a')
        jblob = '\n'+jblob
    else:
        f = open(embeddingPath, 'w')
    f.write(jblob)
    f.close()
    return embedding.tolist()

def clean(name):
    unapproved = '*★†Φ'
    for c in unapproved:
        name = name.replace(c,'-s')
    return name

def loadEmbedding(cset,embeddingId):
    embeddingId = embeddingId.replace('/','').partition('?')[0]
    embeddingPath = os.path.join('embeddings','s-'+cset+'.jsonl')
    if os.path.exists(embeddingPath):
        lines = open(embeddingPath).read().splitlines()
        for line in lines:
            try:
                print(line[28552])
            except:
                pass
            blob = json.loads(line)
            if blob['embeddingId'] == embeddingId:
                embedding = base64.b64decode(blob['embedding'])
                f = io.BytesIO()
                f.write(embedding)
                f.seek(0)
                embedding = numpy.load(f, allow_pickle=False)
                return embedding.tolist()
    return None

def run(collection):
    collection.load()
    # Imae types described at https://scryfall.com/docs/api/images
    image_type = 'png'
    # Get OneDrive folder
    onedrive = os.environ["OneDrive"]

    # the directory to write cards to.
    cacheDir = os.path.join(onedrive, "Pictures\\cards")

    bulkdata = requests.get('https://api.scryfall.com/bulk-data').json()
    cardsUrl = ''
    for format in bulkdata['data']:
        print(format['name'])
        if format['type'] == 'default_cards':
            cardsUrl = format['download_uri']

    cards = requests.get(cardsUrl).json()

    for i in range(len(cards)):
        card = cards[i]
        print(i,'/',len(cards))
        if card['set'] =='plst':
            continue # these are in something else already
        # if card['image_status'] != 'highres_scan':
        #     continue
        # if card['set'] not in ['mkm','dmu','mom','clb','dbl','mh2','afr','sld']:
        #     continue
        # if card['name'] != "Commander's Sphere":
        #     continue
        print('\t',card['released_at'])
        year = card['released_at'].partition('-')[0]
        if 'image_uris' in card.keys():
            print('\t',card['set'], card['collector_number'], card['name'])
            embedding = loadEmbedding(card['set'],card['image_uris'][image_type].partition(image_type)[2])
            if embedding is None:
                cLoc = os.path.join(cacheDir, year+'-'+card['set'], card['set']+"-"+card['collector_number']+".jpg")
                embedding = save(card['image_uris'][image_type], cLoc,card['set'],card['image_uris'][image_type].partition(image_type)[2])
            addToDb(collection, embedding, card['set'], card['collector_number'], card['prices'], card['name'])
        else:
            if len(card['card_faces']) == 0:
                continue
            parts = 'abcd'
            for i in range(len(card['card_faces'])):
                face = card['card_faces'][i]
                if 'image_uris' not in face.keys():
                    continue
                if 'type_line' in face.keys():
                    t = face['type_line']
                else:
                    t = card['type_line']
                collectorNum = card['collector_number']+parts[i]
                print('\t',card['set'], collectorNum, t, face['name'])
                embedding = loadEmbedding(card['set'],face['image_uris'][image_type].partition(image_type)[2])
                if embedding is None:
                    cLoc = os.path.join(cacheDir, year+'-'+card['set'], card['set']+"-"+collectorNum+".jpg")
                    embedding = save(face['image_uris'][image_type], cLoc,card['set'],face['image_uris'][image_type].partition(image_type)[2])
                addToDb(collection,embedding, card['set'], card['collector_number'], card['prices'], card['name'])
                
def connectDB():
    connections.connect("default", host="localhost", port="19530")
    # utility.drop_collection('mtgCards', using='default')
    if utility.has_collection('mtgCards', using='default'):
        return Collection('mtgCards')
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="set", dtype=DataType.VARCHAR,max_length=10),
        FieldSchema(name="collector_number", dtype=DataType.VARCHAR,max_length=10),
        FieldSchema(name="prices", dtype=DataType.JSON),
        FieldSchema(name="name", dtype=DataType.VARCHAR,max_length=150),
        FieldSchema(name="searchName", dtype=DataType.VARCHAR,max_length=150),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1280)
    ]
    schema = CollectionSchema(fields, "Cards setup for Embedding Search")
    collection = Collection("mtgCards", schema)
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 512},
    }
    collection.create_index("embedding", index)
    return collection

if __name__=='__main__':
    collection = connectDB()
    run(collection)
    collection.flush()  