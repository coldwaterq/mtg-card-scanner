import os
import requests
import cv2
import numpy
import io
import json
import base64
import numpy as np
import util
import sys

status = ""
def printUpdate(s, num, name, *args):
    global status
    if status != '':
        print(status)
        status = ''
        print('\t',s, num, name)
    print(*args)

# def addToDb(collection, embedding, cset, collector_number, prices, name):
#     results = collection.query(
#         expr = f'name == "{name.replace('"','\\"')}" and set == "{cset}" and collector_number == "{collector_number}"',
#         output_fields = ["embedding","prices"],
#     )
#     # search_params = {
#     #     "metric_type": "COSINE",
#     #     "params": {"nprobe": 512},
#     # }
#     # result = collection.search(embedding, "embedding", search_params, limit=1, output_fields=["prices"])
#     found = False
#     for result in results:
#         if np.allclose(embedding, result["embedding"]):
#             p = result["prices"]
#             if p==prices:
#                 return
#             ret = collection.delete(f"id in [ {result['id']} ]")
#             global status
#             sys.stdout.write(status+" updating prices for "+cset+"-"+collector_number+"     ")
#             found = True
#     if not found:
#         printUpdate(cset, collector_number, name,"\tinserting")
#     entity = [
#         [cset],
#         [collector_number],
#         [prices],
#         [name],
#         [name.lower()],
#         [embedding[0]]
#     ]
#     insert_result = collection.insert(entity)


def save(url, name,cname,csname,cnum, cset,crarity,cprodId, embeddingId, model):
    printUpdate(cset, cnum, cname,'\tcomputing embedding')
    embeddingId = embeddingId.replace('/','').replace('?','-')
    name = clean(name)
    if not os.path.exists(os.path.dirname(name)):
        os.mkdir(os.path.dirname(name))
    if not os.path.exists(name):
        # os.remove(name)
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            if config["type"]=="lorcana":
                with open('temp.avif', 'wb') as f:
                    for chunk in r:
                        f.write(chunk)
                print("\tdownloaded")
                from PIL import Image,UnidentifiedImageError
                import pillow_avif

                try:
                    img = Image.open('temp.avif')
                    img.save(name)
                except UnidentifiedImageError:
                    return None
                finally:
                    os.remove('temp.avif')
                print ("\tconverted")
            else:
                with open(name, 'wb') as f:
                    for chunk in r:
                        f.write(chunk)
                print("\tdownloaded")
        else:
            print(r.status_code)
            print(r.content)
            return None
    img = cv2.imread(name)
    new_batch = model[0](text=[''],images=img, return_tensors="pt")
    new_batch.to('cuda')
    output = model[1](**new_batch)
    embedding = output.image_embeds.cpu().detach().numpy()
    f = io.BytesIO()
    numpy.save(f, embedding, allow_pickle=False)
    f.seek(0)
    blob = {
        'name':cname,
        'set':csname,
        'num':cnum,
        'setCode':cset,
        'rarity':crarity,
        'prodId':cprodId,
        'embeddingId':embeddingId,
        'embedding':base64.b64encode(f.read()).decode('ascii'),
    }
    setPart = img[int(img.shape[0]/6*5):img.shape[0],:]
    new_batch = model[0](text=[''],images=setPart, return_tensors="pt")
    new_batch.to('cuda')
    output = model[1](**new_batch)
    embedding = output.image_embeds.cpu().detach().numpy()
    f = io.BytesIO()
    numpy.save(f, embedding, allow_pickle=False)
    f.seek(0)
    blob['partialEmbedding'] = base64.b64encode(f.read()).decode('ascii')
    jblob = json.dumps(blob)
    embeddingPath = os.path.join('embeddings',config["type"],'s-'+cset+'.jsonl')
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
    embeddingId = embeddingId.replace('/','').replace('?','-')
    embeddingPath = os.path.join('embeddings',config["type"],'s-'+cset+'.jsonl')
    if os.path.exists(embeddingPath):
        lines = open(embeddingPath).read().splitlines()
        for line in lines:
            blob = json.loads(line)
            if blob['embeddingId'] == embeddingId:
                embedding = base64.b64decode(blob['embedding'])
                f = io.BytesIO()
                f.write(embedding)
                f.seek(0)
                embedding = numpy.load(f, allow_pickle=False)
                return embedding.tolist()
    return None

def runMtg(collection, config, model):
    # collection.load()
    # Imae types described at https://scryfall.com/docs/api/images
    image_type = 'png'
    # Get OneDrive folder

    # the directory to write cards to.
    cacheDir = "G:\\My Drive\\cards\\mtg"

    bulkdata = requests.get('https://api.scryfall.com/bulk-data').json()
    cardsUrl = ''
    for format in bulkdata['data']:
        print(format['name'])
        if format['type'] == 'default_cards':
            cardsUrl = format['download_uri']

    cards = requests.get(cardsUrl).json()

    for i in range(len(cards)):
        card = cards[i]
        if 'tcgplayer_id' not in card.keys():
            card['tcgplayer_id'] = ''
        global status
        status = '\r'+str(i)+'/'+str(len(cards))
        if i%10==0:
            sys.stdout.write(status)
        if card['set'] =='plst':
            continue # these are in something else already
        # if card['image_status'] != 'highres_scan':
        #     continue
        # if card['set'] not in ['m14','10e','avr']:
        #     continue
        # if card['name'] != "Commander's Sphere":
        #     continue
        year = card['released_at'].partition('-')[0]
        if 'image_uris' in card.keys():
            
            embeddingId = card['image_uris'][image_type].partition(image_type)[2]
            embedding = loadEmbedding(card['set'],embeddingId)
            if embedding is None:
                cLoc = os.path.join(cacheDir, year+'-'+card['set'], card['set']+"-"+card['collector_number']+".jpg")
                try:
                    embedding = save(
                        card['image_uris'][image_type], 
                        cLoc,
                        card['name'],
                        card['set_name'],
                        card['collector_number'],
                        card['set'],
                        card['rarity'],
                        card['tcgplayer_id'],
                        embeddingId, 
                        model
                    )
                except KeyError as e:
                    print(card)
                    raise e
            # addToDb(collection, embedding, card['set'], card['collector_number'], card['prices'], card['name'])
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
                embeddingId = face['image_uris'][image_type].partition(image_type)[2]
                embedding = loadEmbedding(card['set'],embeddingId)
                if embedding is None:
                    cLoc = os.path.join(cacheDir, year+'-'+card['set'], card['set']+"-"+collectorNum+".jpg")
                    embedding = save(
                        face['image_uris'][image_type], 
                        cLoc,
                        card['name'],
                        card['set_name'],
                        card['collector_number'],
                        card['set'],
                        card['rarity'],
                        card['tcgplayer_id'],
                        embeddingId, 
                        model
                    )
                # addToDb(collection,embedding, card['set'], card['collector_number'], card['prices'], card['name'])

def runLorcana(collection, config, model):
    collection.load()
    image_type = 'large'
    # Get OneDrive folder
    onedrive = os.environ["OneDrive"]

    # the directory to write cards to.
    cacheDir = os.path.join(onedrive, "Pictures\\cards\\lorcana")

    sets = requests.get('https://api.lorcast.com/v0/sets').json()
    for s in sets['results']:
        cardsUrl = 'https://api.lorcast.com/v0/sets/'+s['code']+'/cards'
        cards = requests.get(cardsUrl).json()
        # cards = cards['results']

        for i in range(len(cards)):
            global status
            
            card = cards[i]
            status = '\r'+str(i)+'/'+str(len(cards))
            if i %10:
                sys.stdout.write(status)
            
            year = card['released_at'].partition('-')[0]
            if 'image_uris' in card.keys():
                embeddingId = card['image_uris']['digital'][image_type].partition(image_type)[2]
                embedding = loadEmbedding(s['code'],embeddingId)
                name = card['name']
                if embedding is None:
                    cLoc = os.path.join(cacheDir, year+'-'+s['code'], s['code']+"-"+card['collector_number']+".jpg")
                    embedding = save(
                        card['image_uris']['digital'][image_type], 
                        cLoc,
                        s['code'],
                        card['collector_number'],
                        name,
                        embeddingId, 
                        model
                    )
                if embedding is None:
                    printUpdate(s['code'], card['collector_number'], name, '\tno good image found')
                    continue
                if card['version'] is not None:
                    name += ' - '+card['version']
                addToDb(collection, embedding, s['code'], card['collector_number'], card['prices'], name)
            else:
                raise Exception('multi-faced cards not working in lorcana yet')

if __name__=='__main__':
    config = util.loadConfig()
    model = util.loadModel(config)
    # collection = util.connectDB(config, create=True)
    collection = None
    if config["type"] == "lorcana":
        runLorcana(collection, config, model)
    elif config["type"] in ["mtg","mtg-test"]:
        runMtg(collection, config, model)
    else:
        print("invalid config type")
    # collection.flush()  