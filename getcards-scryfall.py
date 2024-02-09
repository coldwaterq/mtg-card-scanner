import os
import requests

def save(url, name):
    name = clean(name)
    if os.path.exists(name):
        return
    if not os.path.exists(os.path.dirname(name)):
        os.mkdir(os.path.dirname(name))
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(name, 'wb') as f:
            for chunk in r:
                f.write(chunk)
    print("\tdownloaded")

def clean(name):
    unapproved = '*★'
    for c in unapproved:
        name = name.replace(c,'')
    return name

def run():
    # Imae types described at https://scryfall.com/docs/api/images
    image_type = 'border_crop'
    # Get OneDrive folder
    onedrive = os.environ["OneDrive"]
    print(onedrive)

    # the directory to write cards to.
    cardDir = os.path.join(onedrive, "Pictures\\cards")

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
        if card['image_status'] != 'highres_scan':
            continue
        print('\t',card['released_at'])
        year = card['released_at'].partition('-')[0]
        if 'image_uris' in card.keys():
            print('\t',card['set'], card['collector_number'], card['type_line'], card['name'])
            cLoc = os.path.join(cardDir, year+'-'+card['set'], card['set']+"-"+card['collector_number']+".jpg")
            save(card['image_uris'][image_type], cLoc)
        else:
            if len(card['card_faces']) == 0:
                print('no faces')
                exit()
            parts = 'abcd'
            for i in range(len(card['card_faces'])):
                face = card['card_faces'][i]
                if 'type_line' in face.keys():
                    t = face['type_line']
                else:
                    t = card['type_line']
                collectorNum = card['collector_number']+parts[i]
                print('\t',card['set'], collectorNum, t, face['name'])
                cLoc = os.path.join(cardDir, year+'-'+card['set'], card['set']+"-"+collectorNum+".jpg")
                save(face['image_uris'][image_type], cLoc)
                


if __name__=='__main__':
    run()