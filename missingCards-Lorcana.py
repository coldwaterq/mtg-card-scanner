import csv
import os
import json
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

connections.connect("default", host="localhost", port="19530")
collection = Collection('lorcanaCards')
onedrive = os.environ["OneDrive"]

# the directory to write cards to.
lorcanaDocDir = os.path.join(onedrive, "Documents\\Real World\\Collections\\lorcana")

cards = {'collection':{},'competitive':{}}
for folder,_,fnames in os.walk(lorcanaDocDir):
    if folder.endswith("decks"):
        continue
    for fname in fnames:
        if not fname.endswith('.csv'):
            continue
        name = os.path.join(folder,fname)
        reader = csv.reader(open(name,newline=''))
        rows = []
        for row in reader:
            s,_,cnum = row[0].partition('-')
            name = row[1]
            if s not in cards['collection'].keys():
                res = collection.query(
                    expr=f"set == '{s}'",
                    output_fields=["prices", "collector_number", "name"],
                )
                sDict = {}
                for c in res:
                    
                    p = 10000
                    if len(c['prices']) == 0:
                        p = 0
                    else:
                        for k in c['prices']:
                            v = float(c['prices'][k])
                            p = min(p,v)
                    sDict[c['collector_number']] = {'count':0,'price':p}

                    if c['name'] not in cards['competitive'].keys():
                        cards['competitive'][c['name']] = {'price':p,"locations":{}}
                    elif cards['competitive'][c['name']]['price'] > p:
                        cards['competitive'][c['name']]['price'] = p
                    
                cards['collection'][s]=sDict
            cards['collection'][s][cnum]['count'] +=1
            if fname not in cards['competitive'][name]['locations']:
                cards['competitive'][name]['locations'][fname] = 1
            else:
                cards['competitive'][name]['locations'][fname] +=1

# collectNum = 4
# cost = 0.0
# for s in cards['collection']:
#     stotal = 0
#     for c in cards['collection'][s]:
#         if cards['collection'][s][c]['count'] < collectNum:
#             missingNum = collectNum-cards['collection'][s][c]['count']
#             p = cards['collection'][s][c]['price']*missingNum
#             cost += p
#             # print(s+'-'+c, 'missing',missingNum,'$',cards['collection'][s][c]['price'])#, "total $",p)
#             stotal += missingNum
#     print(s,stotal)
# print('collection cost', cost)


# competitiveNum = 4
# cost = 0.0
# for cname in cards['competitive']:
#     if cards['competitive'][cname]['count'] < competitiveNum:
#         missingNum = competitiveNum-cards['competitive'][cname]['count']
#         p = cards['competitive'][cname]['price']*missingNum
#         cost += p
#         print(cname, 'missing',missingNum,'$',cards['competitive'][cname]['price'], "total $",p)
#     if cards['competitive'][cname]['count'] > competitiveNum and cards['competitive'][cname]['price'] > 2.00:
#         missingNum = cards['competitive'][cname]['count']-competitiveNum
#         p = cards['competitive'][cname]['price']*missingNum
#         cost -= p
#         print(cname, 'extra',missingNum,'$',cards['competitive'][cname]['price'], "total $",p)
# print('competitive cost', cost)

### Decks
# https://lorcanacollectors.com/lorcana-starter-decks/
for folder,_,fnames in os.walk(os.path.join(lorcanaDocDir,'decks')):
    i = 0
    while i < len(fnames):
        name = os.path.join(folder,fnames[i])
        reader = csv.reader(open(name,newline=''))
        row = next(reader)
        if len(row) == 3:
            print(fnames[i],'alread created')
            reader = csv.reader(open(name,newline=''))
            for row in reader:
                cname = row[1]
                locations = json.loads(row[2])
                for loc in locations:
                    cards['competitive'][cname]['locations'][loc]-=locations[loc]
            del(fnames[i])
        else:
            i+=1
        
    for fname in fnames:
        deckTotal = 0
        name = os.path.join(folder,fname)
        reader = csv.reader(open(name,newline=''))
        missing = []
        collected = []
        rows = []
        for row in reader:
            count = int(row[0])
            cname = row[1]
            row = [count,cname]
            totalCount = 0
            for loc in cards['competitive'][cname]['locations']:
                totalCount += cards['competitive'][cname]['locations'][loc]
            deckTotal += min(totalCount,4)
            if count > totalCount:
                missing.append(f'\t{count-totalCount} of {cname}')
            else:
                locations = {}
                for loc in cards['competitive'][cname]['locations']:
                    if count <= cards['competitive'][cname]['locations'][loc]:
                        locations[loc]=count
                        break
                    else:
                        locations[loc]=cards['competitive'][cname]['locations'][loc]
                        count -= cards['competitive'][cname]['locations'][loc]
                collected.append((cname, locations))
                row.append(json.dumps(locations))
            rows.append(row)
        if len(missing) > 0:
            print(fname,deckTotal)
            for line in missing:
                print(line)
        else:
            writer = csv.writer(open(name,'w',newline=''))
            for row in rows:
                writer.writerow(row)
            print(fname,'created, check file for card locations')
            print('quiting because card availability may have changed')
            quit()
        