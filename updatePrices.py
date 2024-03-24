import csv
import os
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

connections.connect("default", host="localhost", port="19530")
collection = Collection('mtgCards')
onedrive = os.environ["OneDrive"]

# the directory to write cards to.
mtgDocDir = os.path.join(onedrive, "Documents\\Real World\\Collections\\mtg")

for folder,_,fnames in os.walk(mtgDocDir):
    for fname in fnames:
        if not fname.endswith('.csv'):
            continue
        oldTotal = 0
        newTotal = 0
        name = os.path.join(folder,fname)
        reader = csv.reader(open(name,newline=''))
        rows = []
        for row in reader:
            foil = row[3].lower()=='true'
            s,_,cnum = row[0].partition('-')
            res = collection.query(
                expr=f"set == '{s}' and collector_number == '{cnum}'",
                output_fields=["prices"],
            )
            prices = res[0]['prices']
            price = 0
            if foil:
                price = prices['usd_foil']
            else:
                price = prices['usd']
            try:
                oldTotal += int(100*float(row[2]))
            except:
                pass
            try:
                newTotal += int(100*float(price))
            except:
                pass
            row[2]=price
            rows.append(row)
        writer = csv.writer(open(name,'w',newline=''))
        for row in rows:
            writer.writerow(row)
        print(fname,oldTotal/100,newTotal/100)