import json
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import numpy as np

def cosineSim(A, embedList):
    cosine = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
    return cosine

def loadConfig():
    try:
        config = json.load(open("config.json"))
    except FileNotFoundError:
        config = {
                "type":"lorcana",
                "encodingModel":"laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
            }
        json.dump(
            config,
            open("config.json",'w')
        )
    return config

def loadModel(config):
    # Tested using Commander's Sphere
    # model_ckpt = "openai/clip-vit-base-patch32" # correct set between 3-8
    # model_ckpt = "openai/clip-vit-large-patch14-336" # can't identify correct set in top 10
    # model_ckpt = "openai/clip-vit-large-patch14" # can't identify correct set in top 10
    # model_ckpt = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" # correct set between 4-6
    # model_ckpt = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" # correct set between 3-5
    # model_ckpt = "facebook/metaclip-b32-400m" # can't identify correct set in top 10
    image_processor = CLIPProcessor.from_pretrained(config["encodingModel"])
    model = CLIPModel.from_pretrained(config["encodingModel"])
    model.eval()
    model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(config["encodingModel"])

    return image_processor, model, tokenizer

def connectDB(config, create=False):
    dbName = config["type"]+"Cards"
    connections.connect("default", host="localhost", port="19530")
    # utility.drop_collection(dbName, using='default')
    if utility.has_collection(dbName, using='default'):
        return Collection(dbName)
    if not create:
        raise(Exception("db doesn't exist"))
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
    collection = Collection(dbName, schema)
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 512},
    }
    collection.create_index("embedding", index)
    return collection
    