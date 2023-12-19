from pandas import read_csv
import torch
from torch import nn
from tqdm import tqdm

TrainSet = read_csv("./data/train.csv")
# TestSet = read_csv("./data/test.csv")

user_id = 0
handled = 0  # The number of record which have been embedded
truncation = 20

# Embedding
# ProductList = [] # The product list mapped ith product to its id
# ProdDict = {} #The product dictionary mapped a product's id to i
# for index,row in TrainSet.iterrows():
#     if not row['product_id'] in ProductList:
#         ProdDict[row['product_id']] = len(ProductList)
#         ProductList.append(row['product_id'])

# print(len(ProductList))

# embedding = nn.Embedding(len(ProductList), 508)
maxlen = 0
maxid=0
print(len(TrainSet))
with tqdm(total=len(TrainSet)) as pbar:
    while handled < len(TrainSet):
        user_id += 1
        PartSet = TrainSet[TrainSet["user_id"] == user_id]
        SequenceLen = len(PartSet)
        handled += SequenceLen
        if SequenceLen>maxlen:
            maxlen = SequenceLen
            maxid = user_id
        pbar.update(handled)
print(maxlen, maxid)