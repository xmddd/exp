from pandas import read_csv
from tqdm import tqdm

TestSet = read_csv(
    "./data/sss.csv",
    usecols=["event_type", "product_id", "user_id"],
    dtype={"event_type": str, "product_id": str, "user_id": str},
)
# dict={'0':'blue','1':'green'}
# list =[0,0,1,1,0]
# print(dict[list])
users = -1  # The number of users in TrainSet after truncating
RecordList = {"Prod": [], "Event": []}  # The truncated records of user
ProdList = ["000"]  # The product list mapped ith product to its id
ProdDict = {
    "000": 0
}  # The product dictionary mapped a product's id to i, the '000' represents the padding product
ProdIndex = []  # The product index in ProdList
for index, row in TestSet.iterrows():
    if not row["product_id"] in ProdList:
        ProdDict[row["product_id"]] = len(ProdList)
        ProdList.append(row["product_id"])
    ProdIndex.append(ProdDict[row["product_id"]])
    # Event.append(EventDict[row["event_type"]])
TestSet = TestSet.groupby("user_id")
print(len(TestSet))
truncation = 3
users = 0
RecordList = []
for user_id, PartSet in tqdm(TestSet):
    i = 0
    # print(user_id, PartSet)
    # print(PartSet[:,'product_id'])
    # print(PartSet.iloc[:]["product_id"])
    # length = len(PartSet)
    while len(PartSet) > i + truncation:
        users += 1
        i += truncation
        ProdList = PartSet.iloc[i : i + truncation]["product_id"].tolist()
        ProdIndexList = [ProdDict[product_id] for product_id in ProdList]
        print(ProdIndexList)
    users += 1
    print("i=", i, "\tlength=", len(PartSet))
    ProdList = PartSet.iloc[i : len(PartSet)]["product_id"].tolist()
    for i in range(truncation - len(ProdList)):
        ProdList.append("000")
    ProdIndexList = [ProdDict[product_id] for product_id in ProdList]
    print(ProdIndexList)
#     RecordList.append(PartSet.loc[i : i + truncation, ["product_id", "event_type"]])
#     for i in range(truncation - 1 - (length - 1) % truncation):
#         PartSet.loc[len(PartSet)] = {
#             "user_id": "0",
#             "product_id": "0",
#             "event_type": "padding",
#         }
#     print(PartSet)
#     print('length=',len(PartSet))
#     for i in range(int(len(PartSet) / truncation)):
#         users += 1
#         print(PartSet.loc[i : i + truncation, ["product_id", "event_type"]])
# print(RecordList)
# ProductList = [] # The product list mapped ith product to its id
# ProdDict = {} #The product dictionary mapped a product's id to i
# embedding_value = []
# for index,row in TestSet.iterrows():
#     if not row['product_id'] in ProductList:
#         ProdDict[row['product_id']] = len(ProductList)
#         ProductList.append(row['product_id'])
#     embedding_value.append(ProdDict[row['product_id']])

# TestSet.insert(loc=len(TestSet.columns),column='embedding',value = embedding_value)
# import torch
# from torch import nn

# embedding = nn.Embedding(10, 3)
# input = torch.LongTensor([5,6,7,8])
# # print(embedding(input).dtype)
# a = torch.tensor([0,1,2])
# b=nn.functional.one_hot(a,num_classes= 4)

# data = [['Google',10],['Runoob',12],['Wiki',13]]

# import pandas as pd
# df = pd.DataFrame(data,columns=['Site','Age'])
# print(df)
# df.insert(loc=0,column='embed',value=b)
# print(df)
