from pandas import read_csv
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


truncation = 20  # The truncation length


class EBDataset(Dataset):  # Electronic Business Dataset
    def __init__(self):
        self.TrainSet = read_csv(
            "./data/train.csv",
            usecols=["event_type", "product_id", "user_id"],
            dtype={"event_type": str, "product_id": str, "user_id": str},
        )
        self.EventDict = {
            "view": 0,
            "cart": 1,
            "remove_from_cart": 2,
            "purchase": 3,
            "padding": 4,
        }  # The id of each event type
        self.users = -1  # The number of users in TrainSet after truncating
        self.RecordList = {"Prod": [], "Event": []}  # The truncated records of user
        self.ProdList = ["000"]  # The product list mapped ith product to its id
        self.ProdDict = {
            "000": 0
        }  # The product dictionary mapped a product's id to i, the '000' represents the padding product

        # Encode products, construct ProdList and ProdDict
        for index, row in tqdm(
            self.TrainSet.iterrows(), total=len(self.TrainSet), desc="Encode products"
        ):
            if not row["product_id"] in self.ProdList:
                self.ProdDict[row["product_id"]] = len(self.ProdList)
                self.ProdList.append(row["product_id"])
        # print(len(self.ProdList))#32735
        # Organize records, do truncation and padding
        self.TrainSet = self.TrainSet.groupby("user_id")
        for user_id, PartSet in tqdm(self.TrainSet, desc="Organize records"):
            i = 0
            # Truncation
            while len(PartSet) > i + truncation:
                self.users += 1  # Create a new user

                # Extract each product's index in its records
                ProdList = PartSet.iloc[i : i + truncation]["product_id"].tolist()
                ProdIndexList = [self.ProdDict[product_id] for product_id in ProdList]
                self.RecordList["Prod"].append(ProdIndexList)

                # Extract each event type in its records
                EventList = PartSet.iloc[i : i + truncation]["event_type"].tolist()
                EventIndexList = [
                    self.EventDict[event_type] for event_type in EventList
                ]
                self.RecordList["Event"].append(EventIndexList)
                i += truncation
                # print(ProdList)

            self.users += 1
            # Padding product
            ProdList = PartSet.iloc[i : len(PartSet)]["product_id"].tolist()
            for j in range(truncation - len(ProdList)):
                ProdList.append("000")
            ProdIndexList = [self.ProdDict[product_id] for product_id in ProdList]
            self.RecordList["Prod"].append(ProdIndexList)

            # Padding event
            EventList = PartSet.iloc[i : len(PartSet)]["event_type"].tolist()
            for j in range(truncation - len(EventList)):
                EventList.append("padding")
            EventIndexList = [self.EventDict[event_type] for event_type in EventList]
            self.RecordList["Event"].append(EventIndexList)
            

    def __len__(self):
        return self.users

    def __getitem__(self, idx):
        """
        Arguments:
            idx:    int, the index of some user
        Return:
            Prods:  Tensor with size [truncation], the products' record of user idx
            Events: Tensor with size [truncation], the events' record of user idx
        """

        Prods = torch.tensor(self.RecordList["Prod"][idx])
        Events = torch.tensor(self.RecordList["Event"][idx])

        src_Prods = Prods[: truncation - 1]
        tgt_Prods = Prods[1:truncation]
        src_Events = Events[: truncation - 1]
        tgt_Events = Events[1:truncation]
        # return Prods, Events
        return src_Prods, src_Events, tgt_Prods, tgt_Events


# TestSet = read_csv("./data/test.csv")
# Embedding

# TotalProducts = len(ProdList)  # 32734
# embedding = nn.Embedding(TotalProducts + 3, 508)

# Event = torch.tensor(Event)
# EmbeddingValues = torch.cat((embedding(ProdIndex), F.one_hot(Event)), dim=1)
# print("EmbeddingValues' size:", EmbeddingValues.size())
# # TrainSet.insert(loc=len(TrainSet.columns),column='embedding',value = EmbeddingValues)

# # print(len(TrainSet))

# TrainSet = TrainSet.groupby("user_id")
# for user_id, PartSet in tqdm(TrainSet):
#     pass
