import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from DataProcess import EBDataset  # Assuming EBDataset class is defined in a file named EBDataset.py
from Transformer import EBModel


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor_tuple = (torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1]))
        return tensor_tuple


batch_size = 4 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_square_subsequent_mask(size):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.float().masked_fill(mask == 0, float(1.0)).masked_fill(mask == 1, float(0.0))
    return mask


def main():

    eb_dataset = EBDataset()
    custom_dataset = CustomDataset(eb_dataset)
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, drop_last=True)#去掉不规整的最后一个数据
    
    n_product=32735 #DataProcess里 print(len(self.ProdList))
    n_event=5
    num_epochs=10
    eb_model = EBModel(n_product, n_event, d_model=512, nhead=8, num_layers=1)
    
    checkpoint_path = ''
    
    for epoch in range(num_epochs):
        for product,event in data_loader:
            product=product.permute(1, 0)
            event=event.permute(1, 0)
            #print(product.shape) #torch.Size([20, 4])
            #print(event.shape)  #torch.Size([20, 4])
            
            truncation=product.shape[0]
            src_mask=generate_square_subsequent_mask(truncation)
            #print(src_mask)
            
            output_prod, output_event=eb_model(product,event,src_mask)
            #print(output_prod.shape)    #torch.Size([20, 4, 32735])
            #print(output_event.shape)   #torch.Size([20, 4, 5])
            
            prod_one_hot = F.one_hot(product, num_classes=32735).float()
            event_one_hot = F.one_hot(event, num_classes=5).float()
            #print(prod_one_hot.shape)    #torch.Size([20, 4, 32735])
            #print(event_one_hot.shape)      #torch.Size([20, 4, 5])
            
            criterion_prod = CrossEntropyLoss()
            criterion_event = CrossEntropyLoss()
            
            loss_prod = criterion_prod(output_prod,prod_one_hot)
            loss_event = criterion_event(output_event,event_one_hot)
            
            total_loss = loss_prod + loss_event
            
            optimizer = optim.Adam(eb_model.parameters(), lr=0.001)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch + 1}, Batch Loss: {total_loss.item()}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': eb_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss.item(),
        }, checkpoint_path)
    # 前向传播
    #output_prod, output_event = eb_model(inputs)

    
if __name__ == "__main__":
    main()


"""
    for idx in range(len(eb_dataset)):
        sample_prods, sample_events = eb_dataset[idx]
        print(f"User {idx + 1}:")
        print("  Products:", sample_prods)
        print("  Events:", sample_events)
        print("\n")
    User 63909:
        Products: [1976, 1976, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Events: [0, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
"""


"""
    print(type(eb_dataset))         <class 'DataProcess.EBDataset'>
    print(type(eb_dataset[0]))      <class 'tuple'>
    print(type(eb_dataset[0][0]))   <class 'list'>
    print(type(eb_dataset[0][1]))   <class 'list'>
    print(eb_dataset[0][0])         [1, 2, 3, 4, 5, 6, 7, 7, 8, 4, 9, 10, 11, 12, 2, 6, 1, 13, 14, 15]
    print(eb_dataset[0][1])         [1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
"""




