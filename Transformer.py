#### model stracture ####
from tempfile import TemporaryDirectory
# from typing import Tuple
from torch import nn, Tensor
from DataProcess import EBDataset
import math, os, torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class EBModel(nn.Module):
    def __init__(
        self, n_product: int, n_event: int, d_model=512, nhead=8, num_layers=1
    ):
        """
        n_product   : the number of product
        n_event     : the number of event
        d_model     : feature size (Must be divisible by nhead)
        nhead       : the numebr of head in model
        num_layers  : the number of Encoder_layer
        dropout     : prevent overfitting
        """

        super().__init__()
        self.model_type = "Transformer"
        self.src_mask = None
        # Define embedding layers
        self.prod_embedding = nn.Embedding(n_product, d_model - 5)
        self.event_embedding = nn.Embedding(n_event, 5)

        # Define PositionalEncoding, to add positional encoding information for input sequence
        self.pos_encoder = PositionalEncoding(d_model)

        self.d_model = d_model
        # self.transformer = nn.Transformer(d_model=self.d_model, nehad = nhead,num_encoder_layers=2,num_decoder_layers=2)

        # Define encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)

        # Define encoder
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        # Define 2 linear network to predict product index and event type
        self.prod_linear = nn.Linear(d_model, n_product)
        self.event_linear = nn.Linear(d_model, n_event)

        self.Softmax = nn.Softmax(dim= -1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.prod_embedding.weight.data.uniform_(-initrange, initrange)
        self.event_embedding.weight.data.uniform_(-initrange, initrange)
        self.prod_linear.bias.data.zero_()
        self.prod_linear.weight.data.uniform_(-initrange, initrange)
        self.event_linear.bias.data.zero_()
        self.event_linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src_product: Tensor,src_event: Tensor ,src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src_product:    Tensor, with size [truncation, batch_size]
            src_event:      Tensor, with size [truncation, batch_size]
            src_mask:       Tensor, with size [truncation, truncation]
        Returns:
            output_prod:    Tensor, with size [truncation, batch_size, n_product]
            output_event:   Tensor, with size [truncation, batch_size, n_event]
        """
        src = torch.cat(
            (self.prod_embedding(src_product), self.event_embedding(src_event)), dim=2
        ) * math.sqrt(self.d_model)
        # Now src has size [truncation, batch_size, d_model]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,src_mask)
        output_prod = self.prod_linear(output)
        output_event = self.event_linear(output)
        
        output_prod = self.Softmax(output_prod)
        output_event = self.Softmax(output_event)
        return output_prod, output_event


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 生成位置编码的位置张量
        position = torch.arange(max_len).unsqueeze(1)
        # 计算位置编码的除数项
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        # 创建位置编码张量
        pe = torch.zeros(max_len, 1, d_model)
        # 使用正弦函数计算位置编码中的奇数维度部分
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # 使用余弦函数计算位置编码中的偶数维度部分
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        #self.pe = pe
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, with size [truncation, batch_size, d_model]
        """
        # 将位置编码添加到输入张量
        x = x + self.pe[: x.size(0)]
        # 应用 dropout
        return self.dropout(x)
