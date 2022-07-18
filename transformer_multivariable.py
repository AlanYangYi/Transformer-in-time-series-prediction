import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import pandas as pd

device = 'cuda'
epochs = 1000
input_features_for_encoder = 2
input_features_for_decoder = 1
projection_feature = 1
decoder_window = 7
# Transformer Parameters
d_model = 512*2  # Embedding Size（token embedding和position编码的维度）
d_ff = 2048*2  # FeedForward dimension (两次线性层中的隐藏层 512->2048->512，线性层是用来做特征提取的），当然最后会再接一个projection层
d_k = d_v = 64*2  # dimension of K(=Q), V（Q和K的维度需要相同，这里为了方便让K=V）
n_layers = 6*2  # number of Encoder of Decoder Layer（Block的个数）
n_heads = 8*2  # number of heads in Multi-Head Attention（有几套头）
encoder_days=30



def make_data(encoder_days=30,decoder_window=7):
    df = pd.read_csv('./original_data.csv')
    df_light=pd.read_csv('./lightGBM.csv')
    def create_count_feature(df1):
        for i in range(0, encoder_days+decoder_window):
            df1['count_' + 'leg' + str(i)] = df1['count'].shift(i)
        df1.DATE = pd.to_datetime(df1['DATE'], format='%Y%m%d')
        df1.dropna(axis=0, how='any', inplace=True)
        return df1
    def create_month_feature(df1):
        df1.DATE = pd.to_datetime(df1['DATE'], format='%Y%m%d')
        for i in range(0, encoder_days+decoder_window):
            df1['month_' + 'leg' + str(i)] = df1.DATE.dt.month.shift(i)
        df1.dropna(axis=0, how='any', inplace=True)
        return df1

    def create_week_feature(df1):
        df1.DATE = pd.to_datetime(df1['DATE'], format='%Y%m%d')
        for i in range(0, encoder_days+decoder_window):
            df1['week_' + 'leg' + str(i)] = df1.DATE.dt.week.shift(i)
        df1.dropna(axis=0, how='any', inplace=True)
        return df1

    def create_day_feature(df1):
        df1.DATE = pd.to_datetime(df1['DATE'], format='%Y%m%d')
        for i in range(0, encoder_days+decoder_window):
            df1['day_' + 'leg' + str(i)] = df1.DATE.dt.day.shift(i)
        df1.dropna(axis=0, how='any', inplace=True)
        return df1

    def create_dayofweek_feature(df1):
        df1.DATE = pd.to_datetime(df1['DATE'], format='%Y%m%d')
        for i in range(0, encoder_days+decoder_window):
            df1['day_of_week_' + 'leg' + str(i)] = df1.DATE.dt.dayofweek.shift(i)
        df1.dropna(axis=0, how='any', inplace=True)
        return df1

    def create_dayofyear_feature(df1):
        df1.DATE = pd.to_datetime(df1['DATE'], format='%Y%m%d')
        for i in range(0, encoder_days+decoder_window):
            df1['day_of_year_' + 'leg' + str(i)] = df1.DATE.dt.day_of_year.shift(i)
        df1.dropna(axis=0, how='any', inplace=True)
        return df1

    def create_quarter_feature(df1):
        df1.DATE = pd.to_datetime(df1['DATE'], format='%Y%m%d')
        for i in range(0, encoder_days+decoder_window):
            df1['quarter' + 'leg' + str(i)] = df1.DATE.dt.quarter.shift(i)
        df1.dropna(axis=0, how='any', inplace=True)
        return df1

    def create_lightGBM_feature(df1):
        for i in range(0, encoder_days+decoder_window):
            df1['lightGBM_' + 'leg' + str(i)] = df1['lightGBM'].shift(i)
        df1.DATE = pd.to_datetime(df1['DATE'], format='%Y-%m-%d')
        df1.dropna(axis=0, how='any', inplace=True)
        return df1

    dfs={}
    dfs['count_feature']=df.groupby('area_id').apply(create_count_feature)
    dfs['month_feature']=df.groupby('area_id').apply(create_month_feature)
    dfs['week_feature'] = df.groupby('area_id').apply(create_week_feature)
    dfs['day_feature']=df.groupby('area_id').apply(create_day_feature)
    dfs['dayofweek_feature'] = df.groupby('area_id').apply(create_dayofweek_feature)
    dfs['dayofyear_feature']=df.groupby('area_id').apply(create_dayofyear_feature)
    dfs['quarter_feature']=df.groupby('area_id').apply(create_quarter_feature)
    dfs['lightGBM_feature']=df_light.groupby('area_id').apply(create_lightGBM_feature)
    return dfs


class MyDataSet(Data.Dataset):
    """自定义DataLoader"""

    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model,max_len,dropout=0.1, ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v)做softmax
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        context = torch.matmul(attn, V)  # context: [batch_size, n_heads, len_q, d_v]
        # context：[[z1,z2,...],[...]]向量, attn注意力稀疏矩阵（用于可视化的）
        return context, attn


class MultiHeadAttention(nn.Module):
    """这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    输入：seq_len x d_model
    输出：seq_len x d_model
    """

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #           线性变换               拆成多头

        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)


        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V)
        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)

        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs):
        """E
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                             )  # enc_inputs to same Q,K,V（未线性变换前）
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs
                                                       )  # 这里的Q,K,V全是Decoder自己的输入
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs
                                                      )  # Attention层的Q(来自decoder) 和 K,V(来自encoder)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn  # dec_self_attn, dec_enc_attn这两个是为了可视化的


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb1 = nn.Linear(input_features_for_encoder, d_model*2)
        self.src_emb2=nn.Linear(d_model*2,d_model)
        self.pos_emb = PositionalEncoding(d_model,max_len=30)  # Transformer中位置编码时固定的，不需要学习
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len,features]
        """
        enc_outputs = self.src_emb1(enc_inputs)
        enc_outputs=self.src_emb2(enc_outputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]


        enc_self_attns = []  # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等，用来看各个词之间的关系
        for layer in self.layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs
                                              )  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            enc_self_attns.append(enc_self_attn)  # 这个只是为了可视化
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb1 = nn.Linear(input_features_for_decoder, d_model*2)
        self.tgt_emb2=nn.Linear(d_model*2,d_model)
        # self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)  # Decoder输入的embed词表
        self.pos_emb = PositionalEncoding(d_model,max_len=7)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])  # Decoder的blocks

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]   # 用在Encoder-Decoder Attention层
        """
        dec_outputs = self.tgt_emb1(dec_inputs)
        dec_outputs=self.tgt_emb2(dec_outputs)
        # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(device)  # [batch_size, tgt_len, d_model]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            # Decoder的Block是上一个Block的输出dec_outputs（变化）和Encoder网络的输出enc_outputs（固定）
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs
                                                             )
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        # self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)
        self.projection = nn.Linear(d_model, projection_feature, bias=False).to(device)


    def forward(self, enc_inputs, dec_inputs):
        """Transformers的输入：两个序列
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        """
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # 经过Encoder网络后，得到的输出还是[batch_size, src_len, d_model]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # dec_outputs: [batch_size, tgt_len, d_model] -> dec_logits: [batch_size,tgt_len, output_features(projection_features)]
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1)




# def infer_process(model, enc_input_, start_symbol):
#     enc_outputs_, enc_self_attns_ = model.encoder(enc_input_)
#     dec_input_ = start_symbol.unsqueeze(-1)
#     predict_result = torch.Tensor().type_as(enc_input_.data)
#     for i in range(decoder_window):
#         with torch.no_grad():
#             dec_outputs_, _, _ = model.decoder(dec_input_, enc_input_, enc_outputs_)
#             projected = model.projection(dec_outputs_)
#             projected = projected.squeeze(-1)
#             next_symbol = projected[:, -1].unsqueeze(-1)
#             dec_input_ = torch.cat((dec_input_.to(device), next_symbol.to(device)), -1)
#             predict_result = torch.cat((predict_result.to(device), next_symbol.to(device)), -1)
#
#     return predict_result



# def test_process(test_loader, model):
#     print("********prediction**********")
#
#     test_loader_=test_loader
#     all_result = torch.Tensor()
#     sum = torch.zeros(1).to(device)
#     number = 0.0
#     all_loss_=torch.zeros(1).to(device)
#     k=0.0
#     for enc_inputs_, dec_inputs_, dec_outputs_ in test_loader_:
#         model.eval()
#         model.to(device)
#         enc_inputs_, dec_inputs_, dec_outputs_ = enc_inputs_.to(device), dec_inputs_.to(device), dec_outputs_.to(device)
#         greedy_dec_predict = infer_process(model, enc_inputs_.to(device), start_symbol=enc_inputs_[:, 0].to(device))
#         result = torch.cat((dec_outputs_.to(device), greedy_dec_predict.to(device)), 1)
#         all_result = torch.cat((all_result.to(device), result.to(device)), 0)
#         mape = torch.div(dec_outputs_.to(device) - greedy_dec_predict.to(device), dec_outputs_.to(device)).abs().sum()
#         sum = sum + mape
#         number = number + greedy_dec_predict.size().numel()
#         loss = criterion(greedy_dec_predict.view(-1), dec_outputs_.view(-1))
#         all_loss_=all_loss_+loss
#         k=k+1.0
#
#     return (sum.data.cpu().numpy() / number,all_loss_.data.cpu().numpy()/k)


def test_process(test_loader, model):
    print("********prediction**********")

    test_loader_=test_loader
    sum = torch.zeros(1).to(device)
    number = 0.0
    all_loss_=torch.zeros(1).to(device)
    k=0.0
    for enc_inputs_, dec_inputs_, dec_outputs_ in test_loader_:
        k=k+1.0
        model.eval()
        model.to(device)
        enc_inputs_, dec_inputs_, dec_outputs_ = enc_inputs_.to(device), dec_inputs_.to(device), dec_outputs_.to(device)
        with torch.no_grad():
            outputs_= model(enc_inputs_, dec_inputs_)
            loss_ = criterion(outputs_,
                              dec_outputs_.view(-1))  # dec_outputs.view(-1):[batch_size * tgt_len * tgt_vocab_size]
            all_loss_ += loss_.item()
        number = number + outputs_.size().numel()
        mape = torch.div(dec_outputs_.view(-1).to(device) - outputs_.to(device), dec_outputs_.view(-1).to(device)).abs().sum()
        sum = sum + mape
    print(f"epoch:{epoch}       test_loss:{all_loss_ / k}")

    return (sum.cpu().numpy() / number, all_loss_.cpu().numpy() / k)

dic_data = make_data()
training_set={}
test_set={}
for feature in dic_data:
  training_set[feature]=dic_data[feature][(('2022-04' > dic_data[feature].DATE))]
  test_set[feature]=dic_data[feature][(dic_data[feature].DATE)>= '2022-04']

enc_inputs=torch.Tensor()
dec_inputs=torch.Tensor()
dec_outputs=torch.Tensor()
enc_inputs_=torch.Tensor()
dec_inputs_=torch.Tensor()
dec_outputs_=torch.Tensor()

for feature in dic_data:


    #if feature in ['month_feature','week_feature','day_feature','dayofweek_feature','dayofyear_feature','quarter_feature']:
    if feature in ['dayofweek_feature','count_feature']:
        enc_inputs = torch.cat(
            (enc_inputs, torch.from_numpy(training_set[feature].iloc[:, 10:].to_numpy()).unsqueeze(-1)), -1)
        enc_inputs_ = torch.cat(
            (enc_inputs_, torch.from_numpy(test_set[feature].iloc[:, 10:].to_numpy()).unsqueeze(-1)), -1)
        # #dec_inputs = torch.cat((dec_inputs, torch.from_numpy(
        #     training_set[feature].iloc[:, [10, 11, 12, 13, 14, 15, 16]].to_numpy()).unsqueeze(-1)), -1)
        # #dec_inputs_ = torch.cat((dec_inputs_, torch.from_numpy(
        #     test_set[feature].iloc[:, [10, 11, 12, 13, 14, 15, 16]].to_numpy()).unsqueeze(-1)), -1)
    if feature in ['dayofweek_feature']:

        dec_inputs = torch.cat((dec_inputs, torch.from_numpy(training_set[feature].iloc[:, [9, 8, 7, 6, 5, 4, 3]].to_numpy()).unsqueeze(-1)), -1)
        dec_inputs_ = torch.cat((dec_inputs_, torch.from_numpy(test_set[feature].iloc[:, [9, 8, 7, 6, 5, 4, 3]].to_numpy()).unsqueeze(-1)), -1)

    if feature == 'count_feature':
        dec_outputs=torch.cat((dec_outputs, torch.from_numpy(training_set[feature].iloc[:, [9, 8, 7, 6, 5, 4, 3]].to_numpy()).unsqueeze(-1)), -1)
        dec_outputs_=torch.cat((dec_outputs_, torch.from_numpy(test_set[feature].iloc[:, [9, 8, 7, 6, 5, 4, 3]].to_numpy()).unsqueeze(-1)), -1)

enc_inputs = enc_inputs.to(torch.float32)
dec_inputs = dec_inputs.to(torch.float32)
dec_outputs = dec_outputs.to(torch.float32)
loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 200, True)


enc_inputs_ = enc_inputs_.to(torch.float32)
dec_inputs_ = dec_inputs_.to(torch.float32)
dec_outputs_ = dec_outputs_.to(torch.float32)
test_loader = Data.DataLoader(MyDataSet(enc_inputs_, dec_inputs_, dec_outputs_), 10, True)


model = Transformer()

criterion = nn.L1Loss()

#optimizer = optim.SGD(model.parameters(), lr=1)

lr = 0.00001  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr,weight_decay=1e-4)
scheduler0=torch.optim.lr_scheduler.StepLR(optimizer,1.0,1.3,verbose=True)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, 5.0, gamma=0.85,verbose=True)
scheduler2=torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.95,verbose=True)
scheduler3=torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.95,verbose=True)



mapes,test_losses,traning_losses=[],[],[]

for epoch in range(epochs):
    all_loss=0.0
    i=0.0
    for enc_inputs, dec_inputs, dec_outputs in loader:
        """
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]
        """
        model.train()
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs= model(enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))  # dec_outputs.view(-1):[batch_size * tgt_len * tgt_vocab_size]
        all_loss += loss.item()
        optimizer.zero_grad()
        i = i + 1
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5, norm_type=2)
        optimizer.step()



    print(f"epoch:{epoch}       training_loss:{all_loss/i}")
    traning_losses.append(all_loss/i)

    if (epoch % 20 == 0 ):
        mape=0.0
        mape,test_loss = test_process(test_loader, model)
        mapes.append(mape)
        test_losses.append(test_loss)
        print(f"epoch:{epoch},  test_mape: {mape}, test_loss:  {test_loss}")
        if mape <= 0.01:
            print("Breaking")
            break
    if epoch <= 40:
        scheduler0.step()
    if epoch>40 and epoch <60:
        scheduler1.step()
    if epoch >60 and scheduler1.get_last_lr()[0] > 0.0001:
        scheduler2.step()

    if all_loss/i < 5:
        scheduler3.step()








