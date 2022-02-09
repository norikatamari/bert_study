from utils.dataloader import get_IMDb_DataLoaders_and_TEXT
import math
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext

# Setup seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class Embedder(nn.Module):
    '''idで示されている単語をベクトルに変換します'''

    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors, freeze=True)  # freeze=Trueにより重みが更新されなくなる

    def forward(self, x):
        x_vec = self.embeddings(x)

        return x_vec


# 動作確認
train_df, val_dl, test_dl, TEXT = get_IMDb_DataLoaders_and_TEXT(
    max_length=256, batch_size=24)

# ミニバッチの用意
batch = next(iter(train_df))

# モデル構築
net1 = Embedder(TEXT.vocab.vectors)

# 入出力
x = batch.Text[0]
x1 = net1(x)  # 単語をベクトル変換

print(x.shape)  # 24, 256 (batchsize, max_seq)
print(x1.shape)  # 24, 256, 300 (batchsize, max_seq, vector)
# print(x)
# print(x1)


class PositionEncoder(nn.Module):
    ''' 入力された単語の位置を示すベクトル情報を付加する'''

    def __init__(self, d_model=300, max_seq_len=256):
        super().__init__()

        self.d_model = d_model  # 単語ベクトルの次元数

        # 単語の順番(pos)と埋め込みベクトルの次元の位置(i)によって、一位に定まる値の表をpeとして作成
        pe = torch.zeros(max_seq_len, d_model)

        # GPU対応
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * i)/d_model)))

        # 表peの先頭に、ミニバッチ次元となる次元を足す
        self.pe = pe.unsqueeze(0)

        # 勾配を計算しないようにする
        self.pe.requires_grad = False

    def forward(self, x):
        # 入力xとPositional Encodingを足し算する
        # xがpeよりも小さいので、大きくする
        ret = math.sqrt(self.d_model)*x + self.pe
        return ret


# 動作確認
net1 = Embedder(TEXT.vocab.vectors)
net2 = PositionEncoder(d_model=300, max_seq_len=256)

# 入出力
x = batch.Text[0]
x1 = net1(x)
x2 = net2(x1)

print(x1.shape)  # 24, 256, 300
print(x2.shape)  # 24, 256, 300
# print(x1[0][0])
# print(x2[0][0])


class Attention(nn.Module):
    '''Transformerは本当はマルチヘッドAttentionだが、わかりやすさを優先してシングルAttentionで実装'''

    def __init__(self, d_model=300):
        super().__init__()

        self.q_linear = nn.Linear(d_model, d_model)  # (300, 300)
        self.v_linear = nn.Linear(d_model, d_model)  # (300, 300)
        self.k_linear = nn.Linear(d_model, d_model)  # (300, 300)

        # 出力時に使用する全結合層
        self.out = nn.Linear(d_model, d_model)  # (300, 300)

        # Attentionの大きさ調整の変数
        self.d_k = d_model

    def forward(self, q, k, v, mask):
        # 全結合層で特徴量を変換
        k = self.k_linear(k)  # (24, 256, 300) * (300, 300) = (24, 256, 300)
        q = self.q_linear(q)  # (24, 256, 300) * (300, 300) = (24, 256, 300)
        v = self.v_linear(v)  # (24, 256, 300) * (300, 300) = (24, 256, 300)

        # Attentionの値を計算
        # 各値を足し算すると大きくなりすぎるので、root(d_k)で割って調整
        # (24, 256, 300) * (24, 300, 256)= (24, 256, 256) 各単語同士の重み
        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)

        # maskをかける
        # <pad>の部分を-1e9(無限大)に置き換え。softmax時に0の値にするため。softmax(-inf)=0
        mask = mask.unsqueeze(1)  # (24, 256) -> (24, 1, 256)
        weights = weights.masked_fill(mask == 0, -1e9)  # mask=Falseの部分を置き換え

        # softmax
        # (24, 256, 256)  これで各単語同士の重みが完成
        normlized_weights = F.softmax(weights, dim=-1)

        # AttentionをValueと行列積
        # (24, 256, 256) * (24, 256, 300) = (24, 256, 300)
        output = torch.matmul(normlized_weights, v)

        # 全結合層で特徴量を変換
        # (24, 256, 300) * (300, 300) = (24, 256, 300)
        output = self.out(output)

        return output, normlized_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        '''Attention層から出力を単純に全結合層2つで特徴量を変換する'''
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)  # (300, 1024)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # (1024, 300)

    def forward(self, x):
        x = self.linear_1(x)  # (24, 256, 300) * (300, 1024) = (24, 256, 1024)
        x = self.dropout(F.relu(x))  # (24, 256, 1024)
        x = self.linear_2(x)  # (24, 256, 1024) * (1024, 300) = (24, 256, 300)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        # LayerNormalization層
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        # Attention層
        self.attn = Attention(d_model)

        # Attentionのあとの全結合層
        self.ff = FeedForward(d_model)

        # Dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 正規化とAttention
        x_normlized = self.norm_1(x)  # (24, 256, 300)
        output, normlized_weights = self.attn(
            x_normlized, x_normlized, x_normlized, mask)  # ポイント！！ self_attentionだからq,k,vは全て同じものを渡す

        # xと加算 # (24, 256, 300) + (24, 256, 300) = (24, 256, 300)
        x2 = x + self.dropout_1(output)

        # 正規化と全結合層
        x_normlized2 = self.norm_2(x2)
        # (24, 256, 300) + (24, 256, 300) = (24, 256, 300)
        output = x2 + self.dropout_2(self.ff(x_normlized2))

        return output, normlized_weights  # (24, 256, 300), (24, 256, 256)


# 動作確認
net1 = Embedder(TEXT.vocab.vectors)
net2 = PositionEncoder(d_model=300, max_seq_len=256)
net3 = TransformerBlock(d_model=300)

# maskの作成
x = batch.Text[0]
input_pad = 1  # 単語のIDにおいて、<pad>:1 のため
input_mask = (x != input_pad)  # <pad>はFalse, それ以外はTrue
print(input_mask.shape)  # 24, 256

# 入出力
x1 = net1(x)  # 単語をベクトル化 (24, 256) -> (24, 256, 300)
x2 = net2(x1)  # Position情報を付加 (24, 256, 300) -> (24, 256, 300)
# Self-Attentionで特徴量を変換 (24, 256, 300) -> (24, 256, 300)
x3, normlized_weights = net3(x2, input_mask)

print(x2.shape)  # 24, 256, 300
print(x3.shape)  # 24, 256, 300
print(normlized_weights.shape)  # 24, 256, 256


class ClassificationHead(nn.Module):
    '''Transformer_Blockの出力を使用し、最後にクラス分類させる'''

    def __init__(self, d_model=300, output_dim=2):
        super().__init__()

        # 全結合層
        self.linear = nn.Linear(d_model, output_dim)  # (300, 2) ポジ、ネガに分類

        # 重み初期化処理
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, x):
        x0 = x[:, 0, :]  # 各ミニバッチの各文の先頭の単語の特徴量(300次元)を取り出す (24, 300)
        out = self.linear(x0)  # (24, 300) * (300, 2) = (24, 2)

        return out


class TransformerClassification(nn.Module):
    '''Transformerでクラス分類させる'''

    def __init__(self, text_embedding_vectors, d_model=300, max_seq_len=256, output_dim=2):
        super().__init__()

        # モデル構築
        self.net1 = Embedder(text_embedding_vectors)
        self.net2 = PositionEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.net3_1 = TransformerBlock(d_model=d_model)
        self.net3_2 = TransformerBlock(d_model=d_model)
        self.net4 = ClassificationHead(output_dim=output_dim, d_model=d_model)

    def forward(self, x, mask):
        x1 = self.net1(x)  # 単語をベクトルにする (24, 256) -> (24, 256, 300)
        x2 = self.net2(x1)  # Position情報を加算 (24, 256, 300) -> (24, 256, 300)
        # Self-Attentionで特徴量を変換 (24, 256, 300) -> (24, 256, 300)
        x3_1, normlized_weights_1 = self.net3_1(x2, mask)
        # Self-Attentionで特徴量を変換 (24, 256, 300) -> (24, 256, 300)
        x3_2, normlized_weights_2 = self.net3_2(x3_1, mask)
        # 最終出力の0単語目を使用して、分類0-1のスカラーを出力 (24, 256, 300) -> (24, 2)
        x4 = self.net4(x3_2)

        return x4, normlized_weights_1, normlized_weights_2


# 動作確認
# ミニバッチの用意
batch = next(iter(train_df))

# モデル構築
net = TransformerClassification(text_embedding_vectors=TEXT.vocab.vectors,
                                d_model=300,
                                max_seq_len=256,
                                output_dim=2)

# 入出力
x = batch.Text[0]
input_mask = (x != input_pad)
out, normlized_weights_1, normlized_weights_2 = net(x, input_mask)

print(out.shape)
print(out)
print(F.softmax(out, dim=1))
