import math
import numpy as np

import torch
from torch import nn

import json

config_file = './weights/bert_config.json'
json_file = open(config_file, 'r')
config = json.load(json_file)

print(config)

from attrdict import AttrDict

config = AttrDict(config)

print(config.hidden_size) # 768
print(config.vocab_size) # 30522
print(config.max_position_embeddings) # 512
print(config.type_vocab_size) # 2

class BertLayerNorm(nn.Module):
    """LayerNormalization層"""

    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size)) # weight
        self.beta = nn.Parameter(torch.zeros(hidden_size)) # bias
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BertEmbeddings(nn.Module):
    """文章の単語ID列と、１文章目か２文章目かの情報を、埋め込みベクトルに変換する"""

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        # 3つのベクトル表現の埋め込み

        # Token Embedding: 単語IDを単語ベクトルに変換
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0) # padding_idx: id=0の単語（[PAD]）のベクトルを0にする 30522, 768

        # Transformer Positional Embedding: 位置情報テンソルをベクトルに変換
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size) # 512, 768

        # Sentence Embedding: 文章の１文目、２文目の情報をベクトルに変換
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size) # 2, 768

        # LayerNormalization層
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # 0.1

    def forward(self, input_ids, token_type_ids=None):
        '''
        input_idx: [batch_size, seq_len]の文章の単語IDの羅列
        token_type_ids: [batch_size, seq_len]の各単語が１文目なのか２文目なのかを示すid
        '''

        # 1. TokenEmbedding
        # 単語IDを単語ベクトルに変換
        words_embeddings = self.word_embeddings(input_ids) # ベクトルに変換

        # 2. Sentence Embedding
        if token_type_ids is None:
            # token_type_idsがない場合は文章の全単語を0にする（全て１文目とする）
            token_type_ids = torch.zeros_like(input_ids)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids) # ベクトルに変換

        # 3. Transformer Positonal Embedding
        seq_length = input_ids.size(1) # 文章の長さ
        position_ids = torch.arange(seq_length,  dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids) # [0, 1, 2,...]のように文章の長さだけ数字が昇順に入ったテンソルを作成 [batchsize, seq_len]

        position_embeddings = self.position_embeddings(position_ids) # ベクトルに変換

        # 3つの埋め込みテンソルを足し合わせる (batch_size, seq_len, hidden_size)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        # LayerNoarmalizationとDropout実行
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class BertLayer(nn.Module):
    '''BERTのBertLayerモジュール（Transformer）'''

    def __init__(self, config):
        super(BertLayer, self).__init__()

        # Self-Attention部分
        self.attention = BertAttention(config)

        # Self-Attentionの出力を処理する全結合層
        self.intermediate = BertIntermediate(config)

        # Self-Attentionによる特徴量とBeretLayerへの元の入力を足し算する層
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        '''
        hidden_states: Embedderモジュールの出力 (batch_size, seq_len, hidden_size)
        attention_mask: Transformerのマスクと同じ働きのマスキング
        attention_show_flg: Self-Attentionの重みを返すかどうか
        '''
        
        if attention_show_flg == True:
            attention_output, attention_probs = self.attention(hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)

            return layer_output, attention_probs

        elif attention_show_flg == False:
            attention_output = self.attention(hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)

            return layer_output

class BertAttention(nn.Module):
    '''BertLayerモジュールのSelf-Attention部分'''
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.selfattn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, attention_show_flg=False):
        '''
        input_tensor: Embeddingsモジュールもしくは全段のBertLayerからの出力
        attention_mask: Transoformerと同じ働きのマスキング
        attention_show_flg: Self-Attentionの重みを返すかのフラグ
        '''
        if attention_show_flg == True:
            self_output, attention_probs = self.selfattn(input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)

            return attention_output, attention_probs

        elif attention_show_flg == False:
            self_output = self.selfattn(input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            
            return attention_output

class BertSelfAttention(nn.Module):
    '''BertAttentionのSelf-Attention'''
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads # 12

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # 768/12=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 12*64=768(hidden_size)

        # Self-Attentionの特徴量を作成する全結合層
        self.query = nn.Linear(config.hidden_size, self.all_head_size) # (768, 768)
        self.key = nn.Linear(config.hidden_size, self.all_head_size) # (768, 768)
        self.value = nn.Linear(config.hidden_size, self.all_head_size) # (768, 768)

        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        '''multi-head Attention用にテンソルの形を変換する
        [batch_size, seq_len, hidden] → [batch_size, 12, seq_len, hidden/12] 
        '''
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        '''
        hidden_states：Embeddingsモジュールもしくは前段のBertLayerからの出力
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''
        # 入力を全結合層で特徴量変換
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # multi-head Attention用にテンソルの形を変換
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 特徴量同士を掛け算して、似ている度合いをAttention_scoresとして求める
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # TODO: attention_head_sizeでなくall_head_size？

        # マスク
        # 掛け算ではなく足し算。
        # この後softmaxで正規化するので、マスクされた部分は-infにしたい。attention_maskには0か-infが入っているので足し算にしている。
        attention_scores = attention_scores + attention_mask

        # Attentionをsoftmaxに掛ける
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # dropout
        attention_probs = self.dropout(attention_probs) # TODO: これはTransformerではやっていなかった

        # Attention Mapを掛け算
        context_layer = torch.matmul(attention_probs, value_layer)

        # multi-head Attentionのテンソルの形を元に戻す
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # attention_showの時は、attention_probsもリターンする
        if attention_show_flg == True:
            return context_layer, attention_probs
        elif attention_show_flg == False:
            return context_layer

class BertSelfOutput(nn.Module):
    '''BertSelfAttentionの出力を処理する全結合層'''
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # (768, 768)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        '''
        hidden_states：BertSelfAttentionの出力テンソル
        input_tensor：Embeddingsモジュールもしくは前段のBertLayerからの出力
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

def gelu(x):
    '''Gaussian Error Linear Unitという活性化関数です。
    LeLUが0でカクっと不連続なので、そこを連続になるように滑らかにした形のLeLUです。
    '''
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertIntermediate(nn.Module):
    '''BERTのTransofrmerBLockモジュールのFeedForward'''
    def __init__(self, config):
        super(BertIntermediate, self).__init__()

        # 全結合層
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size) # (768, 3072)

        # 活性化関数gelu
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        '''
        hidden_states: BertAttentionの出力テンソル
        '''

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

class BertOutput(nn.Module):
    '''BERTのTransformerBlockモジュールのFeedForward'''
    def __init__(self, config):
        super(BertOutput, self).__init__()

        # 全結合層
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size) # (3072, 768)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # 活性化関数gelu
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        '''
        hidden_states： BertIntermediateの出力テンソル
        input_tensor：BertAttentionの出力テンソル
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class BertEncoder(nn.Module):
    def __init__(self, config):
        '''BertLayerモジュールの繰り返し'''
        super(BertEncoder, self).__init__()

        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)]) # 12個のBertLayerのリスト

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, attention_show_flg=False):
        '''
        hidden_states：Embeddingsモジュールの出力
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        output_all_encoded_layers：返り値を全TransformerBlockモジュールの出力にするか、
        それとも、最終層だけにするかのフラグ。
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''

        all_encoder_layers = []

        # BertLayerモジュールの処理を繰り返す
        for layer_module in self.layer:
            if attention_show_flg == True:
                hidden_states, attention_probs = layer_module(hidden_states, attention_mask, attention_show_flg)
            elif attention_show_flg == False:
                hidden_states = layer_module(hidden_states, attention_mask, attention_show_flg)

            # BertLayerから出力された特徴量を12層分全て返却する場合
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        # 最後のBertLayerから出力された特徴量だけを返却する場合
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        if attention_show_flg == True:
            return all_encoder_layers, attention_probs
        elif attention_show_flg == False:
            return all_encoder_layers

class BertPooler(nn.Module):
    '''入力文章の１単語目[cls]の特徴量を変換して保持するためのモジュール'''
    def __init__(self, config):
        super(BertPooler, self).__init__()

        # 全結合層
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # (768, 768)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 1単語目[cls]の特徴量を取得
        first_token_tensor = hidden_states[:, 0]

        # 全結合層で特徴量変換
        pooled_output = self.dense(first_token_tensor)

        # tanh
        pooled_output = self.activation(pooled_output)

        return pooled_output

# 動作確認
# 入力の単語ID列、batch_sizeは2
input_ids = torch.LongTensor([[31, 51, 12, 23, 99], [15, 5, 1, 0, 0]])
print("入力の単語IDのテンソルサイズ: ", input_ids.shape) # (2, 5)

# マスク
attention_mask = torch.LongTensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
print("入力のマスクのテンソルサイズ: ", attention_mask.shape) # (2, 5)

# 文章のID 0が１文目、1が２文目
token_type_ids = torch.LongTensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
print("入力の文章IDのテンソルサイズ: ", token_type_ids.shape) # (2, 5)

# BERTの各モジュール
embeddings = BertEmbeddings(config)
encoder = BertEncoder(config)
pooler = BertPooler(config)

# マスクの変形 [batch_size, 1, 1, seq_length]にする
# Attentionをかけない部分はマイナス無限大にしたいので、-10000を掛け算する
extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (2,5) -> (2,1,1,5)
extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 # mask=1の要素は0になり、mask=1の要素が-10000になる
print("拡張したマスクのテンソルサイズ: ", extended_attention_mask.shape) # (2, 1, 1, 5)

# 順伝播
out1 = embeddings(input_ids, token_type_ids)
print('BertEmbeddingsの出力テンソルサイズ: ', out1.shape) # (2, 5, 768)

out2 = encoder(out1, extended_attention_mask)
print('BertEncoderの最終層の出力テンソルサイズ: ', out2[-1].shape)

out3 = pooler(out2[-1])
print('BertPoolerの出力テンソルサイズ: ', out3.shape)

# memo
# TransformerのAttention = BERTのBertAttention(BertSelfAttention+BertSelfOutput)
# TransformerのFeedForward = BERTのBertIntermediateとBertOutput



class BertModel(nn.Module):
    '''モジュールを全部つなげたBERTモデル'''

    def __init__(self, config):
        super(BertModel, self).__init__()

        # 3つのモジュールを作成
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, attention_show_flg=False):
        '''
        input_ids： [batch_size, sequence_length]の文章の単語IDの羅列
        token_type_ids： [batch_size, sequence_length]の、各単語が1文目なのか、2文目なのかを示すid
        attention_mask：Transformerのマスクと同じ働きのマスキングです
        output_all_encoded_layers：最終出力に12段のTransformerの全部をリストで返すか、最後だけかを指定
        attention_show_flg：Self-Attentionの重みを返すかのフラグ
        '''

        # Attentionのマスクと文の1文目、2文目のidが無ければ作成する
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # マスクの変形　[minibatch, 1, 1, seq_length]にする
        # 後ほどmulti-head Attentionで使用できる形にしたいので
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # マスクは0、1だがソフトマックスを計算したときにマスクになるように、0と-infにする
        # -infの代わりに-10000にしておく
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 順伝搬させる
        # BertEmbeddinsモジュール
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # BertLayerモジュール（Transformer）を繰り返すBertEncoderモジュール
        if attention_show_flg == True:
            '''attention_showのときは、attention_probsもリターンする'''

            encoded_layers, attention_probs = self.encoder(embedding_output,
                                                           extended_attention_mask,
                                                           output_all_encoded_layers, attention_show_flg)

        elif attention_show_flg == False:
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_all_encoded_layers, attention_show_flg)

        # BertPoolerモジュール
        # encoderの一番最後のBertLayerから出力された特徴量を使う
        pooled_output = self.pooler(encoded_layers[-1])

        # output_all_encoded_layersがFalseの場合はリストではなく、テンソルを返す
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # attention_showのときは、attention_probs（1番最後の）もリターンする
        if attention_show_flg == True:
            return encoded_layers, pooled_output, attention_probs
        elif attention_show_flg == False:
            return encoded_layers, pooled_output


# 動作確認
# 入力の用意
input_ids = torch.LongTensor([[31, 51, 12, 23, 99], [15, 5, 1, 0, 0]])
attention_mask = torch.LongTensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])

# BERTモデルを作る
net = BertModel(config)

# 順伝搬させる
encoded_layers, pooled_output, attention_probs = net(
    input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, attention_show_flg=True)

print("encoded_layersのテンソルサイズ：", encoded_layers.shape)
print("pooled_outputのテンソルサイズ：", pooled_output.shape)
print("attention_probsのテンソルサイズ：", attention_probs.shape)

# 学習ずみモデルのロード
weights_path = './weights/pytorch_model.bin'
loaded_state_dict = torch.load(weights_path)

# for s in loaded_state_dict.keys():
#     print(s)

# モデル定義
net = BertModel(config)
net.eval()

param_names = []
for name, param in net.named_parameters():
    # print(name)
    param_names.append(name)

# 学習済みモデルはそのまま使えないので、先頭から順にパラメータの中身を入れていく

# state_dictの名前が違うので前から順番に代入する
# 今回、パラメータの名前は違っていても、対応するものは同じ順番になっています

# 現在のネットワークの情報をコピーして新たなstate_dictを作成
new_state_dict = net.state_dict().copy()

# 新たなstate_dictに学習済みの値を代入
for index, (key_name, value) in enumerate(loaded_state_dict.items()):
    name = param_names[index]  # 現在のネットワークでのパラメータ名を取得
    new_state_dict[name] = value  # 値を入れる
    print(str(key_name)+"→"+str(name))  # 何から何に入ったかを表示

    # 現在のネットワークのパラメータを全部ロードしたら終える
    if index+1 >= len(param_names):
        break

# 新たなstate_dictを実装したBERTモデルに与える
net.load_state_dict(new_state_dict)


# vocabファイルを読み込み、
import collections


def load_vocab(vocab_file):
    """text形式のvocabファイルの内容を辞書に格納します"""
    vocab = collections.OrderedDict()  # (単語, id)の順番の辞書変数
    ids_to_tokens = collections.OrderedDict()  # (id, 単語)の順番の辞書変数
    index = 0

    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()

            # 格納
            vocab[token] = index
            ids_to_tokens[index] = token
            index += 1

    return vocab, ids_to_tokens


# 実行
vocab_file = "./vocab/bert-base-uncased-vocab.txt"
vocab, ids_to_tokens = load_vocab(vocab_file) # 単語:IDの辞書、ID:単語の辞書


from utils.tokenizer import BasicTokenizer, WordpieceTokenizer

# BasicTokenizer, WordpieceTokenizerは、引用文献[2]そのままです
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py
# これらはsub-wordで単語分割を行うクラスになります。
class BertTokenizer(object):
    '''BERT用の文章の単語分割クラスを実装'''

    def __init__(self, vocab_file, do_lower_case=True):
        '''
        vocab_file：ボキャブラリーへのパス
        do_lower_case：前処理で単語を小文字化するかどうか
        '''

        # ボキャブラリーのロード
        self.vocab, self.ids_to_tokens = load_vocab(vocab_file)

        # 分割処理の関数をフォルダ「utils」からimoprt、sub-wordで単語分割を行う
        never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
        # (注釈)上記の単語は途中で分割させない。これで一つの単語とみなす

        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                              never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        '''文章を単語に分割する関数'''
        split_tokens = []  # 分割後の単語たち
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """分割された単語リストをIDに変換する関数"""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])

        return ids

    def convert_ids_to_tokens(self, ids):
        """IDを単語に変換する関数"""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens


# 文章1：銀行口座にアクセスしました。
text_1 = "[CLS] I accessed the bank account. [SEP]"

# 文章2：彼は敷金を銀行口座に振り込みました。
text_2 = "[CLS] He transferred the deposit money into the bank account. [SEP]"

# 文章3：川岸でサッカーをします。
text_3 = "[CLS] We play soccer at the bank of the river. [SEP]"

# 単語分割Tokenizerを用意
tokenizer = BertTokenizer(
    vocab_file="./vocab/bert-base-uncased-vocab.txt", do_lower_case=True)

# 文章を単語分割
tokenized_text_1 = tokenizer.tokenize(text_1)
tokenized_text_2 = tokenizer.tokenize(text_2)
tokenized_text_3 = tokenizer.tokenize(text_3)

# 確認
print(tokenized_text_1)


# 単語をIDに変換する
indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
indexed_tokens_2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)
indexed_tokens_3 = tokenizer.convert_tokens_to_ids(tokenized_text_3)

# 各文章のbankの位置
bank_posi_1 = np.where(np.array(tokenized_text_1) == "bank")[0][0]  # 4
bank_posi_2 = np.where(np.array(tokenized_text_2) == "bank")[0][0]  # 8
bank_posi_3 = np.where(np.array(tokenized_text_3) == "bank")[0][0]  # 6

# seqId（1文目か2文目かは今回は必要ない）

# リストをPyTorchのテンソルに
tokens_tensor_1 = torch.tensor([indexed_tokens_1])
tokens_tensor_2 = torch.tensor([indexed_tokens_2])
tokens_tensor_3 = torch.tensor([indexed_tokens_3])

# bankの単語id
bank_word_id = tokenizer.convert_tokens_to_ids(["bank"])[0]

# 確認
print(tokens_tensor_1)


# 文章をBERTで処理
with torch.no_grad():
    encoded_layers_1, _ = net(tokens_tensor_1, output_all_encoded_layers=True)
    encoded_layers_2, _ = net(tokens_tensor_2, output_all_encoded_layers=True)
    encoded_layers_3, _ = net(tokens_tensor_3, output_all_encoded_layers=True)


# bankの初期の単語ベクトル表現
# これはEmbeddingsモジュールから取り出し、単語bankのidに応じた単語ベクトルなので3文で共通している
bank_vector_0 = net.embeddings.word_embeddings.weight[bank_word_id]

# 文章1のBertLayerモジュール1段目から出力されるbankの特徴量ベクトル
bank_vector_1_1 = encoded_layers_1[0][0, bank_posi_1]

# 文章1のBertLayerモジュール最終12段目から出力されるのbankの特徴量ベクトル
bank_vector_1_12 = encoded_layers_1[11][0, bank_posi_1]

# 文章2、3も同様に
bank_vector_2_1 = encoded_layers_2[0][0, bank_posi_2]
bank_vector_2_12 = encoded_layers_2[11][0, bank_posi_2]
bank_vector_3_1 = encoded_layers_3[0][0, bank_posi_3]
bank_vector_3_12 = encoded_layers_3[11][0, bank_posi_3]

# コサイン類似度を計算
import torch.nn.functional as F

print("bankの初期ベクトル と 文章1の1段目のbankの類似度：",
      F.cosine_similarity(bank_vector_0, bank_vector_1_1, dim=0))
print("bankの初期ベクトル と 文章1の12段目のbankの類似度：",
      F.cosine_similarity(bank_vector_0, bank_vector_1_12, dim=0))

print("文章1の1層目のbank と 文章2の1段目のbankの類似度：",
      F.cosine_similarity(bank_vector_1_1, bank_vector_2_1, dim=0))
print("文章1の1層目のbank と 文章3の1段目のbankの類似度：",
      F.cosine_similarity(bank_vector_1_1, bank_vector_3_1, dim=0))

print("文章1の12層目のbank と 文章2の12段目のbankの類似度：",
      F.cosine_similarity(bank_vector_1_12, bank_vector_2_12, dim=0))
print("文章1の12層目のbank と 文章3の12段目のbankの類似度：",
      F.cosine_similarity(bank_vector_1_12, bank_vector_3_12, dim=0))
