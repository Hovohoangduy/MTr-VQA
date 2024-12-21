import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer
from utils.config import Config
from model.mtr_model import ImageEmbedding, QuesEmbedding, AnsEmbedding
from model.sans import StackAttention
from utils.data_processing import train_loader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1] 
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k) 
    if mask is not None:
        scaled += mask 
    attention = F.softmax(scaled, dim=-1) 
    values = torch.matmul(attention, v)
    return values, attention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x) 
        x = self.linear2(x)
        return x


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) # 512
        self.beta =  nn.Parameter(torch.zeros(parameters_shape)) # 512

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))] # [-1]
        mean = inputs.mean(dim=dims, keepdim=True) 
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt() 
        y = (inputs - mean) / std 
        out = self.gamma * y  + self.beta  
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model) 
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size() 
        qkv = self.qkv_layer(x) 
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) 
        q, k, v = qkv.chunk(3, dim=-1) 
        values, attention = scaled_dot_product(q, k, v, mask) 
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim) 
        out = self.linear_layer(values)
        return out


class MultiHeadCrossAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model , 2 * d_model) # 1024
        self.q_layer = nn.Linear(d_model , d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, y, mask=None):
        batch_size, sequence_length, d_model = x.size()
        kv = self.kv_layer(x) 
        q = self.q_layer(y) 
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim) 
        kv = kv.permute(0, 2, 1, 3) 
        q = q.permute(0, 2, 1, 3) 
        k, v = kv.chunk(2, dim=-1) 
        values, attention = scaled_dot_product(q, k, v, mask) 
        values = values.reshape(batch_size, sequence_length, d_model) 
        out = self.linear_layer(values) 
        return out  


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, decoder_mask):
        _y = y
        y = self.self_attention(y, mask=decoder_mask)
        y = self.dropout1(y)
        y = self.norm1(y + _y)

        _y = y # 30 x 200 x 512
        y = self.encoder_decoder_attention(x, y, mask=None)
        y = self.dropout2(y)
        y = self.norm2(y + _y)

        _y = y 
        y = self.ffn(y) 
        y = self.dropout3(y)
        y = self.norm3(y + _y) 
        return y 

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x, y, mask)
        return y

class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers=1):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) 
                                          for _ in range(num_layers)])

    def forward(self, x, y, mask):
        y = self.layers(x, y, mask)
        return y
    
if __name__=="__main__":
    d_model = 768
    num_heads = 8
    drop_prob = 0.1
    batch_size = 30
    max_sequence_length = Config.MAX_LEN_ANS
    ffn_hidden = 2048
    num_layers = 5

    image_model = ImageEmbedding(output_size=768, mode='train').to(device)
    ques_model = QuesEmbedding(output_size=768).to(device)
    ans_model = AnsEmbedding()

    for batch in train_loader:
        anno_ids, img_ids, images, questions, answers = batch
        if torch.cuda.is_available():
            images = images.cuda()
            questions = questions
            anno_ids = anno_ids
            answers = answers
        
        with torch.no_grad():
            image_embeddings, att_ids = image_model(images, image_ids=anno_ids)
            ques_embeddings = ques_model(questions)
            ans_tokens, answers_embedding = ans_model(answers)
        break 

    image_embeddings = image_embeddings.reshape(Config.TRAIN_BATCH_SIZE, 768, -1).permute(0, 2, 1)
    ques_embeddings = ques_embeddings.unsqueeze(1)

    san_model = StackAttention(d=768, k=512, dropout=True).to(device)
    img_text_att = san_model(image_embeddings.to(device), ques_embeddings.to(device))

    x = answers_embedding.float().to(device) #16 * 48 * 768
    y = img_text_att.unsqueeze(1).expand(-1, max_sequence_length, -1).to(device) # 16 * 768 -> 16 * 48 * 1024
    mask = torch.full([max_sequence_length, max_sequence_length] , float('-inf'))
    mask = torch.triu(mask, diagonal=1).to(device)
    decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers).to(device)
    out = decoder(x, y, mask).to(device)

    tokenizer = AutoTokenizer.from_pretrained(Config.textmodel_dir)
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)

    linear_layer = nn.Linear(d_model, vocab_size).to(device)
    output_logits = linear_layer(out)
    output_probs = F.softmax(output_logits, dim=2)
    output_probs.size()

    tokenizer = AutoTokenizer.from_pretrained(Config.textmodel_dir)

    decoder_output = output_probs
    # print(f"decode output", {decoder_output})
    predicted_tokens = decoder_output.argmax(dim=2)
    print(f"predicted_tokens", {predicted_tokens[-1]})
    print(f"predicted_tokens", {predicted_tokens[1]})
    def tokens_to_text(tokens, tokenizer):
        return [tokenizer.decode(token.item()) for token in tokens]

    for i in range(16):
        generated_text = tokens_to_text(predicted_tokens[-1], tokenizer)
        print(f"Generated text for example {i}: {' '.join(generated_text)}")