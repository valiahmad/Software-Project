# from preprocessing import Preprocess
from transformers import BertModel
import pandas as pd
from torch import tensor
import torch
# df = Preprocess()


def Vectorizer(model, tokenTensor, segmentTensor, concat=True):
    with torch.no_grad():
        outputs = model(tokenTensor, segmentTensor)
        hidden_states = outputs[2]
    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)
    # token_embeddings.size()
    # torch.Size([13, 1, 10, 768])
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # token_embeddings.size()
    # torch.Size([13, 10, 768])
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)
    # token_embeddings.size()


    if concat:
        # Stores the token vectors, with shape [22 x 3,072]
        token_vecs_cat = []
        # `token_embeddings` is a [22 x 12 x 768] tensor.
        # For each token in the sentence...
        for token in token_embeddings:
            # `token` is a [12 x 768] tensor
            # Concatenate the vectors (that is, append them together) from the last 
            # four layers.
            # Each layer vector is 768 values, so `cat_vec` is length 3,072.
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            # Use `cat_vec` to represent `token`.
            token_vecs_cat.append(cat_vec)
        # print ('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))
        # Shape is: 10 x 3072
        return token_vecs_cat
    
    else:
        # Stores the token vectors, with shape [22 x 768]
        token_vecs_sum = []
        # `token_embeddings` is a [22 x 12 x 768] tensor.
        # For each token in the sentence...
        for token in token_embeddings:
            # `token` is a [12 x 768] tensor
            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)
            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)
        # print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))
        # Shape is: 10 x 768
        return token_vecs_sum





df = pd.read_excel('file_small.xlsx')
df['BERT-Tokenized']=df['BERT-Tokenized'].apply(eval)
df['TokenID']=df['TokenID'].apply(eval)
df['SegmentID']=df['SegmentID'].apply(eval)
df['Token-Tensor']=df['Token-Tensor'].apply(lambda x: eval(x))
df['Segment-Tensor']=df['Segment-Tensor'].apply(lambda x: eval(x))


# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True,)

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

df['Vectorized'] = ''
for i in range(len(df)):
    df['Vectorized'].iloc[i] = Vectorizer(model, df['Token-Tensor'].iloc[i], df['Segment-Tensor'].iloc[i])
    print(i)

for i in range(len(df['Vectorized'].iloc[14])):
    print(df['BERT-Tokenized'].iloc[14][i])
    print(i,df['Vectorized'].iloc[14][i])


df.to_excel('file_small_vec.xlsx', index=False)


