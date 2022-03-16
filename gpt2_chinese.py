# -*- coding: utf-8 -*-

### gpt2-chinese
!pip install transformers
import sys
import torch
import numpy
from scipy.special import softmax
import pandas as pd
from transformers import BertTokenizer, GPT2LMHeadModel
# Load file uploading utility
from google.colab import files

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

# Launch file uploader
uploaded = files.upload()

!head data_critical.csv

header = ['CL', 'Sentence']
data = pd.read_csv('data_critical.csv', names = header)
data['CL_Probability'] = ""
data['CL_Surprisal'] = ""

for _, row in data.iterrows():

    tokens = tokenizer.encode(row['Sentence'])
    tokens_tensor = torch.tensor([tokens])
    # Predict all tokens
    with torch.no_grad():
      output = model(tokens_tensor)
      predictions = torch.nn.functional.softmax(output[0],-1)
    # Get the predictions 
    result = predictions[0, -1, :]

    cl_id = tokenizer.encode(row['CL'])

    cl_pro1 = result[cl_id[1]] #na_id[1:3]

    tokens_cl1 = tokenizer.encode(row['Sentence'] + row['CL'][0])
    tokens_tensor_cl1 = torch.tensor([tokens_cl1])
    with torch.no_grad():
      output_cl1 = model(tokens_tensor_cl1)
      predictions_cl1 = torch.nn.functional.softmax(output_cl1[0],-1)
    result_cl1 = predictions_cl1[0, -1, :]

    cl_pro2 = result_cl1[cl_id[2]]

    cl_pro = cl_pro1*cl_pro2

    cl_surprisal = -1*torch.log2( cl_pro )

    # Output
    row['CL_Probability'] = cl_pro.numpy()
    row['CL_Surprisal'] = cl_surprisal.numpy()

data.to_csv('results.csv')