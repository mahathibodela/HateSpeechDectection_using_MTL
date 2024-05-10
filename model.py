import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch
import torch.optim as optim

class MultiTaskModel(nn.Module):
    def __init__(self, hidden_dim, d_model, output1, output2, output3, drop_prob = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()        
        
        self.linear21 = nn.Linear(d_model, hidden_dim)
        self.dropout21 = nn.Dropout(drop_prob)
        self.linear22 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout22 = nn.Dropout(drop_prob)
        self.linear23 = nn.Linear(hidden_dim, output2)
        
        
        self.linear31 = nn.Linear(d_model, hidden_dim)
        self.dropout31 = nn.Dropout(drop_prob)
        self.linear32 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout32 = nn.Dropout(drop_prob)
        self.linear33 = nn.Linear(hidden_dim, output3)
        
    
    def forward(self, input_ids, masks, task_no):
        
        output = self.bert(input_ids, masks) #[32, 80, 768]
        output = output['last_hidden_state']
        cls = output[:,0,:] # [32 x 768]
        
        if task_no[1] == 1:
            y = self.linear21(cls)
            y = self.relu(y)
            y = self.dropout21(y)
            y = self.linear22(y)
            y = self.relu(y)
            y = self.dropout22(y)
            y = self.linear23(y)
        
        else:
            y = self.linear31(cls)
            y = self.relu(y)
            y = self.dropout31(y)
            y = self.linear32(y)
            y = self.relu(y)
            y = self.dropout32(y)
            y = self.linear33(y)
            
        
        return y

 


# model = MultiTaskModel(200, 768, 3, 7, 2)
# model.load_state_dict(torch.load("hateSpeechEmotion.pth", map_location=torch.device('cpu')))
model = torch.load("hateSpeechEmotion.pth", map_location=torch.device('cpu'))
model.eval()



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def preprocess(text):

    inputs = tokenizer(text, max_length = 80,  padding='max_length', truncation=True, return_tensors='pt' )
    return inputs['input_ids'], inputs['attention_mask']

# Define the model prediction function
def predict(text):
    input_ids, attention_mask = preprocess(text)
    task_no = torch.zeros(3)
    task_no[2] = 1
    
    with torch.no_grad():
        output = model(input_ids, attention_mask, task_no)
        prediction = torch.argmax(output, dim=1)
    return "Hate Speech" if prediction == 1 else "Not Hate Speech"



# Create Gradio interface
iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="Hate Speech Detector")

# Launch the Gradio interface
iface.launch()