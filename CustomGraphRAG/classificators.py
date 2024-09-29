import os
import re
import random
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import DataCollatorWithPadding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CFG:
    num_workers=8
    path_11="me5_instruct_11_classes"
    path_39="me5_instruct_39_classes"
    config_path_11='me5_instruct_11_classes/config.pth'
    config_path_39='me5_instruct_39_classes/config.pth'
    model="intfloat/multilingual-e5-large-instruct"
    gradient_checkpointing=False
    batch_size=32
    seed=42
    max_len=512


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# тут нужен препроцесс из трейна
def preprocess(text):
    processed_text = " ".join(re.findall(r"[а-яА-Я0-9 ёЁ\-\.,?!+a-zA-Z]+", text))
    return processed_text

def get_detailed_instruct(task_description: str, query: str) -> str:
    # функция преобразования промпта для instruct версий моделей
    return f'Instruct: {task_description}\nQuery: {query}'

def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=512,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, texts):
        self.cfg = cfg
        self.texts = texts if isinstance(texts, list) else [texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        return inputs
    

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class CustomModel(nn.Module):
    def __init__(self, cfg, num_classes, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(
                cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.fc = nn.Linear(self.config.hidden_size, num_classes)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        feature = average_pool(outputs.last_hidden_state,
                               inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)

        return output
    
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            pred = model(inputs)
                
        preds.append(pred.to('cpu').numpy())
    
    predictions = np.concatenate(preds)
    return predictions

CFG.tokenizer = AutoTokenizer.from_pretrained(os.path.join(CFG.path_11, 'tokenizer'))

model_11 = CustomModel(CFG, num_classes=11, config_path=CFG.config_path_11, pretrained=False)
state_11 = torch.load(os.path.join(CFG.path_11, f"{CFG.model.replace('/', '-')}_fold0_best.pth"),
                   map_location=torch.device('cpu'))
model_11.load_state_dict(state_11['model'])


model_39 = CustomModel(CFG, num_classes=39, config_path=CFG.config_path_39, pretrained=False)
state_39 = torch.load(os.path.join(CFG.path_39, f"{CFG.model.replace('/', '-')}_fold0_best.pth"),
                   map_location=torch.device('cpu'))
model_39.load_state_dict(state_39['model'])

with open ("me5_instruct_11_classes/executor_le.pkl", "rb") as f:
    exec_le_11 = pickle.load(f)
    
with open ("me5_instruct_39_classes/executor_le.pkl", "rb") as f:
    exec_le_39 = pickle.load(f)

def inference_classificators(user_query: str):

    user_query = preprocess(user_query)

    user_queries = []

    if CFG.model in ['intfloat/multilingual-e5-large-instruct']:
        task = """Classify the detailed category of the given user request into one of {num_cats} categories"""
        for num_cats in ['eleven', 'thirty nine']:
            task_ = task.format(num_cats=num_cats)
            user_queries.append(get_detailed_instruct(task_, user_query))
        
    test_dataset_11 = TestDataset(CFG, user_queries[0])
    test_dataset_39 = TestDataset(CFG, user_queries[1])

    test_loader_11 = DataLoader(
        test_dataset_11,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=CFG.tokenizer, padding='longest'),
        num_workers=CFG.num_workers, pin_memory=True, drop_last=False
    )

    test_loader_39 = DataLoader(
        test_dataset_39,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=CFG.tokenizer, padding='longest'),
        num_workers=CFG.num_workers, pin_memory=True, drop_last=False
    )

    predictions_11 = inference_fn(test_loader_11, model_11, device)
    predictions_39 = inference_fn(test_loader_39, model_39, device)

    final_labels_11 = [np.argmax(el) for el in predictions_11]
    final_labels_39 = [np.argmax(el) for el in predictions_39]

    le_final_labels_11 = exec_le_11.inverse_transform(final_labels_11)
    le_final_labels_39 = exec_le_39.inverse_transform(final_labels_39)
    return le_final_labels_11[0], le_final_labels_39[0]