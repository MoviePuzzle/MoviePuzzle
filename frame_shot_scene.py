import os
import itertools
from tqdm import tqdm
import collections

import faiss
import numpy as np
# from sklearn.cluster import  KMeans
from kmeans_pytorch import kmeans, kmeans_predict
import torch
from torch import nn

from transformers import BertModel, BertForNextSentencePrediction, BertForSequenceClassification, T5EncoderModel, BeitModel, CLIPModel, AlbertModel
from transformers import CLIPVisionConfig, BertTokenizer, AlbertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


from utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

data_path = 'data/movienet/VideoReorder-MovieNet'
split = 'train'
# train_data = VideoReorderMovieNetDataFolder(root=data_path, split=split, layer='shot')
train_data = NewVideoReorderMovieNetDataFolder(root=data_path, split=split, layer='')
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=(split == 'train'), num_workers=0, pin_memory=True, collate_fn=lambda x: x)

split = 'val'
# val_data = VideoReorderMovieNetDataFolder(root=data_path, split=split, layer='shot')
val_data = NewVideoReorderMovieNetDataFolder(root=data_path, split=split, layer='')
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=2, shuffle=(split == 'train'), num_workers=0, pin_memory=True, collate_fn=lambda x: x)

# # 1 naive net
# net = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(2048, 1024),
#     nn.ReLU(),
#     nn.Linear(1024, 512),
#     nn.ReLU(),
#     nn.Linear(512,2)
# )
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
# net.apply(init_weights)

# bert

class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        # self.embed_dim = config.hidden_size
        self.embed_dim = 768
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))
        
        self.proj = nn.Linear(self.embed_dim, 128)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        embeddings = self.proj(embeddings)
        return embeddings #B, seq_len, 768

class CLS_HEAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768*2, 768)
        self.activation = nn.Tanh()
        self.dense2 = nn.Linear(768, 512)
        self.activation2 = nn.ReLU()
        self.cls_header = nn.Linear(512, 2)
        
        # self.dense = nn.Linear(4096*2, 4096)
        # self.activation = nn.Tanh()
        # self.dense2 = nn.Linear(4096, 1024)
        # self.activation2 = nn.ReLU()
        # self.cls_header = nn.Linear(1024, 2)

    def forward(self, hidden_states: torch.Tensor, pos1=0, pos2=0) -> torch.Tensor:
        # # mean pooling
        # pooled_token_tensor = hidden_states.mean(1)
        
        # cat pooling
        pooled_token_tensor = torch.cat((hidden_states[:, pos1], hidden_states[:, pos2]), dim=-1)
        
        # # first token pooling
        # pooled_token_tensor = hidden_states[:, 0]
        
        # # middel token pooling
        # pooled_token_tensor = hidden_states[:, pos2]
        
        pooled_output = self.dense(pooled_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dense2(pooled_output)
        pooled_output = self.activation2(pooled_output)
        output = self.cls_header(pooled_output)
        return output

class CLS_SHOT_HEAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768*2, 768)
        self.activation = nn.Tanh()
        self.dense2 = nn.Linear(768, 512)
        self.activation2 = nn.ReLU()
        self.cls_header = nn.Linear(512, 2)
        
        # self.dense = nn.Linear(4096*2, 4096)
        # self.activation = nn.Tanh()
        # self.dense2 = nn.Linear(4096, 1024)
        # self.activation2 = nn.ReLU()
        # self.cls_header = nn.Linear(1024, 2)

    def forward(self, hidden_states: torch.Tensor, pos1=[0], pos2=[0]) -> torch.Tensor:
        # # mean pooling
        # pooled_token_tensor = hidden_states.mean(1)
        
        # cat pooling
        # pooled_token_tensor = torch.cat((hidden_states[:,pos1], hidden_states[:,pos2]), dim=-1)
        pooled_token_tensor = torch.cat((torch.index_select(hidden_states, 1, pos1), torch.index_select(hidden_states, 1, pos2)).mean(1), dim=-1)
        
        # # first token pooling
        # pooled_token_tensor = hidden_states[:, 0]
        
        # # middel token pooling
        # pooled_token_tensor = hidden_states[:, pos2]
        
        pooled_output = self.dense(pooled_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dense2(pooled_output)
        pooled_output = self.activation2(pooled_output)
        output = self.cls_header(pooled_output)
        return output


class CLUSTER_HEAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()
        self.dense2 = nn.Linear(768, 512)
        self.activation2 = nn.ReLU()
        self.cluster_header = nn.Linear(512, 512)
        
        # self.dense = nn.Linear(4096, 4096)
        # self.activation = nn.Tanh()
        # self.dense2 = nn.Linear(4096, 2048)
        # self.activation2 = nn.ReLU()
        # self.cluster_header = nn.Linear(2048, 1024)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # # mean pooling
        # pooled_token_tensor = hidden_states.mean(1)
        
        # # cat pooling
        # pooled_token_tensor = torch.cat((hidden_states[:,pos1], hidden_states[:,pos2]), dim=-1)
        
        # first token pooling
        pooled_token_tensor = hidden_states[:, 0]
        
        # # middel token pooling
        # pooled_token_tensor = hidden_states[:, pos2]
        
        pooled_output = self.dense(pooled_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dense2(pooled_output)
        pooled_output = self.activation2(pooled_output)
        output = self.cluster_header(pooled_output)
        return output


clip_embeding = nn.Sequential(
    nn.Linear(1024, 768),
    nn.ReLU()
)
clip_embeding.apply(init_weights)

# in frame reorder
net = AlbertModel.from_pretrained('prev_trained_models/albert-base-v2', num_hidden_layers=1)
# net = AlbertModel.from_pretrained('prev_trained_models/albert-xxlarge-v2', num_hidden_layers=1)
# net = AlbertModel.from_pretrained('prev_trained_models/albert-xxlarge-v2')
# for name, param in net.named_parameters():
#     if name.startswith("bert.encoder.layer.1"): param.requires_grad = False
# net = BertForNextSentencePrediction.from_pretrained('prev_trained_models/bert-base-uncased', num_hidden_layers=6)
# tokenizer = AlbertTokenizer.from_pretrained('prev_trained_models/albert-base-v2')
tokenizer = AlbertTokenizer.from_pretrained('prev_trained_models/albert-xxlarge-v2')
vit_config = CLIPVisionConfig.from_pretrained('prev_trained_models/clip-vit-base-patch32')
# vit_model = CLIPModel.from_pretrained('prev_trained_models/clip-vit-base-patch32')
# vit_embed = vit_model.vision_model.embeddings
state_dict = torch.load('prev_trained_models/clip-vit-base-patch32/pytorch_model.bin')
vit_embed = CLIPVisionEmbeddings(vit_config)
embed_state_dict = {'class_embedding':state_dict['vision_model.embeddings.class_embedding'],
                    'position_ids':state_dict['vision_model.embeddings.position_ids'],
                    'patch_embedding.weight':state_dict['vision_model.embeddings.patch_embedding.weight'],
                    'position_embedding.weight':state_dict['vision_model.embeddings.position_embedding.weight'],}
vit_embed.load_state_dict(embed_state_dict, strict=False)
# vit_embed 
# vit_embed.apply(init_weights)
cls_head = CLS_HEAD()
cls_head.apply(init_weights)

# frame cluster
# cluster_net = AlbertModel.from_pretrained('prev_trained_models/albert-base-v2', num_hidden_layers=1)
cluster_head = CLUSTER_HEAD()
cluster_head.apply(init_weights)
cluster_head.to(device)

shot_cluster_head = CLUSTER_HEAD()
shot_cluster_head.apply(init_weights)
shot_cluster_head.to(device)

# shot reorder
# shot_net = AlbertModel.from_pretrained('prev_trained_models/albert-base-v2', num_hidden_layers=2)
# shot_net.to(device)
shot_vit_embed = CLIPVisionEmbeddings(vit_config)
shot_vit_embed.apply(init_weights)
# shot_cls_head = CLS_SHOT_HEAD()
shot_cls_head = CLS_HEAD()
shot_cls_head.apply(init_weights)
shot_vit_embed.to(device)
shot_cls_head.to(device)

scene_cls_head = CLS_HEAD()
scene_cls_head.apply(init_weights)
scene_cls_head.to(device)

# clip_embeding.to(device)
net.to(device)
vit_embed.to(device)
cls_head.to(device)
# wandb.watch(net)

lr = 5e-5
epoch = 1

loss_func = nn.CrossEntropyLoss()
loss_func.to(device)
cluster_loss_func = nn.CrossEntropyLoss()
cluster_loss_func.to(device)
# optim = torch.optim.AdamW(net.parameters(), lr=lr)


# create optimizer and lr_schedule
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optim = AdamW(optimizer_grouped_parameters,
                lr=lr, eps=1e-8)
t_total = len(train_dataloader) * epoch
warmup_iters = t_total * 0.1
lr_scheduler = get_linear_schedule_with_warmup(optim, warmup_iters, t_total)


du_metric_func = DoubleLengthMatching
tr_metric_func = TripleLengthMatching
softmax = nn.Softmax(dim=-1)
# train
for e in range(epoch):
    print(f'epoch {e}:')

    # train
    loss_epoch_shot_list = []
    score_epoch_shot_list = []
    loss_epoch_cluster_list = []
    
    loss_epoch_scene_list = []
    score_epoch_scene_list = []
    loss_epoch_scene_cluster_list = []
    
    loss_epoch_list = []
    score_epoch_list = []
    with tqdm(train_dataloader, unit='batch') as tepoch:
        for batch_data in tepoch:

            tepoch.set_description(f'epoch {e}')
            
            loss_batch_cluster_list = []
            loss_batch_list = []
            score_batch_list = []
            loss_batch_shot_list = []
            score_batch_shot_list = []
            
            loss_batch_scene_cluster_list = []
            loss_batch_scene_list = []
            score_batch_scene_list = []

            for item_data in batch_data:        
                text, img, img_id, shot_id, scene_id = item_data
                img = torch.load(img)
                img = img.to(device)
                # L, _,  H = features.size() # L * 1 * 1024
                L = len(text)
                
                
                
                # # ==>1. optimize in frame ordering
                # coups = list(itertools.combinations(range(L), 2))
                coups = list(itertools.permutations(range(L), 2))
                sample_length = min(10, len(coups))
                selected_indexs = random.sample(coups, sample_length)
                loss_sample = 0
                score_sample = 0
                loss_cluster = 0
                loss_sample_shot = 0
                score_sample_shot = 0
                loss_cluster_scene = 0
                loss_sample_scene = 0
                score_sample_scene = 0
                for (idx1, idx2) in selected_indexs:
                    # bert_tokens = tokenizer.encode_plus(
                    #     text[idx1],
                    #     text[idx2],
                    #     add_special_tokens=True,
                    #     max_length=256
                    # )["input_ids"]
                    tokens1 = tokenizer.encode(text[idx1])
                    tokens2 = tokenizer.encode(text[idx2])
                    cls1, cls2 = 0, len(tokens1)
                    bert_tokens = tokens1 + tokens2
                    
                    text_ids = torch.LongTensor(bert_tokens).unsqueeze(0).to(device)
                    # text_ids = tokenizer.encode([text[idx1] + '[SEP]' + text[idx2]]).to(device)
                    text_embeddings = net.embeddings(text_ids)
                    
                    vis_pt = torch.stack((img[idx1], img[idx2]), dim=0)
                    vis_embeddings = vit_embed(vis_pt)
                    vis_len = vis_embeddings.size(1)
                    vis_embeddings = vis_embeddings.view(1, vis_len*2, -1)
                    bert_embedding = torch.cat((text_embeddings, vis_embeddings), dim=1)
                    hidden_states = net(inputs_embeds=bert_embedding)[0]
                    cls3, cls4 = text_embeddings.size(1), text_embeddings.size(1)+vis_len
                    
                    # order loss
                    output = cls_head(hidden_states, cls1, cls2)
                    loss_sample += loss_func(output, torch.tensor([1]).to(device) if img_id[idx1] < img_id[idx2] else torch.tensor([0]).to(device))
                    PRED = get_order_list(output.detach().reshape(-1).cpu())
                    GT = get_order_list([img_id[idx1], img_id[idx2]])
                    score_sample += int(PRED == GT)
                    
                
                    # # cluster loss
                    # cluster_output = cluster_head(hidden_states)
                    # loss_cluster += cluster_loss_func(cluster_output, torch.tensor([int(shot_id[idx1] == shot_id[idx2])]).to(device))
                    
                
                
                # # ==> optimize clustering and shot ordering
                shuffled_list = [(idx, img_id[idx], shot_id[idx]) for idx in range(len(img_id))]
                ori_list = sorted(shuffled_list, key=lambda x: x[1])
                # init shot text and image
                n_shot = len(set(shot_id.tolist()))
                indice_dct = collections.defaultdict(list)
                for idx, img_id_idx, shot_id_idx in ori_list:
                    indice_dct[shot_id_idx.item()].append(idx)
                indice_list = [list(indice_dct[x]) for x in indice_dct.keys()]
                
                # # ==> cl loss for clustering
                cluster_sample = 0
                
                for shot_seg_list in indice_list:
                    if len(shot_seg_list) >= 2 and len(indice_list) > 1:
                        pos_list = random.sample(shot_seg_list, 2)
                        anchor_idx, pos_idx = pos_list
                        remain = [x for x in indice_list if x != shot_seg_list]
                        neg_idx = random.sample(random.sample(remain, 1)[0], 1)[0]
                        
                        anchor_tokens = tokenizer.encode(text[anchor_idx], max_length=64, truncation=True, padding='max_length')
                        pos_tokens = tokenizer.encode(text[pos_idx], max_length=64, truncation=True, padding='max_length')
                        neg_tokens = tokenizer.encode(text[neg_idx], max_length=64, truncation=True, padding='max_length')
                        # anchor_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[anchor_idx])).unsqueeze(0).to(device)) 
                        # pos_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[pos_idx])).unsqueeze(0).to(device)) 
                        # neg_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[neg_idx])).unsqueeze(0).to(device)) 
                        text_token_ids = torch.LongTensor([anchor_tokens, pos_tokens, neg_tokens]).to(device)
                        text_embeddings = net.embeddings(text_token_ids)
                        
                        vis_pt = torch.stack((img[anchor_idx], img[pos_idx], img[neg_idx]), dim=0)
                        vis_embeddings = vit_embed(vis_pt)
                        
                        # anchor_embeds = torch.cat((text_embeddings[0].unsqueeze(0), vis_embeddings[0].unsqueeze(0)))
                        # pos_embeds = torch.cat((text_embeddings[1].unsqueeze(0), vis_embeddings[1].unsqueeze(0)))
                        # neg_embeds = torch.cat((text_embeddings[2].unsqueeze(0), vis_embeddings[2].unsqueeze(0)))
                        input_embeddings = torch.cat((text_embeddings, vis_embeddings), dim=1)
                        # input_embeddings = torch.cat((vis_embeddings, text_embeddings), dim=1)
                        hidden_states = net(inputs_embeds=input_embeddings)[0]
                        cluster_output = cluster_head(hidden_states)
                        loss_cluster += cl_loss(cluster_output[0].unsqueeze(0), cluster_output[1].unsqueeze(0), cluster_output[2].unsqueeze(0)).squeeze(0)
                        cluster_sample += 1
                
                
                # for shot_seg_list in indice_list:
                #     if len(shot_seg_list) >= 2:
                #         for pos_list in itertools.permutations(range(len(shot_seg_list)), 2):
                #             # pos_list = random.sample(shot_seg_list, 2)
                #             anchor_idx, pos_idx = shot_seg_list[pos_list[0]], shot_seg_list[pos_list[1]]
                #             remain = [x for x in indice_list if x != shot_seg_list]
                #             neg_idx = random.sample(random.sample(remain, 1)[0], 1)[0]
                            
                #             anchor_tokens = tokenizer.encode(text[anchor_idx], max_length=64, truncation=True, padding='max_length')
                #             pos_tokens = tokenizer.encode(text[pos_idx], max_length=64, truncation=True, padding='max_length')
                #             neg_tokens = tokenizer.encode(text[neg_idx], max_length=64, truncation=True, padding='max_length')
                #             # anchor_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[anchor_idx])).unsqueeze(0).to(device)) 
                #             # pos_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[pos_idx])).unsqueeze(0).to(device)) 
                #             # neg_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[neg_idx])).unsqueeze(0).to(device)) 
                #             text_token_ids = torch.LongTensor([anchor_tokens, pos_tokens, neg_tokens]).to(device)
                #             text_embeddings = net.embeddings(text_token_ids)
                            
                #             vis_pt = torch.stack((img[anchor_idx], img[pos_idx], img[neg_idx]), dim=0)
                #             vis_embeddings = vit_embed(vis_pt)
                            
                #             # anchor_embeds = torch.cat((text_embeddings[0].unsqueeze(0), vis_embeddings[0].unsqueeze(0)))
                #             # pos_embeds = torch.cat((text_embeddings[1].unsqueeze(0), vis_embeddings[1].unsqueeze(0)))
                #             # neg_embeds = torch.cat((text_embeddings[2].unsqueeze(0), vis_embeddings[2].unsqueeze(0)))
                #             input_embeddings = torch.cat((text_embeddings, vis_embeddings), dim=1)
                #             # input_embeddings = torch.cat((vis_embeddings, text_embeddings), dim=1)
                #             hidden_states = net(inputs_embeds=input_embeddings)[0]
                #             cluster_output = cluster_head(hidden_states)
                #             loss_cluster += cl_loss(cluster_output[0].unsqueeze(0), cluster_output[1].unsqueeze(0), cluster_output[2].unsqueeze(0)).squeeze(0)
                #             cluster_sample += 1
                #     else:
                #         anchor_idx, pos_idx = shot_seg_list[0], shot_seg_list[0]
                #         remain = [x for x in indice_list if x != shot_seg_list]
                #         neg_idx = random.sample(random.sample(remain, 1)[0], 1)[0]
                        
                #         anchor_tokens = tokenizer.encode(text[anchor_idx], max_length=64, truncation=True, padding='max_length')
                #         pos_tokens = tokenizer.encode(text[pos_idx], max_length=64, truncation=True, padding='max_length')
                #         neg_tokens = tokenizer.encode(text[neg_idx], max_length=64, truncation=True, padding='max_length')
                #         # anchor_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[anchor_idx])).unsqueeze(0).to(device)) 
                #         # pos_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[pos_idx])).unsqueeze(0).to(device)) 
                #         # neg_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[neg_idx])).unsqueeze(0).to(device)) 
                #         text_token_ids = torch.LongTensor([anchor_tokens, pos_tokens, neg_tokens]).to(device)
                #         text_embeddings = net.embeddings(text_token_ids)
                        
                #         vis_pt = torch.stack((img[anchor_idx], img[pos_idx], img[neg_idx]), dim=0)
                #         vis_embeddings = vit_embed(vis_pt)
                        
                #         # anchor_embeds = torch.cat((text_embeddings[0].unsqueeze(0), vis_embeddings[0].unsqueeze(0)))
                #         # pos_embeds = torch.cat((text_embeddings[1].unsqueeze(0), vis_embeddings[1].unsqueeze(0)))
                #         # neg_embeds = torch.cat((text_embeddings[2].unsqueeze(0), vis_embeddings[2].unsqueeze(0)))
                #         input_embeddings = torch.cat((text_embeddings, vis_embeddings), dim=1)
                #         # input_embeddings = torch.cat((vis_embeddings, text_embeddings), dim=1)
                #         hidden_states = net(inputs_embeds=input_embeddings)[0]
                #         cluster_output = cluster_head(hidden_states)
                #         loss_cluster += cl_loss(cluster_output[0].unsqueeze(0), cluster_output[1].unsqueeze(0), cluster_output[2].unsqueeze(0)).squeeze(0)
                #         cluster_sample += 1

        
                # indice_list = []
                # tmp_indice_list = []
                # prev_idx = -1
                # for tp in ori_list:
                #     if tmp_indice_list == []:
                #         tmp_indice_list.append(tp[0])
                #     else:
                #         if tp[2] != prev_idx:
                #             indice_list.append(tmp_indice_list)
                #             tmp_indice_list = [tp[0]]
                #         else:
                #             tmp_indice_list.append(tp[0])
                #     prev_idx = tp[2]
                # if indice_list[-1] != tmp_indice_list:
                #     indice_list.append(tmp_indice_list)
                
                text_list = []
                img_list = []
                for item_indice_list in indice_list:
                    item_text_list = []
                    for item_indice in item_indice_list:
                        item_text_list.append(text[item_indice])
                    text_list.append(''.join(item_text_list))
                    # text_list.append(item_text_list)
                    item_img_list = []
                    for item_indice in item_indice_list:
                        item_img_list.append(img[item_indice])
                    img_list.append(torch.stack(item_img_list, dim=0))
                    
                    
                shot_coups = list(itertools.permutations(range(len(text_list)), 2))
                sampled_shot_coups = random.sample(shot_coups, min(len(shot_coups), 10))
                for (idx1, idx2) in sampled_shot_coups:
                    # text_tokens1 = []
                    # text_pos1 = []
                    # for text in text_list[idx1]:
                    #     text_pos1.append(len(text_tokens1))
                    #     text_token1 = tokenizer.encode(text)
                    #     text_tokens1 += text_token1
                    
                    # text_tokens2 = []
                    # text_pos2 = []
                    # for text in text_list[idx2]:
                    #     text_pos2.append(len(text_tokens2) + text_pos1[-1])
                    #     text_token2 = tokenizer.encode(text)
                    #     text_tokens2 += text_token2
                    # text_pos1 = torch.LongTensor(text_pos1).to(device)
                    # text_pos2 = torch.LongTensor(text_pos2).to(device)
                    # tokens1 = text_tokens1
                    # tokens2 = text_tokens2
                    tokens1 = tokenizer.encode(text_list[idx1])
                    tokens2 = tokenizer.encode(text_list[idx2])
                    cls1, cls2 = 0, len(tokens1)
                    bert_tokens = tokens1 + tokens2
                    text_ids = torch.LongTensor(bert_tokens).unsqueeze(0).to(device)
                    # text_ids = tokenizer.encode([text[idx1] + '[SEP]' + text[idx2]]).to(device)
                    # text_embeddings = shot_net.embeddings(text_ids)
                    text_embeddings = net.embeddings(text_ids)
                    
                    img_clip1 = shot_vit_embed(img_list[idx1])
                    vis_len1 = img_clip1.size(1)
                    vis_embeddings1 = img_clip1.view(1, vis_len1*img_clip1.size(0), -1)
                    img_clip2 = shot_vit_embed(img_list[idx2])
                    vis_len2 = img_clip2.size(1)
                    vis_embeddings2 = img_clip2.view(1, vis_len2*img_clip2.size(0), -1)
                    
                    bert_embedding = torch.cat((text_embeddings, torch.cat((vis_embeddings1, vis_embeddings2), dim=1)), dim=1)
                    # hidden_states = shot_net(inputs_embeds=bert_embedding)[0]
                    hidden_states = net(inputs_embeds=bert_embedding)[0]
                    output = shot_cls_head(hidden_states, cls1, cls2)
                    # output = shot_cls_head(hidden_states, text_pos1, text_pos2)
                    loss_sample_shot += loss_func(output, torch.tensor([1]).to(device) if idx1 < idx2 else torch.tensor([0]).to(device))
                    
                    PRED = get_order_list(output.detach().reshape(-1).cpu())
                    GT = get_order_list([idx1, idx2])
                    score_sample_shot += int(PRED == GT)
                    
                # # =======> shot level clustering
                shuffled_list = [(idx, img_id[idx], scene_id[idx]) for idx in range(len(img_id))]
                ori_list = sorted(shuffled_list, key=lambda x: x[1])
                # init shot text and image
                n_scene = len(set(scene_id.tolist()))
                indice_dct = collections.defaultdict(list)
                for idx, img_id_idx, scene_id_idx in ori_list:
                    indice_dct[scene_id_idx.item()].append(idx)
                indice_list = [list(indice_dct[x]) for x in indice_dct.keys()]
                
                # # =========> cl loss for cluster
                scene_cluster_sample = 0
                
                for scene_seg_list in indice_list:
                    if len(scene_seg_list) >= 2 and len(indice_list) > 1:
                        pos_list = random.sample(scene_seg_list, 2)
                        anchor_idx, pos_idx = pos_list
                        remain = [x for x in indice_list if x != scene_seg_list]
                        neg_idx = random.sample(random.sample(remain, 1)[0], 1)[0]
                        
                        anchor_tokens = tokenizer.encode(text[anchor_idx], max_length=64, truncation=True, padding='max_length')
                        pos_tokens = tokenizer.encode(text[pos_idx], max_length=64, truncation=True, padding='max_length')
                        neg_tokens = tokenizer.encode(text[neg_idx], max_length=64, truncation=True, padding='max_length')
                        # anchor_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[anchor_idx])).unsqueeze(0).to(device)) 
                        # pos_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[pos_idx])).unsqueeze(0).to(device)) 
                        # neg_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[neg_idx])).unsqueeze(0).to(device)) 
                        text_token_ids = torch.LongTensor([anchor_tokens, pos_tokens, neg_tokens]).to(device)
                        text_embeddings = net.embeddings(text_token_ids)
                        
                        vis_pt = torch.stack((img[anchor_idx], img[pos_idx], img[neg_idx]), dim=0)
                        vis_embeddings = vit_embed(vis_pt)
                        
                        # anchor_embeds = torch.cat((text_embeddings[0].unsqueeze(0), vis_embeddings[0].unsqueeze(0)))
                        # pos_embeds = torch.cat((text_embeddings[1].unsqueeze(0), vis_embeddings[1].unsqueeze(0)))
                        # neg_embeds = torch.cat((text_embeddings[2].unsqueeze(0), vis_embeddings[2].unsqueeze(0)))
                        input_embeddings = torch.cat((text_embeddings, vis_embeddings), dim=1)
                        # input_embeddings = torch.cat((vis_embeddings, text_embeddings), dim=1)
                        hidden_states = net(inputs_embeds=input_embeddings)[0]
                        cluster_output = cluster_head(hidden_states)
                        loss_cluster_scene += cl_loss(cluster_output[0].unsqueeze(0), cluster_output[1].unsqueeze(0), cluster_output[2].unsqueeze(0)).squeeze(0)
                        scene_cluster_sample += 1
                
                
                # if len(indice_list) > 1:
                #     for scene_seg_list in indice_list:
                #         if len(scene_seg_list) >= 2:
                #             cluster_coups = list(itertools.permutations(range(len(scene_seg_list)), 2))
                #             sampled_cluster_coups = random.sample(cluster_coups, min(len(cluster_coups), 10))
                #             for pos_list in sampled_cluster_coups:
                #                 # pos_list = random.sample(shot_seg_list, 2)
                #                 anchor_idx, pos_idx = scene_seg_list[pos_list[0]], scene_seg_list[pos_list[1]]
                #                 remain = [x for x in indice_list if x != scene_seg_list]
                #                 neg_idx = random.sample(random.sample(remain, 1)[0], 1)[0]
                                
                #                 anchor_tokens = tokenizer.encode(text[anchor_idx], max_length=64, truncation=True, padding='max_length')
                #                 pos_tokens = tokenizer.encode(text[pos_idx], max_length=64, truncation=True, padding='max_length')
                #                 neg_tokens = tokenizer.encode(text[neg_idx], max_length=64, truncation=True, padding='max_length')
                #                 # anchor_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[anchor_idx])).unsqueeze(0).to(device)) 
                #                 # pos_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[pos_idx])).unsqueeze(0).to(device)) 
                #                 # neg_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[neg_idx])).unsqueeze(0).to(device)) 
                #                 text_token_ids = torch.LongTensor([anchor_tokens, pos_tokens, neg_tokens]).to(device)
                #                 text_embeddings = net.embeddings(text_token_ids)
                                
                #                 vis_pt = torch.stack((img[anchor_idx], img[pos_idx], img[neg_idx]), dim=0)
                #                 vis_embeddings = vit_embed(vis_pt)
                                
                #                 # anchor_embeds = torch.cat((text_embeddings[0].unsqueeze(0), vis_embeddings[0].unsqueeze(0)))
                #                 # pos_embeds = torch.cat((text_embeddings[1].unsqueeze(0), vis_embeddings[1].unsqueeze(0)))
                #                 # neg_embeds = torch.cat((text_embeddings[2].unsqueeze(0), vis_embeddings[2].unsqueeze(0)))
                #                 input_embeddings = torch.cat((text_embeddings, vis_embeddings), dim=1)
                #                 # input_embeddings = torch.cat((vis_embeddings, text_embeddings), dim=1)
                #                 hidden_states = net(inputs_embeds=input_embeddings)[0]
                #                 cluster_output = shot_cluster_head(hidden_states)
                #                 loss_cluster_scene += cl_loss(cluster_output[0].unsqueeze(0), cluster_output[1].unsqueeze(0), cluster_output[2].unsqueeze(0)).squeeze(0)
                #                 scene_cluster_sample += 1
                #         else:
                #             anchor_idx, pos_idx = scene_seg_list[0], scene_seg_list[0]
                #             remain = [x for x in indice_list if x != scene_seg_list]
                #             neg_idx = random.sample(random.sample(remain, 1)[0], 1)[0]
                            
                #             anchor_tokens = tokenizer.encode(text[anchor_idx], max_length=64, truncation=True, padding='max_length')
                #             pos_tokens = tokenizer.encode(text[pos_idx], max_length=64, truncation=True, padding='max_length')
                #             neg_tokens = tokenizer.encode(text[neg_idx], max_length=64, truncation=True, padding='max_length')
                #             # anchor_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[anchor_idx])).unsqueeze(0).to(device)) 
                #             # pos_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[pos_idx])).unsqueeze(0).to(device)) 
                #             # neg_embeds =  net.embeddings(torch.LongTensor(tokenizer.encode(text[neg_idx])).unsqueeze(0).to(device)) 
                #             text_token_ids = torch.LongTensor([anchor_tokens, pos_tokens, neg_tokens]).to(device)
                #             text_embeddings = net.embeddings(text_token_ids)
                            
                #             vis_pt = torch.stack((img[anchor_idx], img[pos_idx], img[neg_idx]), dim=0)
                #             vis_embeddings = vit_embed(vis_pt)
                            
                #             # anchor_embeds = torch.cat((text_embeddings[0].unsqueeze(0), vis_embeddings[0].unsqueeze(0)))
                #             # pos_embeds = torch.cat((text_embeddings[1].unsqueeze(0), vis_embeddings[1].unsqueeze(0)))
                #             # neg_embeds = torch.cat((text_embeddings[2].unsqueeze(0), vis_embeddings[2].unsqueeze(0)))
                #             input_embeddings = torch.cat((text_embeddings, vis_embeddings), dim=1)
                #             # input_embeddings = torch.cat((vis_embeddings, text_embeddings), dim=1)
                #             hidden_states = net(inputs_embeds=input_embeddings)[0]
                #             cluster_output = shot_cluster_head(hidden_states)
                #             loss_cluster_scene += cl_loss(cluster_output[0].unsqueeze(0), cluster_output[1].unsqueeze(0), cluster_output[2].unsqueeze(0)).squeeze(0)
                #             scene_cluster_sample += 1

                # # ============> scene order
                text_list = []
                img_list = []
                for item_indice_list in indice_list:
                    item_text_list = []
                    for item_indice in item_indice_list:
                        item_text_list.append(text[item_indice])
                    text_list.append(''.join(item_text_list))
                    item_img_list = []
                    if len(item_indice_list) > 3:
                        img_idx_lst = np.linspace(start=0, stop=len(item_indice_list)-1, num=3).astype(int).tolist()
                        new_item_indice_list = [item_indice_list[x] for x in img_idx_lst]
                        item_indice_list = new_item_indice_list
                    for item_indice in item_indice_list:
                        item_img_list.append(img[item_indice])
                    img_list.append(torch.stack(item_img_list, dim=0))
                scene_coups = list(itertools.permutations(range(len(text_list)), 2))
                sampled_scene_coups = random.sample(scene_coups, min(len(scene_coups), 10))
                for (idx1, idx2) in sampled_scene_coups:
                    tokens1 = tokenizer.encode(text_list[idx1], max_length=106, truncation=True, padding='max_length')
                    tokens2 = tokenizer.encode(text_list[idx2], max_length=106, truncation=True, padding='max_length')
                    cls1, cls2 = 0, len(tokens1)
                    bert_tokens = tokens1 + tokens2
                    text_ids = torch.LongTensor(bert_tokens).unsqueeze(0).to(device)
                    # text_ids = tokenizer.encode([text[idx1] + '[SEP]' + text[idx2]]).to(device)
                    # text_embeddings = shot_net.embeddings(text_ids)
                    text_embeddings = net.embeddings(text_ids)
                    
                    img_clip1 = shot_vit_embed(img_list[idx1])
                    vis_len1 = img_clip1.size(1)
                    vis_embeddings1 = img_clip1.view(1, vis_len1*img_clip1.size(0), -1)
                    img_clip2 = shot_vit_embed(img_list[idx2])
                    vis_len2 = img_clip2.size(1)
                    vis_embeddings2 = img_clip2.view(1, vis_len2*img_clip2.size(0), -1)
                    
                    bert_embedding = torch.cat((text_embeddings, torch.cat((vis_embeddings1, vis_embeddings2), dim=1)), dim=1)
                    # hidden_states = shot_net(inputs_embeds=bert_embedding)[0]
                    hidden_states = net(inputs_embeds=bert_embedding)[0]
                    output = scene_cls_head(hidden_states, cls1, cls2)
                    loss_sample_scene += loss_func(output, torch.tensor([1]).to(device) if idx1 < idx2 else torch.tensor([0]).to(device))
                    
                    PRED = get_order_list(output.detach().reshape(-1).cpu())
                    GT = get_order_list([idx1, idx2])
                    score_sample_scene += int(PRED == GT)
                    
                    
                
                
                if scene_cluster_sample == 0:
                    loss_batch_scene_cluster_list.append(0)
                else:
                    loss_batch_scene_cluster_list.append(loss_cluster / scene_cluster_sample)
            
                if cluster_sample == 0:
                    loss_batch_cluster_list.append(0)
                else:
                    loss_batch_cluster_list.append(loss_cluster / cluster_sample)
                loss_batch_list.append(loss_sample / sample_length)
                score_batch_list.append(score_sample/ sample_length)
                loss_batch_shot_list.append(loss_sample_shot/ len(sampled_shot_coups))
                score_batch_shot_list.append(score_sample_shot/ len(sampled_shot_coups))
                
                if len(sampled_scene_coups) == 0:
                    loss_batch_scene_list.append(0)
                    score_batch_scene_list.append(1)  
                else:
                    loss_batch_scene_list.append(loss_sample_scene/ len(sampled_scene_coups))
                    score_batch_scene_list.append(score_sample_scene/ len(sampled_scene_coups))
                
                
            # calcuclate avearge batch
            score_step = sum(score_batch_list) / len(score_batch_list)
            loss_step = sum(loss_batch_list) / len(loss_batch_list)
            loss_cluster_step = sum(loss_batch_cluster_list) / len(loss_batch_cluster_list)
            score_step_shot = sum(score_batch_shot_list) / len(score_batch_shot_list)
            loss_step_shot = sum(loss_batch_shot_list) / len(loss_batch_shot_list)
            loss_cluster_scene_step = sum(loss_batch_scene_cluster_list) / len(loss_batch_scene_cluster_list)
            score_step_scene = sum(score_batch_scene_list) / len(score_batch_scene_list)
            loss_step_scene = sum(loss_batch_scene_list) / len(loss_batch_scene_list)
            
            loss_step += (loss_cluster_step + loss_step_shot + loss_cluster_scene_step + loss_step_scene)
            loss_step.backward()
            optim.step()
            lr_scheduler.step()
            # for param in net.parameters():
            #     param.grad = None
            # caculate avearge score
            score_epoch_list.append(score_step)
            loss_epoch_list.append(float(loss_step))
            loss_epoch_cluster_list.append(float(loss_cluster_step))
            score_epoch_shot_list.append(float(score_step_shot))
            loss_epoch_shot_list.append(float(loss_step_shot))
            
            loss_epoch_scene_cluster_list.append(float(loss_cluster_scene_step))
            score_epoch_scene_list.append(float(score_step_scene))
            loss_epoch_scene_list.append(float(loss_step_scene))
            # wandb.log({'train loss':loss_step.item(), 'train score':score_step})
            tepoch.set_postfix(zlr=lr_scheduler.get_lr()[0], loss=sum(loss_epoch_list)/len(loss_epoch_list), loss_cluster=sum(loss_epoch_cluster_list)/len(loss_epoch_cluster_list), score=sum(score_epoch_list)/len(score_epoch_list), shot_score=sum(score_epoch_shot_list)/len(score_epoch_shot_list), 
                               scene_cluster=sum(loss_epoch_scene_cluster_list)/len(loss_epoch_scene_cluster_list), scene_score=sum(score_epoch_scene_list)/len(score_epoch_scene_list))

    score_epoch = sum(score_epoch_list) / len(score_epoch_list)
    loss_epoch = sum(loss_epoch_list) / len(loss_epoch_list)
    print('train loss = ', loss_epoch, 'train score = ', score_epoch)  

    # val
    with torch.no_grad():
        loss_epoch_list = []
        score_epoch_list = []
        tr_score_epoch_list = []
        du_score_epoch_list = []
        with tqdm(val_dataloader, unit='batch') as tepoch:
            for batch_data in tepoch:
                tepoch.set_description(f'epoch {e}')
                loss_batch_list = []
                score_batch_list = []
                du_batch_bs_epoch_list = []
                tr_batch_bs_epoch_list = []
                
                feature_list = []
                for item_data in batch_data:        
                    text, img, img_id, shot_id, scene_id = item_data # BSZ, LEN, 1024
                    img = torch.load(img)
                    img = img.to(device)
                    
                    # scene level cluster then order
                    n_scene = len(set(scene_id.tolist()))
                    text_tokens = []
                    for txt in text:
                        text_token = tokenizer.encode(txt,
                                                        max_length=64,
                                                        truncation=True,
                                                        padding='max_length')
                        text_tokens.append(text_token)
                    text_token_ids = torch.LongTensor(text_tokens).to(device)
                    text_embeddings = net.embeddings(text_token_ids)
                    vis_embeddings = vit_embed(img)
                    item_embeddings = torch.cat((text_embeddings, vis_embeddings), dim=1)
                    # item_embeddings = torch.cat((vis_embeddings, text_embeddings), dim=1)
                    hidden_states = shot_cluster_head(net(inputs_embeds=item_embeddings)[0])
                    
                    # X = hidden_states[:, 0].cpu()
                    # kmeans = faiss.Kmeans(X.size(-1), n_scene, niter=20, gpu=True)
                    # kmeans.train(X)
                    # _, cluster_output = kmeans.index.search(X, 1)
                    
                    # cluster_output = KMeans(n_clusters=n_scene).fit(hidden_states[:,0].cpu().numpy()).labels_
                    
                    if n_scene == 1:
                        tp = [list(range(len(text)))]
                    else:
                        cluster_output, _ = kmeans(X=hidden_states, num_clusters=n_scene, distance='cosine', device=device, iter_limit=1000, tqdm_flag=False)
                        # cluster_output, _ = kmeans(X=hidden_states, num_clusters=n_scene, distance='soft_dtw', device=device, iter_limit=10)
                        
                        tp = [[] for _ in range(n_scene)]
                        for idx, label in enumerate(cluster_output):
                            tp[label.item()].append(idx)
                        new_tp = []
                        for tt in tp:
                            if tt: new_tp.append(tt)
                        tp = new_tp
                    
                    
                    # # ============ in scene: shot-level cluster then order
                    scene_ordered_tp = []
                    for shot_indice in tp:
                        shot_img = torch.index_select(img, 0, torch.LongTensor(shot_indice).to(device))
                        # shot_img = [img[x] for x in shot_indice]
                        shot_text = [text[x] for x in shot_indice]
                        n_shot = len(set([shot_id[x].item() for x in shot_indice]))
                        
                        text_tokens = []
                        for txt in shot_text:
                            text_token = tokenizer.encode(txt,
                                                            max_length=64,
                                                            truncation=True,
                                                            padding='max_length')
                            text_tokens.append(text_token)
                        text_token_ids = torch.LongTensor(text_tokens).to(device)
                        text_embeddings = net.embeddings(text_token_ids)
                        vis_embeddings = vit_embed(shot_img)
                        item_embeddings = torch.cat((text_embeddings, vis_embeddings), dim=1)
                    
                        hidden_states = cluster_head(net(inputs_embeds=item_embeddings)[0])

                        if n_shot == 1:
                            shot_tp = [shot_indice]
                            
                        else:
                            cluster_output, _ = kmeans(X=hidden_states, num_clusters=n_shot, distance='cosine', device=device, iter_limit=1000, tqdm_flag=False)
                        
                            shot_tp = [[] for _ in range(n_shot)]
                            for idx, label in enumerate(cluster_output):
                                shot_tp[label.item()].append(shot_indice[idx])
                            shot_new_tp = []
                            for tt in shot_tp:
                                if tt: shot_new_tp.append(tt)
                            shot_tp = shot_new_tp
                    
                        # reorder in frame
                        ordered_tp = []
                        for frame_list in shot_tp:
                            
                            if len(frame_list) == 1:
                                ordered_tp.append(frame_list)
                            else:
                                frame_text_lst = []
                                frame_img_lst = []
                                frame_imgid_lst = []
                                for ii in frame_list:
                                    frame_text_lst.append(text[ii])
                                    frame_img_lst.append(img[ii])
                                    frame_imgid_lst.append(img_id[ii])
                                inframe_coups = list(itertools.permutations(range(len(frame_text_lst)), 2))
                                if len(frame_list) == 1:
                                    ordered_tp.append(frame_list)
                                else:
                                    score_dct = {}
                                    for (idx1, idx2) in inframe_coups:
                                        tokens1 = tokenizer.encode(frame_text_lst[idx1])
                                        tokens2 = tokenizer.encode(frame_text_lst[idx2])
                                        cls1, cls2 = 0, len(tokens1)
                                        bert_tokens = tokens1 + tokens2
                                        text_ids = torch.LongTensor(bert_tokens).unsqueeze(0).to(device)
                                        text_embeddings = net.embeddings(text_ids)
                                        vis_pt = torch.stack((frame_img_lst[idx1], frame_img_lst[idx2]), dim=0).to(device)
                                        vis_embeddings = vit_embed(vis_pt)
                                        vis_len = vis_embeddings.size(1)
                                        vis_embeddings = vis_embeddings.view(1, vis_len*2, -1)
                                        bert_embedding = torch.cat((text_embeddings, vis_embeddings), dim=1)
                                        output = net(inputs_embeds=bert_embedding)[0]
                                        output = cls_head(output, cls1, cls2)
                                        score = softmax(output).detach().cpu()[0]
                                        
                                        score_dct[(idx1, idx2)] = score[1]
                                        # score_dct[(idx2, idx1)] = score[0]
                                    
                                    frame_sorted_id = solve(torch.LongTensor(list(range(len(frame_imgid_lst)))), score_dct)
                                    order_frame_text_lst = []
                                    order_frame_img_lst = []
                                    order_frame_imgid_lst = []
                                    order_frame_idx_lst = []
                                    for iii in frame_sorted_id:
                                        order_frame_text_lst.append(frame_text_lst[iii])
                                        order_frame_img_lst.append(frame_img_lst[iii])
                                        order_frame_imgid_lst.append(frame_imgid_lst[iii])
                                        order_frame_idx_lst.append(frame_list[iii])
                                    ordered_tp.append(order_frame_idx_lst)
                        
                        # reorder shot
                        indice_list = ordered_tp
                        text_list = []
                        img_list = []
                        for item_indice_list in indice_list:
                            item_text_list = []
                            for item_indice in item_indice_list:
                                item_text_list.append(text[item_indice])
                            text_list.append(''.join(item_text_list))
                            item_img_list = []
                            if len(item_indice_list) > 3:
                                img_idx_lst = np.linspace(start=0, stop=len(item_indice_list)-1, num=3).astype(int).tolist()
                                new_item_indice_list = [item_indice_list[x] for x in img_idx_lst]
                                item_indice_list = new_item_indice_list
                            for item_indice in item_indice_list:
                                item_img_list.append(img[item_indice])
                            img_list.append(torch.stack(item_img_list, dim=0))
                        shot_coups = list(itertools.permutations(range(len(text_list)), 2))
                        score_dct = {}
                        for (idx1, idx2) in shot_coups:
                            tokens1 = tokenizer.encode(text_list[idx1])
                            tokens2 = tokenizer.encode(text_list[idx2])
                            cls1, cls2 = 0, len(tokens1)
                            bert_tokens = tokens1 + tokens2
                            text_ids = torch.LongTensor(bert_tokens).unsqueeze(0).to(device)
                            # text_ids = tokenizer.encode([text[idx1] + '[SEP]' + text[idx2]]).to(device)
                            # text_embeddings = shot_net.embeddings(text_ids)
                            text_embeddings = net.embeddings(text_ids)
                            
                            img_clip1 = shot_vit_embed(img_list[idx1])[:3]
                            vis_len1 = img_clip1.size(1)
                            vis_embeddings1 = img_clip1.view(1, vis_len1*img_clip1.size(0), -1)
                            img_clip2 = shot_vit_embed(img_list[idx2])[:3]
                            vis_len2 = img_clip2.size(1)
                            vis_embeddings2 = img_clip2.view(1, vis_len2*img_clip2.size(0), -1)
                            
                            bert_embedding = torch.cat((text_embeddings, torch.cat((vis_embeddings1, vis_embeddings2), dim=1)), dim=1)
                            # hidden_states = shot_net(inputs_embeds=bert_embedding)[0]
                            hidden_states = net(inputs_embeds=bert_embedding)[0]
                            output = shot_cls_head(hidden_states, cls1, cls2)
                            score = softmax(output).detach().cpu()[0]
                            
                            score_dct[(idx1, idx2)] = score[1]
                            # score_dct[(idx2, idx1)] = score[0]
                        shot_sorted_id = solve(torch.LongTensor(list(range(len(text_list)))), score_dct)
                        order_shot_lst = []
                        for iii in shot_sorted_id:
                            order_shot_lst += indice_list[iii]
                            
                        scene_ordered_tp.append(order_shot_lst)
                        
                    indice_list = scene_ordered_tp
                    text_list = []
                    img_list = []
                    for item_indice_list in indice_list:
                        item_text_list = []
                        for item_indice in item_indice_list:
                            item_text_list.append(text[item_indice])
                        text_list.append(''.join(item_text_list))
                        item_img_list = []
                        for item_indice in item_indice_list:
                            item_img_list.append(img[item_indice])
                        img_list.append(torch.stack(item_img_list, dim=0))
                    scene_coups = list(itertools.permutations(range(len(text_list)), 2))
                    # scene_coups = list(itertools.combinations(range(len(text_list)), 2))
                    score_dct = {}
                    for (idx1, idx2) in scene_coups:
                        tokens1 = tokenizer.encode(text_list[idx1], max_length=106, truncation=True, padding='max_length')
                        tokens2 = tokenizer.encode(text_list[idx2], max_length=106, truncation=True, padding='max_length')
                        cls1, cls2 = 0, len(tokens1)
                        bert_tokens = tokens1 + tokens2
                        text_ids = torch.LongTensor(bert_tokens).unsqueeze(0).to(device)
                        # text_ids = tokenizer.encode([text[idx1] + '[SEP]' + text[idx2]]).to(device)
                        # text_embeddings = shot_net.embeddings(text_ids)
                        text_embeddings = net.embeddings(text_ids)
                        
                        img_clip1 = shot_vit_embed(img_list[idx1])[:3]
                        vis_len1 = img_clip1.size(1)
                        vis_embeddings1 = img_clip1.view(1, vis_len1*img_clip1.size(0), -1)
                        img_clip2 = shot_vit_embed(img_list[idx2])[:3]
                        vis_len2 = img_clip2.size(1)
                        vis_embeddings2 = img_clip2.view(1, vis_len2*img_clip2.size(0), -1)
                        
                        bert_embedding = torch.cat((text_embeddings, torch.cat((vis_embeddings1, vis_embeddings2), dim=1)), dim=1)
                        # hidden_states = shot_net(inputs_embeds=bert_embedding)[0]
                        hidden_states = net(inputs_embeds=bert_embedding)[0]
                        output = scene_cls_head(hidden_states, cls1, cls2)
                        score = softmax(output).detach().cpu()[0]
                        
                        # score_dct[(idx1, idx2)] = max(1-score_dct.get((idx2, idx1), 1), score[1])
                        score_dct[(idx1, idx2)] = score[1]
                        # score_dct[(idx2, idx1)] = score[0]
                    scene_sorted_id = solve(torch.LongTensor(list(range(len(text_list)))), score_dct)
                    order_scene_lst = []
                    for iii in scene_sorted_id:
                        order_scene_lst += indice_list[iii]
                        
                    sorted_id = []
                    for iii in order_scene_lst: 
                        sorted_id.append(img_id[iii].item())
                    
                

                    
                    
                    
                    
                    du_bs_score = du_metric_func(sorted_id, list(range(len(img_id))))
                    tr_bs_score = tr_metric_func(sorted_id, list(range(len(img_id))))
                    du_batch_bs_epoch_list.append(du_bs_score)
                    tr_batch_bs_epoch_list.append(tr_bs_score)
                
                
                # # calcuclate avearge batch
                # score_step = sum(score_batch_list) / len(score_batch_list)
                # loss_step = sum(loss_batch_list) / len(loss_batch_list)
                
                du_bs_score = sum(du_batch_bs_epoch_list) / len(du_batch_bs_epoch_list)
                tr_bs_score = sum(tr_batch_bs_epoch_list) / len(tr_batch_bs_epoch_list)

                # # caculate avearge score
                # score_epoch_list.append(score_step)
                # loss_epoch_list.append(float(loss_step))
                du_score_epoch_list.append(du_bs_score)
                tr_score_epoch_list.append(tr_bs_score)
                # wandb.log({'val loss':loss_step.item(), 'val score':score_step})
                # tepoch.set_postfix(loss=sum(loss_epoch_list)/len(loss_epoch_list), score=sum(score_epoch_list)/len(score_epoch_list), pair_score=sum(du_score_epoch_list)/len(du_score_epoch_list))
                tepoch.set_postfix(du_score=sum(du_score_epoch_list)/len(du_score_epoch_list))

        # score_epoch = sum(score_epoch_list) / len(score_epoch_list)
        # loss_epoch = sum(loss_epoch_list) / len(loss_epoch_list)
        du_epoch = sum(du_score_epoch_list) / len(du_score_epoch_list)
        tr_epoch = sum(tr_score_epoch_list) / len(tr_score_epoch_list)
        # print('val loss = ', loss_epoch, 'val score = ', score_epoch, 'test_in_domain du = ', du_epoch, 'text_in_domain tr = ', tr_epoch)  
        print('val_in_domain du = ', du_epoch, 'val_in_domain tr = ', tr_epoch)  


# test
split = 'test_in_domain'
test_in_data = NewVideoReorderMovieNetDataFolder(root=data_path, split=split, layer='')
test_in_dataloader = torch.utils.data.DataLoader(test_in_data, batch_size=32, shuffle=(split == 'train'), num_workers=0, pin_memory=True, collate_fn=lambda x: x)

split = 'test_out_domain'
test_out_data = NewVideoReorderMovieNetDataFolder(root=data_path, split=split, layer='')
test_out_dataloader = torch.utils.data.DataLoader(test_out_data, batch_size=32, shuffle=(split == 'train'), num_workers=0, pin_memory=True, collate_fn=lambda x: x)


du_metric_func = DoubleLengthMatching
tr_metric_func = TripleLengthMatching
softmax = nn.Softmax(dim=-1)
# test in domain
with torch.no_grad():
    loss_epoch_list = []
    score_epoch_list = []
    tr_score_epoch_list = []
    du_score_epoch_list = []
    with tqdm(test_in_dataloader, unit='batch') as tepoch:
        for batch_data in tepoch:
            tepoch.set_description(f'inference in domain')
            loss_batch_list = []
            score_batch_list = []
            du_batch_bs_epoch_list = []
            tr_batch_bs_epoch_list = []
            
            feature_list = []
            for item_data in batch_data:        
                text, img, img_id, shot_id, scene_id = item_data # BSZ, LEN, 1024
                img = torch.load(img)
                img = img.to(device)
                
                # scene level cluster then order
                n_scene = len(set(scene_id.tolist()))
                text_tokens = []
                for txt in text:
                    text_token = tokenizer.encode(txt,
                                                    max_length=64,
                                                    truncation=True,
                                                    padding='max_length')
                    text_tokens.append(text_token)
                text_token_ids = torch.LongTensor(text_tokens).to(device)
                text_embeddings = net.embeddings(text_token_ids)
                vis_embeddings = vit_embed(img)
                item_embeddings = torch.cat((text_embeddings, vis_embeddings), dim=1)
                # item_embeddings = torch.cat((vis_embeddings, text_embeddings), dim=1)
                hidden_states = shot_cluster_head(net(inputs_embeds=item_embeddings)[0])
                
                # X = hidden_states[:, 0].cpu()
                # kmeans = faiss.Kmeans(X.size(-1), n_scene, niter=20, gpu=True)
                # kmeans.train(X)
                # _, cluster_output = kmeans.index.search(X, 1)
                
                # cluster_output = KMeans(n_clusters=n_scene).fit(hidden_states[:,0].cpu().numpy()).labels_
                
                if n_scene == 1:
                    tp = [list(range(len(text)))]
                else:
                    cluster_output, _ = kmeans(X=hidden_states, num_clusters=n_scene, distance='cosine', device=device, iter_limit=1000, tqdm_flag=False)
                    # cluster_output, _ = kmeans(X=hidden_states, num_clusters=n_scene, distance='soft_dtw', device=device, iter_limit=10)
                    
                    tp = [[] for _ in range(n_scene)]
                    for idx, label in enumerate(cluster_output):
                        tp[label.item()].append(idx)
                    new_tp = []
                    for tt in tp:
                        if tt: new_tp.append(tt)
                    tp = new_tp
                
                
                # # ============ in scene: shot-level cluster then order
                scene_ordered_tp = []
                for shot_indice in tp:
                    shot_img = torch.index_select(img, 0, torch.LongTensor(shot_indice).to(device))
                    # shot_img = [img[x] for x in shot_indice]
                    shot_text = [text[x] for x in shot_indice]
                    n_shot = len(set([shot_id[x].item() for x in shot_indice]))
                    
                    text_tokens = []
                    for txt in shot_text:
                        text_token = tokenizer.encode(txt,
                                                        max_length=64,
                                                        truncation=True,
                                                        padding='max_length')
                        text_tokens.append(text_token)
                    text_token_ids = torch.LongTensor(text_tokens).to(device)
                    text_embeddings = net.embeddings(text_token_ids)
                    vis_embeddings = vit_embed(shot_img)
                    item_embeddings = torch.cat((text_embeddings, vis_embeddings), dim=1)
                
                    hidden_states = cluster_head(net(inputs_embeds=item_embeddings)[0])

                    if n_shot == 1:
                        shot_tp = [shot_indice]
                        
                    else:
                        cluster_output, _ = kmeans(X=hidden_states, num_clusters=n_shot, distance='cosine', device=device, iter_limit=1000, tqdm_flag=False)
                    
                        shot_tp = [[] for _ in range(n_shot)]
                        for idx, label in enumerate(cluster_output):
                            shot_tp[label.item()].append(shot_indice[idx])
                        shot_new_tp = []
                        for tt in shot_tp:
                            if tt: shot_new_tp.append(tt)
                        shot_tp = shot_new_tp
                
                    # reorder in frame
                    ordered_tp = []
                    for frame_list in shot_tp:
                        
                        if len(frame_list) == 1:
                            ordered_tp.append(frame_list)
                        else:
                            frame_text_lst = []
                            frame_img_lst = []
                            frame_imgid_lst = []
                            for ii in frame_list:
                                frame_text_lst.append(text[ii])
                                frame_img_lst.append(img[ii])
                                frame_imgid_lst.append(img_id[ii])
                            inframe_coups = list(itertools.permutations(range(len(frame_text_lst)), 2))
                            if len(frame_list) == 1:
                                ordered_tp.append(frame_list)
                            else:
                                score_dct = {}
                                for (idx1, idx2) in inframe_coups:
                                    tokens1 = tokenizer.encode(frame_text_lst[idx1])
                                    tokens2 = tokenizer.encode(frame_text_lst[idx2])
                                    cls1, cls2 = 0, len(tokens1)
                                    bert_tokens = tokens1 + tokens2
                                    text_ids = torch.LongTensor(bert_tokens).unsqueeze(0).to(device)
                                    text_embeddings = net.embeddings(text_ids)
                                    vis_pt = torch.stack((frame_img_lst[idx1], frame_img_lst[idx2]), dim=0).to(device)
                                    vis_embeddings = vit_embed(vis_pt)
                                    vis_len = vis_embeddings.size(1)
                                    vis_embeddings = vis_embeddings.view(1, vis_len*2, -1)
                                    bert_embedding = torch.cat((text_embeddings, vis_embeddings), dim=1)
                                    output = net(inputs_embeds=bert_embedding)[0]
                                    output = cls_head(output, cls1, cls2)
                                    score = softmax(output).detach().cpu()[0]
                                    
                                    score_dct[(idx1, idx2)] = score[1]
                                    # score_dct[(idx2, idx1)] = score[0]
                                
                                frame_sorted_id = solve(torch.LongTensor(list(range(len(frame_imgid_lst)))), score_dct)
                                order_frame_text_lst = []
                                order_frame_img_lst = []
                                order_frame_imgid_lst = []
                                order_frame_idx_lst = []
                                for iii in frame_sorted_id:
                                    order_frame_text_lst.append(frame_text_lst[iii])
                                    order_frame_img_lst.append(frame_img_lst[iii])
                                    order_frame_imgid_lst.append(frame_imgid_lst[iii])
                                    order_frame_idx_lst.append(frame_list[iii])
                                ordered_tp.append(order_frame_idx_lst)
                    
                    # reorder shot
                    indice_list = ordered_tp
                    text_list = []
                    img_list = []
                    for item_indice_list in indice_list:
                        item_text_list = []
                        for item_indice in item_indice_list:
                            item_text_list.append(text[item_indice])
                        text_list.append(''.join(item_text_list))
                        item_img_list = []
                        if len(item_indice_list) > 3:
                            img_idx_lst = np.linspace(start=0, stop=len(item_indice_list)-1, num=3).astype(int).tolist()
                            new_item_indice_list = [item_indice_list[x] for x in img_idx_lst]
                            item_indice_list = new_item_indice_list
                        for item_indice in item_indice_list:
                            item_img_list.append(img[item_indice])
                        img_list.append(torch.stack(item_img_list, dim=0))
                    shot_coups = list(itertools.permutations(range(len(text_list)), 2))
                    score_dct = {}
                    for (idx1, idx2) in shot_coups:
                        tokens1 = tokenizer.encode(text_list[idx1])
                        tokens2 = tokenizer.encode(text_list[idx2])
                        cls1, cls2 = 0, len(tokens1)
                        bert_tokens = tokens1 + tokens2
                        text_ids = torch.LongTensor(bert_tokens).unsqueeze(0).to(device)
                        # text_ids = tokenizer.encode([text[idx1] + '[SEP]' + text[idx2]]).to(device)
                        # text_embeddings = shot_net.embeddings(text_ids)
                        text_embeddings = net.embeddings(text_ids)
                        
                        img_clip1 = shot_vit_embed(img_list[idx1])[:3]
                        vis_len1 = img_clip1.size(1)
                        vis_embeddings1 = img_clip1.view(1, vis_len1*img_clip1.size(0), -1)
                        img_clip2 = shot_vit_embed(img_list[idx2])[:3]
                        vis_len2 = img_clip2.size(1)
                        vis_embeddings2 = img_clip2.view(1, vis_len2*img_clip2.size(0), -1)
                        
                        bert_embedding = torch.cat((text_embeddings, torch.cat((vis_embeddings1, vis_embeddings2), dim=1)), dim=1)
                        # hidden_states = shot_net(inputs_embeds=bert_embedding)[0]
                        hidden_states = net(inputs_embeds=bert_embedding)[0]
                        output = shot_cls_head(hidden_states, cls1, cls2)
                        score = softmax(output).detach().cpu()[0]
                        
                        score_dct[(idx1, idx2)] = score[1]
                        # score_dct[(idx2, idx1)] = score[0]
                    shot_sorted_id = solve(torch.LongTensor(list(range(len(text_list)))), score_dct)
                    order_shot_lst = []
                    for iii in shot_sorted_id:
                        order_shot_lst += indice_list[iii]
                        
                    scene_ordered_tp.append(order_shot_lst)
                    
                indice_list = scene_ordered_tp
                text_list = []
                img_list = []
                for item_indice_list in indice_list:
                    item_text_list = []
                    for item_indice in item_indice_list:
                        item_text_list.append(text[item_indice])
                    text_list.append(''.join(item_text_list))
                    item_img_list = []
                    for item_indice in item_indice_list:
                        item_img_list.append(img[item_indice])
                    img_list.append(torch.stack(item_img_list, dim=0))
                scene_coups = list(itertools.permutations(range(len(text_list)), 2))
                # scene_coups = list(itertools.combinations(range(len(text_list)), 2))
                score_dct = {}
                for (idx1, idx2) in scene_coups:
                    tokens1 = tokenizer.encode(text_list[idx1], max_length=106, truncation=True, padding='max_length')
                    tokens2 = tokenizer.encode(text_list[idx2], max_length=106, truncation=True, padding='max_length')
                    cls1, cls2 = 0, len(tokens1)
                    bert_tokens = tokens1 + tokens2
                    text_ids = torch.LongTensor(bert_tokens).unsqueeze(0).to(device)
                    # text_ids = tokenizer.encode([text[idx1] + '[SEP]' + text[idx2]]).to(device)
                    # text_embeddings = shot_net.embeddings(text_ids)
                    text_embeddings = net.embeddings(text_ids)
                    
                    img_clip1 = shot_vit_embed(img_list[idx1])[:3]
                    vis_len1 = img_clip1.size(1)
                    vis_embeddings1 = img_clip1.view(1, vis_len1*img_clip1.size(0), -1)
                    img_clip2 = shot_vit_embed(img_list[idx2])[:3]
                    vis_len2 = img_clip2.size(1)
                    vis_embeddings2 = img_clip2.view(1, vis_len2*img_clip2.size(0), -1)
                    
                    bert_embedding = torch.cat((text_embeddings, torch.cat((vis_embeddings1, vis_embeddings2), dim=1)), dim=1)
                    # hidden_states = shot_net(inputs_embeds=bert_embedding)[0]
                    hidden_states = net(inputs_embeds=bert_embedding)[0]
                    output = scene_cls_head(hidden_states, cls1, cls2)
                    score = softmax(output).detach().cpu()[0]
                    
                    # score_dct[(idx1, idx2)] = max(1-score_dct.get((idx2, idx1), 1), score[1])
                    score_dct[(idx1, idx2)] = score[1]
                    # score_dct[(idx2, idx1)] = score[0]
                scene_sorted_id = solve(torch.LongTensor(list(range(len(text_list)))), score_dct)
                order_scene_lst = []
                for iii in scene_sorted_id:
                    order_scene_lst += indice_list[iii]
                    
                sorted_id = []
                for iii in order_scene_lst: 
                    sorted_id.append(img_id[iii].item())
                
            

                
                
                
                
                du_bs_score = du_metric_func(sorted_id, list(range(len(img_id))))
                tr_bs_score = tr_metric_func(sorted_id, list(range(len(img_id))))
                du_batch_bs_epoch_list.append(du_bs_score)
                tr_batch_bs_epoch_list.append(tr_bs_score)
            
            
            # # calcuclate avearge batch
            # score_step = sum(score_batch_list) / len(score_batch_list)
            # loss_step = sum(loss_batch_list) / len(loss_batch_list)
            
            du_bs_score = sum(du_batch_bs_epoch_list) / len(du_batch_bs_epoch_list)
            tr_bs_score = sum(tr_batch_bs_epoch_list) / len(tr_batch_bs_epoch_list)

            # # caculate avearge score
            # score_epoch_list.append(score_step)
            # loss_epoch_list.append(float(loss_step))
            du_score_epoch_list.append(du_bs_score)
            tr_score_epoch_list.append(tr_bs_score)
            # wandb.log({'val loss':loss_step.item(), 'val score':score_step})
            # tepoch.set_postfix(loss=sum(loss_epoch_list)/len(loss_epoch_list), score=sum(score_epoch_list)/len(score_epoch_list), pair_score=sum(du_score_epoch_list)/len(du_score_epoch_list))
            tepoch.set_postfix(du_score=sum(du_score_epoch_list)/len(du_score_epoch_list))

    # score_epoch = sum(score_epoch_list) / len(score_epoch_list)
    # loss_epoch = sum(loss_epoch_list) / len(loss_epoch_list)
    du_epoch = sum(du_score_epoch_list) / len(du_score_epoch_list)
    tr_epoch = sum(tr_score_epoch_list) / len(tr_score_epoch_list)
    # print('val loss = ', loss_epoch, 'val score = ', score_epoch, 'test_in_domain du = ', du_epoch, 'text_in_domain tr = ', tr_epoch)  
    print('test_in_domain du = ', du_epoch, 'text_in_domain tr = ', tr_epoch)  

# test out domain
with torch.no_grad():
    loss_epoch_list = []
    score_epoch_list = []
    tr_score_epoch_list = []
    du_score_epoch_list = []
    with tqdm(test_out_dataloader, unit='batch') as tepoch:
        for batch_data in tepoch:
            tepoch.set_description(f'inference out domain')
            loss_batch_list = []
            score_batch_list = []
            du_batch_bs_epoch_list = []
            tr_batch_bs_epoch_list = []
            
            feature_list = []
            for item_data in batch_data:        
                text, img, img_id, shot_id, scene_id = item_data # BSZ, LEN, 1024
                img = torch.load(img)
                img = img.to(device)
                
                # scene level cluster then order
                n_scene = len(set(scene_id.tolist()))
                text_tokens = []
                for txt in text:
                    text_token = tokenizer.encode(txt,
                                                    max_length=64,
                                                    truncation=True,
                                                    padding='max_length')
                    text_tokens.append(text_token)
                text_token_ids = torch.LongTensor(text_tokens).to(device)
                text_embeddings = net.embeddings(text_token_ids)
                vis_embeddings = vit_embed(img)
                item_embeddings = torch.cat((text_embeddings, vis_embeddings), dim=1)
                # item_embeddings = torch.cat((vis_embeddings, text_embeddings), dim=1)
                hidden_states = shot_cluster_head(net(inputs_embeds=item_embeddings)[0])
                
                # X = hidden_states[:, 0].cpu()
                # kmeans = faiss.Kmeans(X.size(-1), n_scene, niter=20, gpu=True)
                # kmeans.train(X)
                # _, cluster_output = kmeans.index.search(X, 1)
                
                # cluster_output = KMeans(n_clusters=n_scene).fit(hidden_states[:,0].cpu().numpy()).labels_
                
                if n_scene == 1:
                    tp = [list(range(len(text)))]
                else:
                    cluster_output, _ = kmeans(X=hidden_states, num_clusters=n_scene, distance='cosine', device=device, iter_limit=1000, tqdm_flag=False)
                    # cluster_output, _ = kmeans(X=hidden_states, num_clusters=n_scene, distance='soft_dtw', device=device, iter_limit=10)
                    
                    tp = [[] for _ in range(n_scene)]
                    for idx, label in enumerate(cluster_output):
                        tp[label.item()].append(idx)
                    new_tp = []
                    for tt in tp:
                        if tt: new_tp.append(tt)
                    tp = new_tp
                
                
                # # ============ in scene: shot-level cluster then order
                scene_ordered_tp = []
                for shot_indice in tp:
                    shot_img = torch.index_select(img, 0, torch.LongTensor(shot_indice).to(device))
                    # shot_img = [img[x] for x in shot_indice]
                    shot_text = [text[x] for x in shot_indice]
                    n_shot = len(set([shot_id[x].item() for x in shot_indice]))
                    
                    text_tokens = []
                    for txt in shot_text:
                        text_token = tokenizer.encode(txt,
                                                        max_length=64,
                                                        truncation=True,
                                                        padding='max_length')
                        text_tokens.append(text_token)
                    text_token_ids = torch.LongTensor(text_tokens).to(device)
                    text_embeddings = net.embeddings(text_token_ids)
                    vis_embeddings = vit_embed(shot_img)
                    item_embeddings = torch.cat((text_embeddings, vis_embeddings), dim=1)
                
                    hidden_states = cluster_head(net(inputs_embeds=item_embeddings)[0])

                    if n_shot == 1:
                        shot_tp = [shot_indice]
                        
                    else:
                        cluster_output, _ = kmeans(X=hidden_states, num_clusters=n_shot, distance='cosine', device=device, iter_limit=1000, tqdm_flag=False)
                    
                        shot_tp = [[] for _ in range(n_shot)]
                        for idx, label in enumerate(cluster_output):
                            shot_tp[label.item()].append(shot_indice[idx])
                        shot_new_tp = []
                        for tt in shot_tp:
                            if tt: shot_new_tp.append(tt)
                        shot_tp = shot_new_tp
                
                    # reorder in frame
                    ordered_tp = []
                    for frame_list in shot_tp:
                        
                        if len(frame_list) == 1:
                            ordered_tp.append(frame_list)
                        else:
                            frame_text_lst = []
                            frame_img_lst = []
                            frame_imgid_lst = []
                            for ii in frame_list:
                                frame_text_lst.append(text[ii])
                                frame_img_lst.append(img[ii])
                                frame_imgid_lst.append(img_id[ii])
                            inframe_coups = list(itertools.permutations(range(len(frame_text_lst)), 2))
                            if len(frame_list) == 1:
                                ordered_tp.append(frame_list)
                            else:
                                score_dct = {}
                                for (idx1, idx2) in inframe_coups:
                                    tokens1 = tokenizer.encode(frame_text_lst[idx1])
                                    tokens2 = tokenizer.encode(frame_text_lst[idx2])
                                    cls1, cls2 = 0, len(tokens1)
                                    bert_tokens = tokens1 + tokens2
                                    text_ids = torch.LongTensor(bert_tokens).unsqueeze(0).to(device)
                                    text_embeddings = net.embeddings(text_ids)
                                    vis_pt = torch.stack((frame_img_lst[idx1], frame_img_lst[idx2]), dim=0).to(device)
                                    vis_embeddings = vit_embed(vis_pt)
                                    vis_len = vis_embeddings.size(1)
                                    vis_embeddings = vis_embeddings.view(1, vis_len*2, -1)
                                    bert_embedding = torch.cat((text_embeddings, vis_embeddings), dim=1)
                                    output = net(inputs_embeds=bert_embedding)[0]
                                    output = cls_head(output, cls1, cls2)
                                    score = softmax(output).detach().cpu()[0]
                                    
                                    score_dct[(idx1, idx2)] = score[1]
                                    # score_dct[(idx2, idx1)] = score[0]
                                
                                frame_sorted_id = solve(torch.LongTensor(list(range(len(frame_imgid_lst)))), score_dct)
                                order_frame_text_lst = []
                                order_frame_img_lst = []
                                order_frame_imgid_lst = []
                                order_frame_idx_lst = []
                                for iii in frame_sorted_id:
                                    order_frame_text_lst.append(frame_text_lst[iii])
                                    order_frame_img_lst.append(frame_img_lst[iii])
                                    order_frame_imgid_lst.append(frame_imgid_lst[iii])
                                    order_frame_idx_lst.append(frame_list[iii])
                                ordered_tp.append(order_frame_idx_lst)
                    
                    # reorder shot
                    indice_list = ordered_tp
                    text_list = []
                    img_list = []
                    for item_indice_list in indice_list:
                        item_text_list = []
                        for item_indice in item_indice_list:
                            item_text_list.append(text[item_indice])
                        text_list.append(''.join(item_text_list))
                        item_img_list = []
                        if len(item_indice_list) > 3:
                            img_idx_lst = np.linspace(start=0, stop=len(item_indice_list)-1, num=3).astype(int).tolist()
                            new_item_indice_list = [item_indice_list[x] for x in img_idx_lst]
                            item_indice_list = new_item_indice_list
                        for item_indice in item_indice_list:
                            item_img_list.append(img[item_indice])
                        img_list.append(torch.stack(item_img_list, dim=0))
                    shot_coups = list(itertools.permutations(range(len(text_list)), 2))
                    score_dct = {}
                    for (idx1, idx2) in shot_coups:
                        tokens1 = tokenizer.encode(text_list[idx1])
                        tokens2 = tokenizer.encode(text_list[idx2])
                        cls1, cls2 = 0, len(tokens1)
                        bert_tokens = tokens1 + tokens2
                        text_ids = torch.LongTensor(bert_tokens).unsqueeze(0).to(device)
                        # text_ids = tokenizer.encode([text[idx1] + '[SEP]' + text[idx2]]).to(device)
                        # text_embeddings = shot_net.embeddings(text_ids)
                        text_embeddings = net.embeddings(text_ids)
                        
                        img_clip1 = shot_vit_embed(img_list[idx1])[:3]
                        vis_len1 = img_clip1.size(1)
                        vis_embeddings1 = img_clip1.view(1, vis_len1*img_clip1.size(0), -1)
                        img_clip2 = shot_vit_embed(img_list[idx2])[:3]
                        vis_len2 = img_clip2.size(1)
                        vis_embeddings2 = img_clip2.view(1, vis_len2*img_clip2.size(0), -1)
                        
                        bert_embedding = torch.cat((text_embeddings, torch.cat((vis_embeddings1, vis_embeddings2), dim=1)), dim=1)
                        # hidden_states = shot_net(inputs_embeds=bert_embedding)[0]
                        hidden_states = net(inputs_embeds=bert_embedding)[0]
                        output = shot_cls_head(hidden_states, cls1, cls2)
                        score = softmax(output).detach().cpu()[0]
                        
                        score_dct[(idx1, idx2)] = score[1]
                        # score_dct[(idx2, idx1)] = score[0]
                    shot_sorted_id = solve(torch.LongTensor(list(range(len(text_list)))), score_dct)
                    order_shot_lst = []
                    for iii in shot_sorted_id:
                        order_shot_lst += indice_list[iii]
                        
                    scene_ordered_tp.append(order_shot_lst)
                    
                indice_list = scene_ordered_tp
                text_list = []
                img_list = []
                for item_indice_list in indice_list:
                    item_text_list = []
                    for item_indice in item_indice_list:
                        item_text_list.append(text[item_indice])
                    text_list.append(''.join(item_text_list))
                    item_img_list = []
                    for item_indice in item_indice_list:
                        item_img_list.append(img[item_indice])
                    img_list.append(torch.stack(item_img_list, dim=0))
                scene_coups = list(itertools.permutations(range(len(text_list)), 2))
                # scene_coups = list(itertools.combinations(range(len(text_list)), 2))
                score_dct = {}
                for (idx1, idx2) in scene_coups:
                    tokens1 = tokenizer.encode(text_list[idx1], max_length=106, truncation=True, padding='max_length')
                    tokens2 = tokenizer.encode(text_list[idx2], max_length=106, truncation=True, padding='max_length')
                    cls1, cls2 = 0, len(tokens1)
                    bert_tokens = tokens1 + tokens2
                    text_ids = torch.LongTensor(bert_tokens).unsqueeze(0).to(device)
                    # text_ids = tokenizer.encode([text[idx1] + '[SEP]' + text[idx2]]).to(device)
                    # text_embeddings = shot_net.embeddings(text_ids)
                    text_embeddings = net.embeddings(text_ids)
                    
                    img_clip1 = shot_vit_embed(img_list[idx1])[:3]
                    vis_len1 = img_clip1.size(1)
                    vis_embeddings1 = img_clip1.view(1, vis_len1*img_clip1.size(0), -1)
                    img_clip2 = shot_vit_embed(img_list[idx2])[:3]
                    vis_len2 = img_clip2.size(1)
                    vis_embeddings2 = img_clip2.view(1, vis_len2*img_clip2.size(0), -1)
                    
                    bert_embedding = torch.cat((text_embeddings, torch.cat((vis_embeddings1, vis_embeddings2), dim=1)), dim=1)
                    # hidden_states = shot_net(inputs_embeds=bert_embedding)[0]
                    hidden_states = net(inputs_embeds=bert_embedding)[0]
                    output = scene_cls_head(hidden_states, cls1, cls2)
                    score = softmax(output).detach().cpu()[0]
                    
                    # score_dct[(idx1, idx2)] = max(1-score_dct.get((idx2, idx1), 1), score[1])
                    score_dct[(idx1, idx2)] = score[1]
                    # score_dct[(idx2, idx1)] = score[0]
                scene_sorted_id = solve(torch.LongTensor(list(range(len(text_list)))), score_dct)
                order_scene_lst = []
                for iii in scene_sorted_id:
                    order_scene_lst += indice_list[iii]
                    
                sorted_id = []
                for iii in order_scene_lst: 
                    sorted_id.append(img_id[iii].item())
                
            

                
                
                
                
                du_bs_score = du_metric_func(sorted_id, list(range(len(img_id))))
                tr_bs_score = tr_metric_func(sorted_id, list(range(len(img_id))))
                du_batch_bs_epoch_list.append(du_bs_score)
                tr_batch_bs_epoch_list.append(tr_bs_score)
            
            
            # # calcuclate avearge batch
            # score_step = sum(score_batch_list) / len(score_batch_list)
            # loss_step = sum(loss_batch_list) / len(loss_batch_list)
            
            du_bs_score = sum(du_batch_bs_epoch_list) / len(du_batch_bs_epoch_list)
            tr_bs_score = sum(tr_batch_bs_epoch_list) / len(tr_batch_bs_epoch_list)

            # # caculate avearge score
            # score_epoch_list.append(score_step)
            # loss_epoch_list.append(float(loss_step))
            du_score_epoch_list.append(du_bs_score)
            tr_score_epoch_list.append(tr_bs_score)
            # wandb.log({'val loss':loss_step.item(), 'val score':score_step})
            # tepoch.set_postfix(loss=sum(loss_epoch_list)/len(loss_epoch_list), score=sum(score_epoch_list)/len(score_epoch_list), pair_score=sum(du_score_epoch_list)/len(du_score_epoch_list))
            tepoch.set_postfix(du_score=sum(du_score_epoch_list)/len(du_score_epoch_list))

    # score_epoch = sum(score_epoch_list) / len(score_epoch_list)
    # loss_epoch = sum(loss_epoch_list) / len(loss_epoch_list)
    du_epoch = sum(du_score_epoch_list) / len(du_score_epoch_list)
    tr_epoch = sum(tr_score_epoch_list) / len(tr_score_epoch_list)
    # print('val loss = ', loss_epoch, 'val score = ', score_epoch, 'test_in_domain du = ', du_epoch, 'text_in_domain tr = ', tr_epoch)  
    print('test_out_domain du = ', du_epoch, 'text_out_domain tr = ', tr_epoch)  

