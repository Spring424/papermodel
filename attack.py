import torch
import torch.nn as nn
from parse import args
from client import FedRecClient
import numpy as np
import random


class BaselineAttackClient(FedRecClient):
    def __init__(self, train_ind, m_item, dim):
        super().__init__(train_ind, [], [], m_item, dim)

    def train_(self, items_emb, linear_layers):
        a, b, c, _ = super().train_(items_emb, linear_layers)
        return a, b, c, None

    def eval_(self, _items_emb, _linear_layers):
        return None, None


class AttackClient(nn.Module):
    def __init__(self, target_items, m_item, dim ):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)
        self.DNA_SIZE = self.dim
        self.pre_item_emb = None
        self.flag = 1
        self.id=random.randint(0, 20)
        self.item_rec=[]
        self.pre_linear_layers = None
        self.label_x = random.randint(0, 1)
        self.label_y = torch.full([10], float(self.label_x))


    def forward(self, user_emb, items_emb, linear_layers):
        user_emb = user_emb.repeat(len(items_emb), 1)
        v = torch.cat((user_emb, items_emb), dim=-1)

        for i, (w, b) in enumerate(linear_layers):
            v = v @ w.t() + b
            if i < len(linear_layers) - 1:
                v = v.relu()
            else:
                v = v.sigmoid()
        return v.view(-1)

    def train_on_user_emb(self, user_emb, items_emb, linear_layers):
        predictions = self.forward(user_emb, items_emb, linear_layers)
        loss = nn.BCELoss()(predictions, torch.ones(len(self._target_)).to(args.device))
        return loss

    def simLoss(self, tar_emb, item_emb):

        cosine_embedding_loss = torch.nn.CosineEmbeddingLoss(margin=0.5, reduction='mean')
        label = torch.Tensor([1]).to(args.device)
        loss = cosine_embedding_loss(tar_emb, item_emb, label)
        # print("loss",loss)
        x = torch.reshape(tar_emb, [8])
        y = torch.reshape(item_emb, [8])
        sim = torch.cosine_similarity(x, y, dim=0)
        # print("sim",sim)
        return loss

    def distance(self, item_emb):
        item_sim = torch.Tensor([])
        if self.pre_item_emb is not None:
            for item in range(len(item_emb)):
                # dis = torch.sum(torch.abs(item_emb[item] - self.pre_item_emb[item]), dim=0)
                # dis = torch.cosine_similarity(item_emb[item], self.pre_item_emb[item], dim=0)
                pdist = nn.PairwiseDistance(p=2)
                a = item_emb[item].reshape([1, args.dim])
                b = self.pre_item_emb[item].reshape([1, args.dim])
                dis = pdist(a, b)
                dis_tensor = torch.Tensor([dis])
                if item is 0:
                    item_sim = dis_tensor
                    continue
                item_sim = torch.cat((item_sim, dis_tensor), 0)
        self.pre_item_emb = item_emb.clone()

        return item_sim



    def train_(self, items_emb, linear_layers):
        linear_layers_grad = [[0, 0] for (w, b) in linear_layers]
        poison_grad = 0
        max_item = self.distance(items_emb)
        if max_item.numel() is 0:
            return 0,self._target_, poison_grad, linear_layers_grad, None
        _,top100=max_item.topk(100)
        if self.flag==1:
            _, item_rec = max_item.topk(100)
            self.item_rec = item_rec.cpu().tolist()
            self.flag = 0

        random_item_rec = random.sample(self.item_rec,10)
        x = np.random.choice([1, 0], size=1, p=[0.2, 0.8])
        y = torch.full([10], float(x))
        target_emb = items_emb[self._target_].clone().detach().requires_grad_(True)
        item_emb = items_emb[random_item_rec].clone().detach().requires_grad_(True)
        linear_layers = [(w.clone().detach().requires_grad_(True),
                          b.clone().detach().requires_grad_(True))
                         for (w, b) in linear_layers]

        nn.init.normal_(self._user_emb.weight, std=0.01)
        for i in range(30):
            predictions = self.forward(self._user_emb.weight.requires_grad_(True),
                                       item_emb, linear_layers)
            loss = nn.BCELoss()(predictions,y.to(args.device))
            self._user_emb.zero_grad()
            loss.backward()
            self._user_emb.weight.data.add_(self._user_emb.weight.grad, alpha=-args.lr)
        poison_loss = self.train_on_user_emb(self._user_emb.weight, target_emb, linear_layers)

        poison_loss.backward()
        poison_grad = target_emb.grad


        linear_layers_grad = [[w.grad, b.grad] for (w, b) in linear_layers]


        return 1, self._target_, poison_grad, linear_layers_grad, None


    def eval_(self, _items_emb, _linear_layers):
        return None, None
