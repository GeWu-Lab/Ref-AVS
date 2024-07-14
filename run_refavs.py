import sys
import os
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import threshold, normalize
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from configs import args
from datasets import REFAVS
from models import REFAVS_Model_Base

from scripts.train import train, test
from logs.write_log import write_log 



def run(model):
    train_dataset = REFAVS('train', args)
    val_dataset = REFAVS('val', args) 
    test_dataset_s = REFAVS('test_s', args)  # seen
    test_dataset_u = REFAVS('test_u', args)  # unseen
    test_dataset_n = REFAVS('test_n', args)  # null

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=False, collate_fn=collate_fn)
    if args.val == 'val':
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_fn)
    elif args.val == 'test_s':
        val_loader = DataLoader(test_dataset_s, batch_size=4, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_fn)
    elif args.val == 'test_u':
        val_loader = DataLoader(test_dataset_u, batch_size=4, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_fn)
    elif args.val == 'test_n':
        val_loader = DataLoader(test_dataset_n, batch_size=4, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_fn)
    
  
    tuned_num = 0
    for name, param in model.named_parameters():
        param.requires_grad = False
        for _n in args.train_params: 
            if _n in name:
                # print('yes:', _n, name)
                param.requires_grad = True  # finetune
                tuned_num += 1 
    
    if args.show_params:
        print('>>> check params with grad:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("- Requires_grad:", name)

    message = f'All: {sum(p.numel() for p in model.parameters()) / 1e6}M\n'
    message += f'Train-able: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M\n'
    print(message)

    # optimizer
    params1 = [{'params': [p for name, p in model.named_parameters() if p.requires_grad], 'lr': args.lr}]
    params = params1 
    optimizer = torch.optim.AdamW(params)

    train_losses = []
    m_s, f_s, null_s = [], [], []  # miou, f1, metric_s for null
    max_miou = 0
    
    # model
    model = model.cuda()
    for idx_ep in range(args.epochs):
        print(f'[Epoch] {idx_ep}')
        currentDateAndTime = datetime.now().strftime("%y%m%d_%H_%M_%S_%f")
        
        if args.train:
            model.train()
            loss_train = train(model, train_loader, optimizer, idx_ep, args)
            train_losses.append(loss_train)
        
        if args.val:
            model.eval()
            m, f = test(model, val_loader, optimizer, idx_ep, args)
            m_s.append(m)
            f_s.append(f)
            
            print(m, currentDateAndTime)
            ckpt_save_path = f"{args.save_ckpt}/ckpt_best_miou.pth"
            
            with open(args.log_path, 'a') as f:
                f.write(f"Epoch: {idx_ep}: {m_s} | {f_s}\n")

            if m >= max_miou and args.val == 'val':
                max_miou = m
                torch.save(model.state_dict(), ckpt_save_path)
                print(f'>>> saved ckpt at {ckpt_save_path} with miou={max_miou}')
                with open(args.log_path, 'a') as f:
                    f.write(f"Best miou at epoch: {idx_ep}: {max_miou}. Saved at {ckpt_save_path}.\n")
        
        print(f'train-losses: {train_losses} | miou: {m_s} | f-score{f_s}')

def collate_fn(batch):
    img_recs = []
    mask_recs = []
    image_sizes = []
    uids = []
    
    audio_feats = []
    text_feats = []
    audio_recs = []
    text_recs = []

    for data in batch:
        uids.append(data[0])
        mask_recs.append(data[1])
        img_recs.append(data[2])
        image_sizes.append(data[3])
        audio_feats.append(data[4])
        text_feats.append(data[5])
        audio_recs.append(data[6])
        text_recs.append(data[7])

    return uids, mask_recs, img_recs, image_sizes, audio_feats, text_feats, audio_recs, text_recs

if __name__ == '__main__':
    print(vars(args))
    m2f_avs = REFAVS_Model_Base(cfgs=args)

    if str(args.val).startswith('test'):
        ckpt = args.checkpoint

        print('>>> load ckpt from:', ckpt)
        m2f_avs.load_state_dict(torch.load(ckpt), strict=True)

    run(m2f_avs)
