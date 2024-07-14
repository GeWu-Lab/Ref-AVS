
import torch
import numpy as np
from utils import pyutils
from utils import utility 

avg_meter_miou = pyutils.AverageMeter('miou')
avg_meter_F = pyutils.AverageMeter('F_score')

def train(model, train_loader, optimizer, idx_ep, args):
    print('>>> Train start ...')
    model.train()
    
    losses = []
    
    for batch_idx, batch_data in enumerate(train_loader):
        loss_vid, _ = model(batch_data)
        loss_vid = torch.mean(torch.stack(loss_vid))

        optimizer.zero_grad()
        loss_vid.backward()
        optimizer.step()
        
        losses.append(loss_vid.item())
        print(f'[tr] loss_{idx_ep}_{batch_idx}/{len(train_loader.dataset)//train_loader.batch_size}: {loss_vid.item()} | mean_loss: {np.mean(losses)}', end='\r')
    
    return np.mean(losses)
    
def test(model, test_loader, optimizer, idx_ep, args):
    model.eval()
    
    null_s_list = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            uid, mask_recs, img_recs, image_sizes, feat_aud, feat_text, rec_audio, rec_text = batch_data
            _, vid_preds = model(batch_data)
            mask_recs = [torch.stack(mask_rec, dim=0) for mask_rec in mask_recs]
            vid_preds_t = torch.stack(vid_preds, dim=0).squeeze().cuda().view(-1, 1, 256, 256) 
            vid_masks_t = torch.stack(mask_recs, dim=0).squeeze().cuda().view(-1, 1, 256, 256) 

            if args.val == 'test_n':
                null_s = utility.metric_s_for_null(vid_preds_t)
                null_s_list.append(null_s.cpu().numpy())
                print(f'[te] loss_{idx_ep}_{batch_idx}/{len(test_loader.dataset)//test_loader.batch_size}: s={null_s} | mean={np.mean(np.array(null_s_list))}  ')

            else:
                miou = utility.mask_iou(vid_preds_t, vid_masks_t)  
                avg_meter_miou.add({'miou': miou})

                F_score = utility.Eval_Fmeasure(vid_preds_t, vid_masks_t, './logger', device=f'cuda:{args.gpu_id}')  
                avg_meter_F.add({'F_score': F_score})
                
                print(f'[te] loss_{idx_ep}_{batch_idx}/{len(test_loader.dataset)//test_loader.batch_size}: miou={miou:.03f} | F={F_score:.03f} |  ', end='\r')
    
    if args.val == 'test_n':
        miou_epoch = np.mean(np.array(null_s_list))
        F_epoch = miou_epoch  # fake name, just for null_s
    else:
        miou_epoch = (avg_meter_miou.pop('miou')).item()
        F_epoch = (avg_meter_F.pop('F_score'))
    
    return miou_epoch, F_epoch

