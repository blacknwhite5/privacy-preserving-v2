import torch.nn.functional as F
import torch

import numpy as np

def to_crop_resize(img, bbox, landm, resize=128):
    buff = torch.zeros(img.size(0), img.size(1), resize, resize)
    landm_buff = torch.zeros_like(landm)
    for i, (box, mark) in enumerate(zip(bbox, landm)):
        # image cropping and resizing
        x1, y1, x2, y2 = box 
        im = img[i,:,y1:y2,x1:x2].unsqueeze(0)
        buff[i] = F.interpolate(im, [resize, resize], mode='bilinear', align_corners=False) 

        # landmark rescaling
        h, w = im.shape[2:]
        y_scale, x_scale = resize / h, resize / w 
        mark = mark.to(torch.float64)
        mark[::2] = (mark[::2]-x1) * x_scale
        mark[1::2] = (mark[1::2]-y1) * y_scale
        landm_buff[i] = mark.to(torch.int64)
    return buff, landm_buff

def to_restore_size(img, cropped, bbox):
    buff = torch.zeros_like(img)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = box
        img[i,:,y1:y2,x1:x2] = F.interpolate(cropped[i].unsqueeze(0), 
                                             [y2-y1, x2-x1], mode='bilinear', align_corners=False)[0]
    return img


def to_expand_bbox(bbox, landm, h, w, percentage=0.05): 
    shrink_percentage = np.random.uniform(-percentage, percentage)

    pnts_x = landm[:,0::2]
    pnts_y = landm[:,1::2]
    max_x = pnts_x.max(1)[0]
    min_x = pnts_x.min(1)[0]
    max_y = pnts_y.max(1)[0]
    min_y = pnts_y.min(1)[0]
    minmax_xy = torch.stack([min_x, min_y, max_x, max_y])
    minmax_xy = minmax_xy.t()

    gap = (bbox - minmax_xy).to(torch.float64).to(bbox.device)

    x1 = torch.stack([bbox[:,0], min_x]).t().min(1, keepdim=True)[0]
    y1 = torch.stack([bbox[:,1], min_y]).t().min(1, keepdim=True)[0]
    x2 = torch.stack([bbox[:,2], max_x]).t().max(1, keepdim=True)[0]
    y2 = torch.stack([bbox[:,3], max_y]).t().max(1, keepdim=True)[0]
    expanded_ = torch.cat([x1, y1, x2, y2], dim=1)+ (gap * (0.8+shrink_percentage)).to(bbox.dtype)
#    expanded_ = torch.cat([x1, y1, x2, y2], dim=1)+ (gap * 0.8).to(bbox.dtype)

    expanded_[:,0] = torch.where(expanded_[:,0] > 0, expanded_[:,0], 0*torch.ones_like(expanded_[:,0]))
    expanded_[:,1] = torch.where(expanded_[:,1] > 0, expanded_[:,1], 0*torch.ones_like(expanded_[:,1]))
    expanded_[:,2] = torch.where(expanded_[:,2] < w, expanded_[:,2], w*torch.ones_like(expanded_[:,2]))
    expanded_[:,3] = torch.where(expanded_[:,3] < h, expanded_[:,3], h*torch.ones_like(expanded_[:,3]))

    return expanded_

def to_apply_mask(img, bbox):
    """mask bbox of image"""
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = box
#        img[i][:,y1:y2,x1:x2] = 2*(128/255)-1
#        noise = np.random.normal(0.0, 0.1)
#        img[i][:,y1:y2,x1:x2] = torch.from_numpy(noise) 
        img[i][:,y1:y2,x1:x2] = img[i][:,y1:y2,x1:x2].normal_(0.0, 0.1) 
    return img

def to_onehot(landm, h, w):
    onehot = torch.zeros(landm.shape[0], 1, h, w)
    for i, mark in enumerate(landm):
        mark = torch.where(mark > 127, torch.tensor(127).to(mark.device), mark)
        onehot[i,0,mark[0],mark[1]] = 1
        onehot[i,0,mark[2],mark[3]] = 1
        onehot[i,0,mark[4],mark[5]] = 1
        onehot[i,0,mark[6],mark[7]] = 1
        onehot[i,0,mark[8],mark[9]] = 1
    return onehot

def softmax_(fake_cls):
    return torch.nn.functional.softmax(fake_cls.detach(), dim=1)

def accuracy_(real_cls, predicted):
    argmax = torch.argmax(predicted, dim=1)
    correct_ = torch.where(argmax==real_cls, torch.ones_like(argmax), torch.zeros_like(argmax)).to(predicted.dtype)
    return torch.sum(correct_) / len(real_cls)


# Load fewer layers of pre-trained models if possible
def load(model, cpk_file, netname=None):
    if netname is not None:
        pretrained_dict = torch.load(cpk_file)[netname]
    else:
        pretrained_dict = torch.load(cpk_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
