import torch.nn.functional as F
import torch

import numpy as np

def to_crop_resize(img, bbox, mark, resize=128):
    # image cropping and resizing
    x1, y1, x2, y2 = bbox 
    im = img[:,y1:y2,x1:x2].unsqueeze(0)
    buff = F.interpolate(im, [resize, resize], mode='bilinear', align_corners=False)[0]

    # landmark rescaling
    h, w = im.shape[2:]
    y_scale, x_scale = resize / h, resize / w 
    mark = mark.to(torch.float64)
    mark[::2] = (mark[::2]-x1) * x_scale
    mark[1::2] = (mark[1::2]-y1) * y_scale
    landm_buff = mark.to(torch.int64)
    return buff, landm_buff

def to_restore_size(img, cropped, bbox):
    buff = torch.zeros_like(img)
    x1, y1, x2, y2 = bbox
    img[i,y1:y2,x1:x2] = F.interpolate(cropped[i].unsqueeze(0), 
                                         [y2-y1, x2-x1], mode='linear')[0]
    return img


def to_expand_bbox(bbox, landm, h, w, percentage=0.05): 
    shrink_percentage = np.random.uniform(-percentage, percentage)

    pnts_x = landm[0::2]
    pnts_y = landm[1::2]
    max_x = pnts_x.max()
    min_x = pnts_x.min()
    max_y = pnts_y.max()
    min_y = pnts_y.min()
    minmax_xy = torch.stack([min_x, min_y, max_x, max_y])
    minmax_xy = minmax_xy.t()

    gap = (bbox - minmax_xy).to(torch.float64).to(bbox.device)

    x1 = torch.stack([bbox[0], min_x]).t().min()
    y1 = torch.stack([bbox[1], min_y]).t().min()
    x2 = torch.stack([bbox[2], max_x]).t().max()
    y2 = torch.stack([bbox[3], max_y]).t().max()
#    expanded_ = torch.cat([x1, y1, x2, y2], dim=1)+ (gap * (0.5+shrink_percentage)).to(bbox.dtype)
#    expanded_ = torch.cat([x1, y1, x2, y2], dim=1)+ (gap * 0.8).to(bbox.dtype)
    expanded_ = torch.stack([x1, y1, x2, y2])+ (gap * 0.3).to(bbox.dtype)

    expanded_[0] = torch.where(expanded_[0] > 0, expanded_[0], 0*torch.ones_like(expanded_[0]))
    expanded_[1] = torch.where(expanded_[1] > 0, expanded_[1], 0*torch.ones_like(expanded_[1]))
    expanded_[2] = torch.where(expanded_[2] < w, expanded_[2], w*torch.ones_like(expanded_[2]))
    expanded_[3] = torch.where(expanded_[3] < h, expanded_[3], h*torch.ones_like(expanded_[3]))

    return expanded_

def to_apply_mask(img, bbox):
    """mask bbox of image"""
    x1, y1, x2, y2 = bbox
    img[:,y1:y2,x1:x2] = img[:,y1:y2,x1:x2].normal_(0.0, 0.1) 
    return img

def to_onehot(mark, h, w):
    onehot = torch.zeros(1, h, w)
    mark = torch.where(mark > 127, torch.tensor(127).to(mark.device), mark)
    onehot[0,mark[0],mark[1]] = 1
    onehot[0,mark[2],mark[3]] = 1
    onehot[0,mark[4],mark[5]] = 1
    onehot[0,mark[6],mark[7]] = 1
    onehot[0,mark[8],mark[9]] = 1
    return onehot
