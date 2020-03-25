import torch.nn.functional as F
import torch

def to_crop_resize(img, bbox, resize=128):
    buff = torch.zeros(img.size(0), img.size(1), resize, resize)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = box
        im = img[i,:,y1:y2,x1:x2].unsqueeze(0)
        buff[i] = F.interpolate(im, [resize, resize], mode='bilinear', align_corners=False)
    return buff 

def to_restore_size(img, cropped, bbox):
    buff = torch.zeros_like(img)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = box
        img[i,:,y1:y2,x1:x2] = F.interpolate(cropped[i].unsqueeze(0), [y2-y1, x2-x1], mode='bilinear')[0]
    return img


def to_expand_bbox(bbox, landm, h, w): 
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
    expanded_ = torch.cat([x1, y1, x2, y2], dim=1)+ (gap * 0.8).to(bbox.dtype)

    expanded_[:,0] = torch.where(expanded_[:,0] > 0, expanded_[:,0], 0*torch.ones_like(expanded_[:,0]))
    expanded_[:,1] = torch.where(expanded_[:,1] > 0, expanded_[:,1], 0*torch.ones_like(expanded_[:,1]))
    expanded_[:,2] = torch.where(expanded_[:,2] < w, expanded_[:,2], w*torch.ones_like(expanded_[:,2]))
    expanded_[:,3] = torch.where(expanded_[:,3] < h, expanded_[:,3], h*torch.ones_like(expanded_[:,3]))

    return expanded_

def to_apply_mask(img, bbox):
    """mask bbox of image"""
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = box
        img[i][:,y1:y2,x1:x2] = 2*(128/255)-1
    return img

def softmax_(fake_cls):
    return torch.nn.functional.softmax(fake_cls.detach(), dim=1)

def correct_(real_cls, predicted):
    argmax = torch.argmax(predicted, dim=1)
    correct_ = torch.where(argmax==real_cls, torch.ones_like(argmax), torch.zeros_like(argmax))
    return torch.sum(correct_) / len(real_cls)
