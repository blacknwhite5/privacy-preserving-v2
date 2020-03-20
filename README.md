bbox area : 1  
batch_size = 1  
lr = 0.00005  
alpha = 0.5  
beta = 0.999  
lambda_photo = 100  

[D]  
loss_D_real = loss_BCE(D_real, torch.ones_like(D_real))  
loss_D_fake = loss_BCE(D_fake, torch.zeros_like(D_fake))  
loss_D_cls_real = loss_CE(D_real_cls, cls)  
loss_D = loss_D_real + loss_D_fake + loss_D_cls_real  


[G]  
last_activation = sigmoid  
loss_G_fake = loss_BCE(D_fake, torch.ones_like(D_fake))  
loss_G_cls_fake = -((1/celeba.classes)*torch.ones(1, celeba.classes).to(device) \  
                        * nn.LogSoftmax(dim=1)(D_fake_cls)).sum(dim=1).mean()  
loss_photorealistic = loss_L1(img, fake)  
loss_G = loss_G_fake + loss_G_cls_fake + lambda_photo * loss_photorealistic  
