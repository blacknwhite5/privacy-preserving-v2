### ver1: can+inpaint

```
initialize in bbox area : 1  
batch_size = 1  
lr = 0.00005  
alpha = 0.5  
beta = 0.999  
lambda_photo = 100  

[D]  
loss = BCE + CE

[G]  
last_activation = none 
loss = BCE + CAN + lambda_photo * L1
```

### ver2: can+mse+ce

```
initialize in bbox area : (128/255)*2 -1 
batch_size = 16
lr = 0.0005
alpha = 0.0
beta = 0.999
lambda_photo = 100

[D]  
loss = MSE + CE

[G]  
last_activation = none 
loss = MSE + CAN + lambda_photo * L1
```
 - 학습이 오래 진행될수록 G의 성능이 떨어지는 경향을 보임.
 - 15에폭에서 가장 좋은 퍼포먼스를 보임.
