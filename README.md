# NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications
https://arxiv.org/abs/1804.03230

This is our implementation of NetAdapt.

## useage
- python netadapt.py pretrain ... Training and saving the MobileNet will start.
- python netadapt.py maketable ... Calculating and saving the lookup-table will start.
- python netadapt.py search ... The main pruning procedure will start.

The adapted model will be outputted to "./adapted_model.pickle".

# experimental result
- GTX1080Ti MobileNet CIFAR-10 compression-rate=0.9
layer2, numfilter=34
layer3, numfilter=49
layer4, numfilter=91
layer5, numfilter=144
layer6, numfilter=256
layer7, numfilter=447
layer8, numfilter=482
layer9, numfilter=468
layer10, numfilter=436
layer11, numfilter=471
layer12, numfilter=512
layer13, numfilter=1024
reduction = 0.105497025671
test accuracy = 0.811116  
base_time = 0.13710842136667933  
adapted_time = 0.0943378491526019  
(both batchsize = 128)  
exact-latency = 0.6880529161684906%  
lookup-table is so poor approximation

- Core i5 MobileNet CIFAR-10 compression-rate=0.2
layer2, numfilter=1
layer3, numfilter=1
layer4, numfilter=1
layer5, numfilter=94
layer6, numfilter=76
layer7, numfilter=249
layer8, numfilter=280
layer9, numfilter=251
layer10, numfilter=324
layer11, numfilter=280
layer12, numfilter=1
layer13, numfilter=1024
reduction = 0.781648723048
test accuracy = 0.698279


