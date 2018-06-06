# NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications
https://arxiv.org/abs/1804.03230

This is our Implementation of NetAdapt.

## useage
- python netadapt.py pretrain ... Training and saving the MobileNet will start.
- python netadapt.py maketable ... Calculating and saving the lookup-table will start.
- python netadapt.py search ... The main pruning procedure will start.

The adapted model will be outputted to "./adapted_model.pickle".
