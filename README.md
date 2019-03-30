# mlpcw4 - Adversarial Transfer Learning: From Accuracy to Adversarial Robustness

### In this work, we try to band together adversarial robustness and transfer learning. We show that vanilla transfer learn-
### ing ca not transfer robustness across domains under white-box attacks. However, it maintains robustness against black box adversaries. We propose a method of injecting adversarial examples during transfer learning, to maintain robustness against both types of attacks. This technique can be beneficial for developing robust models in tasks where transfer learning is applicable. Our method achieves comparable performance to adversarially training from scratch while requiring significantly less computational resources.

* Network Arxhitecture : ResNet, DenseNet
* Attacks: FSGM, PGD, Black-Box attacks
* Transfer Learning Techniques: Weight Sharing, finetuning, 


Github references:

* [PyTorch Implementations of Resnets](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)

* [Resnet trained on CIFAR10](https://github.com/kuangliu/pytorch-cifar)

* [Proper resnet implementation for CIFAR10:](https://github.com/akamaster/pytorch_resnet_cifar10)

* [Adversarial Attacks]( https://github.com/wanglouis49/pytorch-adversarial_box/blob/master/mnist_adv_train.py?fbclid=IwAR33o24Orm2MBaIiErR-hxcr6sZX-XXcOtt72r-hTuo3nYBDdtx6Ng_raOM)

* [Data Loaders, experiment builder](https://github.com/CSTR-Edinburgh/mlpractical/tree/mlp2018-9/mlp_cluster_tutorial)
