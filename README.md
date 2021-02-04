# Code for "Out-of-Distribution Samples Have Weak Representation Couplings"

## requirement
* Python 3.7
* Pytorch 1.1
* scikit-learn
* tqdm
* pandas
* scipy

## Datasets
### In-distribution Datasets
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)

Our codes will download the two in-distribution datasets automatically.

### Out-of-Distribtion Datasets
The following four out-of-distribution datasets are provided by [ODIN](https://github.com/ShiyuLiang/odin-pytorch)
* [TinyImageNet (r)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
* [TinyImageNet (c)](https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz)
* [LSUN (r)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)
* [LSUN (c)](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)

Each out-of-distribution dataset should be put in the corresponding subdir in [./data_RCL](./data_RCL)

## Train and Test
Run the script [demo.sh](./code_RCL/demo.sh). 
