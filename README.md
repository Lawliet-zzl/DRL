# Code for "Out-of-Distribution Detection by Exploiting Complementary Information"

## requirement
* Python 3.7
* Pytorch 1.1
* torchvision 0.3
* scikit-learn
* tqdm
* pandas
* scipy

## Datasets
### In-distribution Datasets
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

Our codes will download the in-distribution dataset automatically.

### Out-of-Distribtion Datasets
* [CIFAR100](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
* [CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
* [StanfordDogs120](http://vision.stanford.edu/aditya86/ImageNetDogs/)
* [OxfordPets37](https://www.robots.ox.ac.uk/~vgg/data/pets/)
* [Oxfordflowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/)
* [Caltech256](https://www.kaggle.com/jessicali9530/caltech256)
* [DTD47](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
* [COCO](http://images.cocodataset.org/zips/val2017.zip)

Each out-of-distribution dataset should be put in the corresponding subdir in [./data](./data)

## Train and Test
Run the script [demo.sh](./code/demo.sh). 
