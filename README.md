# Image Generation with GAN

## Goal
In this assignment you will be asked to implement a Generative Adversarial Networks (GAN) with [MNIST data set](http://yann.lecun.com/exdb/mnist/). This project will be completed in Python 3 using [Pytorch](https://pytorch.org/tutorials/). 

<img src="https://github.com/yanhuata/DS504CS586-S20/blob/master/project3/pic/goal.png" width="80%">

#### Data set

MNIST is a dataset composed of handwritten numbers and their labels. Each MNIST image is a 28\*28 grey-scale image, which is labeled with an integer value between 0 and 9, corresponding to the actual value in the image. MNIST is provided in Pytorch as 28\*28 matrices containing numbers ranging from 0 to 255. There are 60000 images and labels in the training data set and 10000 images and labels in the test data set. Since this project is an unsupervised learning project, you can only use the 60000 images for your training. 

Download MNIST dataset directly in Pytorch:

```python
from torchvision import datasets

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
```

#### Installing Software and Dependencies 

* [Install Anaconda](https://docs.anaconda.com/anaconda/install/)
* [Create virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
* Install packages (e.g. pip install torch)
