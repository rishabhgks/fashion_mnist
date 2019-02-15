# fashion_mnist
An implementation of Adrian Rosebrock's blog on running a CNN through the MNIST Fashion Dataset. Model used is MiniVGGNet

On training the model with 2 convolution set of layers (CONV -> RELU -> CONV -> RELU -> POOL) and filters of 32 and 64, the accuracy was around 94% on the testing set.
On Saving, the model size was 13.5 mb
However, when i trained the model with 3 convolution set of layers and filters of 32, 64 and 128 respectively for each layer, the accuracy got down to 93% on the testing set.
Also, on saving the model size was 7.5mb
