# Autoencoder based Image Compression

The aim of this project is to build on current works in lossy Compressive Autoencoders (CAE).

The file 'Simple CAE MNIST.py' translates the work found from [1] into PyTorch and modifies the model, mainly a convolutional layer, to improve performance.

## Results
The orginal code from [1] using a stored format of 16 bits (rather than 2 bits which was used in the post) yielded the following results:

The modified model had the following results for 16 bits:

## References
[1] https://blog.paperspace.com/autoencoder-image-compression-keras/ 

