# Autoencoder based Image Compression

The aim of this project is to build on current works in lossy Compressive Autoencoders (CAE).

The file 'Simple CAE MNIST.py' translates the work found from [1] into PyTorch and modifies the model, mainly a convolutional layer, to improve performance.

## Results
The orginal code from [1] using a stored format of 16 neurons yielded the following results:

![alt text](https://github.com/jackgrebenc/Autoencoder-based-Image-Compression/blob/main/Output%20of%20original%20model.png)

The modified model had the following results for 16 neurons:

![alt text](https://github.com/jackgrebenc/Autoencoder-based-Image-Compression/blob/main/Output%20Images/Output%20of%20modified%20model.png)

## References
[1] https://blog.paperspace.com/autoencoder-image-compression-keras/ 

