# transfer-learning
In this short tutorial, I showed how to conduct a deep learning project with transfer learning, which is generally better than building our own model from scratch. The data used in this tutorial is from a Kaggle competition "Dogs vs. Cats" and "dogs vs cats vs horses vs human". In which, I used a pre-trained model VGG16, you can use other pre-trained models from Keras.application, the process will be the same.

Here is why transfer learning is what one should consider before building a brand new model:
1. the lower layers in a deep learning model, especially the convolution models, are learning general features, which are more likely to be problem independent.While the higher layers may refer to specific features depending on the problem. We just need to freeze its lower layers, ditch its final fewer layers, and add our own output layer.
2. Using a pre-trained model can not only shorten the training time but also improve accuracy.

In this short tutorial, I provided two different ways to implement transfer learning in transfer_learning_2_methods.py: The first method is very primitive using direct propagation, passing the image data into the first layer and passing its output to the next layer, and so much so until the last layer;The second method would look much elegant than the first method by creating a pipeline. It starts by creating a Sequential model, then adds an Input layer, then a VGG16 base model, and custom layers to the model sequentially.

In the folder dog_vs_cat, you can find the example of adapting VGG16 into a binary classifier with a fully connected layer or a global average pooling layer; similar in the dog_cat_horse_human, you can find the example of adapting VGG16 into 4 classes classifier. 
