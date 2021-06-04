<h1 align="center"> Image segmentation using ML tools </h1>
<h5 align="center">A project of evaluation of performance of ML techniques for the semantic segmentation of images</h5>

</p>
<p align="center">
<img src ="./media/readme/MACHINE LEARNING.gif" alt="Logo" width="1200"/>
</p>

Considering the impact on the world of artificial intelligence and process automation in recent years, semi-autonomous systems have been developed that act responding to signals from their environment. An example is food delivery robots that establish their trajectory from images of their surroundings. To develop a system like this it is necessary to make use of segmentation algorithms trained to recognize obstacles, and to train these algorithms, databases of segmented images corresponding to the environment in which the robot will operate are required.

To facilitate the image segmentation process, the project based on ML techniques available in this repository was developed.

---
<h3 align="left"> Generation of the dataset </h3>

For the training of the segmentation algorithms, a data set of 40 color images with their respective mask was used. For the training of the models, a file separated by commas is created that contains in each row the information corresponding to each pixel of an image. In the columns of the document the information of the coordinates in X and Y, in R and tetha, the value of each of the color components and the value of the mask at that point are stored. After this, the data is normalized between -1 and 1 and with this the training data set is generated, which in total contains 2.621440 data, of which 80% is used for training, while the remaining 20% is used to validate the training.

---
<h3 align="left"> Results </h3>

The Image Labelling App dependencies, compilation, and configuration are packaged in a Docker Image. Before continuing, make sure you have Docker installed on your device. If it is not installed and you are working on Linux, you can run the following commands in a terminal:

| **ML model** | **MCC** |
|:------------------|:------------:|
|SVM(linear)        |0.5275        |
|Logistic regression|0.8378        |

| **ML model** | **ACC** | **PPV** |
|:------------------|:------------:|:------------:|
|ANN        |0.9221      |0.9447|

---
<h2 align="center">Which is the best?</h2>

The best performance is evidently that of the ANNs.

---

<h2 align="center"> How to improve those results? </h2>

For image segmentation it is better to use deep learning techniques. The results drastically improve both in the MCC evaluation and in the inference time. This can be evidenced by looking at the results of the project developed by us and available [here](https://github.com/Kiwi-PUJ).

---
<h2 align="left"> This project is being developed by: </h2>

✈️ Andrea Juliana Ruiz Gómez, [GitHub](https://github.com/andrearuizg), Email: andrea_ruiz@javeriana.edu.co

🏎️ Pedro Elí Ruiz Zárate, [GitHub](https://github.com/PedroRuizCode), Email: pedro.ruiz@javeriana.edu.co


<h3 align="left"> With the support of: </h3>

👨🏻‍🏫 Francisco Carlos Calderón Bocanegra, [GitHub](https://github.com/calderonf)
