# HebrewLettersClassifier
This is a Hebrew letter recognition project that uses both a feedforward neural network (FNN) and a convolutional neural network (CNN) to recognize handwritten letters. The application allows users to draw a letter, and the models will attempt to recognize it. Currently, the application only recognizes non-sofit letters, but support for sofit letters will be added in the future.

The models are implemented using NumPy, and the web application is built using Django for the backend. It is hosted on PythonAnywhere, and in addition to the pre-trained models, the HebrewLettersClassifier application allows users to retrain the models on new data online. This makes it easy for users to improve the models' accuracy over time, and helps ensure that the application stays up-to-date with new data.

You can try the application out for yourself at the following link: https://abed359.pythonanywhere.com/letter_recognize/
