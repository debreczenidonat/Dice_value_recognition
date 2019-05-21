# Dice_value_recognition
A simple image recognition project

The experiment is about training a model, that recognizes the value of a thrown dice real-time. The "eye" of the system is a raspberry pi camera module, fixed above the dice. The model gives the probability of 7 possible outcomes: one for each face of the dice (from 1 to 6) and an extra 0 value, which is active when no dice is present. You can see the probabilities predicted by the model on the bar chart. The final prediction is the value with the highest bar.
For this project, I collected my own training data, by taking photos of the dice with different alignments and face values. Then, I used Keras functional API to define a convolutional neural network (CNN), and train it on the collected pictures. Experimenting with different CNN architectures was way more interesting, than collecting the data.
When the system is running, the raspberry pi unit streams the pictures constantly to my pc, where the trained model is deployed. The simultaneously made predictions are shown on the bar chart.

Check the video: https://youtu.be/wM-BXt1r3Hs
