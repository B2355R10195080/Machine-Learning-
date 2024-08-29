import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split



# Sample dataset (features: Weight (grams), Fur Length (0=Short, 1=Long), Ears Shape (0=Pointed, 1=Round); target: Animal Type)

X = np.array([

    [3000, 1, 0],  # Cat

    [4000, 1, 0],  # Cat

    [5000, 0, 1],  # Dog

    [6000, 0, 1],  # Dog

    [3500, 1, 1],  # Cat

    [7000, 0, 1],  # Dog

    [3200, 1, 0],  # Cat

    [7500, 0, 1]   # Dog

])

y = np.array(['Cat', 'Cat', 'Dog', 'Dog', 'Cat', 'Dog', 'Cat', 'Dog'])  # Corresponding animal types



# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



# Create and train the model

model = DecisionTreeClassifier()

model.fit(X_train, y_train)



# Get user input

weight = float(input("Enter the weight of the animal in grams: "))

fur_length = int(input("Enter the fur length (0=Short, 1=Long): "))

ears_shape = int(input("Enter the ears shape (0=Pointed, 1=Round): "))



# Make a prediction

predicted_animal = model.predict([[weight, fur_length, ears_shape]])

print(f"The predicted type of animal is: {predicted_animal[0]}")


Explanation:

The Python code uses a decision tree classifier to determine 
if an animal is a cat or a dog based on its weight, fur length, and ear shape. 
The dataset is split into training and testing sets, with the model trained on 
the training data. After training, the code prompts the user for input values 
(weight, fur length, and ear shape) and predicts whether the animal is a cat or 
a dog based on these features. The script focuses on training the model and making 
predictions, without evaluating accuracy.



