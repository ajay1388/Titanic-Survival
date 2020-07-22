# Titanic-Survival

## Objective
This project is binary classification problem, where the passenger either survived (`1`) or died (`0`). Here is a list of the columns of the dataset:

* `PassengerID` - Unique ID for each column
* `Pclass` - Class of the passenger's ticket. Either 1, 2 or 3.
* `Cabin` - Passenger's cabin number
* `Sex` - Passenger's sex (male or female)
* `Age` - Passenger's age
* `Sibsp` - Number of sibling or spouses aboard the Titanic
* `Parch` - Number of parents or children aboard the Titanic
* `Ticket` - Passenger's ticket number
* `Fare` - The price paid for the passenger's ticket
* `Survived` - Whether the passenger survived (`1`) or not (`0`)
* `Embarked` - Port where the passenger embarked. Can be:
    * `C` - Cherbourg
    * `Q` - Queenstown
    * `S` - Southampton
    
## Software and Libraries

This Project Uses the following softwares and libraries:

  * Python 3.8
  * pandas
  * numpy
  * seaborn
  * matplotlib
  * random
  * sklearn
  
## What to predict:

  For each passenger in the test set,Our model will be trained to predict whether or not they survived the sinking of the Titanic.
  
## Built With
  
  In this Project **Gaussian Naive Bayes** Model is used
  
## Inside The Project

  #### Creating Training and Testing Data set

  ```python
  x_train,x_test,y_train,y_test = tts(x,y,test_size=0.2,random_state=0)

  ```
  #### Training the model

  ```python
  gaussian = gnb()
  gaussian.fit(x_train,y_train)
  ```
  #### Making the prediction

  ```python
  y_pred = gaussian.predict(x_test)
  ```

# Accuracy
```python
print("Accuracy Using Naive Bayes ",acc_gn)
```
Accuracy Using Naive Bayes **82.68%**
