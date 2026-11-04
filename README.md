# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1:
Write your own stepsLoad and Preprocess the Dataset:

Import the required libraries, load the dataset, handle missing values, encode categorical variables, and normalize the features.

### STEP 2:
Split the Dataset:

Divide the dataset into training and testing sets.

### STEP 3:
Build the Neural Network:

Create a Sequential model with an input layer, hidden layers using ReLU activation, and an output layer with Softmax activation for the four customer segments.

### STEP 4:
Train the Model:

Compile the model using the Adam optimizer and categorical cross-entropy loss function, then train it using the training data.

### STEP 5:
Evaluate and Predict:

Evaluate the model on the test dataset, measure its accuracy, and use it to predict the customer segment (A, B, C, or D) for new customers 





## PROGRAM

### Name:

### Register Number:

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("customers.csv")
df

df.drop(columns = ['ID'], inplace = True)
df

df.fillna({'Work_Experience' : 0, 'Family_Size': df['Family_Size'].median()}, inplace = True)

df.columns

cat_cols = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1','Segmentation']
for  col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
df

x = df.drop(columns = ['Segmentation'])
y = df['Segmentation'].values
xt,xst,yt,yts = train_test_split(x,y,test_size = 0.2, random_state = 42)

scaler1 = StandardScaler()
scaler2 = StandardScaler()
xt = scaler1.fit_transform(xt)
xst = scaler2.fit_transform(xst)

xt = torch.FloatTensor(xt)
xst = torch.FloatTensor(xst)
yt = torch.FloatTensor(yt)
yst = torch.FloatTensor(yts)

train = TensorDataset(xt, yt)
test = TensorDataset(xst,yst)
trl = DataLoader(train,batch_size=16,shuffle = True)
tsl = DataLoader(test,batch_size = 16)

class classification_model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, 32)
        self.l2 = nn.Linear(32, 16)
        self.l3 = nn.Linear(16, 8)
        self.l4 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x

         
model = classification_model(xt.shape[1])
criterion = nn.CrossEntropyLoss()
op = optim.Adam(model.parameters(), lr = 0.001)

epochs = 100
for i in range(epochs):
    for a,b in trl:
        pred = model(a)
        loss = criterion(pred,b.long())
        loss.backward()
        op.step()
        if i % 10 == 0:
            print(f"Loss {i + 1} / e Loss: {loss.item()}")

act = []
pre = []
with torch.no_grad():
    for a,b in tsl:
        pred = model(a)
        _,output = torch.max(pred,1)
        pre.extend(output.numpy())
        act.extend(b.numpy())

accuracy = accuracy_score(act,pre)
conf_matrix = confusion_matrix(act,pre)
class_report = classification_report(act,pre,target_names = ['A','B','C','D'])
print("Accuracy = ",accuracy)
print("Confusion Matrix = \n",conf_matrix)
print("Classification Report = \n",class_report)

sns.heatmap(conf_matrix,annot = True,fmt = 'd', cmap = 'Blues', xticklabels = ['A','B','C','D'], yticklabels = ['A','B','C','D'])
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Confusion Matrix")

```

### Dataset Information
<img width="942" height="291" alt="image" src="https://github.com/user-attachments/assets/6597889d-7b2c-40ca-8072-23ad2ce6affe" />


### OUTPUT
### Accuracy Score
<img width="310" height="36" alt="image" src="https://github.com/user-attachments/assets/bb8f5c22-a33d-4d30-9fb0-b549e08d7601" />


## Confusion Matrix

<img width="216" height="118" alt="image" src="https://github.com/user-attachments/assets/79c41c92-6a90-4c08-bc95-beba7d428109" />


## Classification Report

<img width="587" height="267" alt="image" src="https://github.com/user-attachments/assets/2d65d90d-8cf9-4db2-93e7-103522d11f07" />


### heat map of confusion matrix

<img width="732" height="613" alt="image" src="https://github.com/user-attachments/assets/48f46354-94ee-4979-8de2-d8fd6d0c22f9" />


## RESULT
The program has been executed successfully.
