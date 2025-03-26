#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical


# In[23]:


data = pd.read_csv("winequality-red.csv")


# In[24]:


print(data.info())


# In[25]:


print(data.head())


# In[26]:


plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# In[27]:


X = data.drop("quality", axis=1)
y = data["quality"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# In[29]:


y_train_cat = to_categorical(y_train - y_train.min())
y_test_cat = to_categorical(y_test - y_train.min())

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}


# In[30]:


ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_train_cat.shape[1], activation='softmax')
])


# In[31]:


ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ann_history = ann_model.fit(X_train, y_train_cat, epochs=100, validation_data=(X_test, y_test_cat), batch_size=16, class_weight=class_weight_dict)


# In[32]:


y_pred_ann = ann_model.predict(X_test)
y_pred_ann_labels = np.argmax(y_pred_ann, axis=1) + y_train.min()  


# In[33]:


print("ANN Classification Report:")
print(classification_report(y_test, y_pred_ann_labels, zero_division=1))


# In[34]:


plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_ann_labels), annot=True, fmt='d', cmap='Blues')
plt.title("ANN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[35]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(ann_history.history['loss'], label='Train Loss')
plt.plot(ann_history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("ANN Loss Over Epochs")

plt.subplot(1, 2, 2)
plt.plot(ann_history.history['accuracy'], label='Train Accuracy')
plt.plot(ann_history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("ANN Accuracy Over Epochs")

plt.show()


# In[36]:


X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(1, X_train.shape[1])),
    LSTM(64),
    Dense(y_train_cat.shape[1], activation='softmax')
])

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm_history = lstm_model.fit(X_train_lstm, y_train_cat, epochs=100, validation_data=(X_test_lstm, y_test_cat), 
                              batch_size=16, class_weight=class_weight_dict)

y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_lstm_labels = np.argmax(y_pred_lstm, axis=1) + y_train.min()  


# In[37]:


print("LSTM Classification Report:")
print(classification_report(y_test, y_pred_lstm_labels, zero_division=1))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_lstm_labels), annot=True, fmt='d', cmap='Blues')
plt.title("LSTM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[38]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(lstm_history.history['loss'], label='Train Loss')
plt.plot(lstm_history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("LSTM Loss Over Epochs")

plt.subplot(1, 2, 2)
plt.plot(lstm_history.history['accuracy'], label='Train Accuracy')
plt.plot(lstm_history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("LSTM Accuracy Over Epochs")

plt.show()

