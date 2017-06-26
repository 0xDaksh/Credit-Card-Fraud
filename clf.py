
# coding: utf-8

# In[1]:

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras


# In[2]:

df = pd.read_csv('creditcard.csv')
df.head(1)


# In[3]:

df['Class'].unique() # 0 = no fraud, 1 = fraudulent


# In[4]:

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# In[5]:

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=1)


# In[6]:

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[7]:

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[8]:

clf = Sequential([
    Dense(units=16, kernel_initializer='uniform', input_dim=30, activation='relu'),
    Dense(units=18, kernel_initializer='uniform', activation='relu'),
    Dropout(0.25),
    Dense(20, kernel_initializer='uniform', activation='relu'),
    Dense(24, kernel_initializer='uniform', activation='relu'),
    Dense(1, kernel_initializer='uniform', activation='sigmoid')
])


# In[9]:

clf.summary()


# In[10]:

clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[11]:

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, # for tensorboard
          write_graph=True, write_images=True)
clf.fit(X_train, Y_train, batch_size=15, epochs=5, callbacks=[tbCallBack])


# In[12]:

score = clf.evaluate(X_test, Y_test, batch_size=128)
print('\nAnd the Score is ', score[1] * 100, '%')


# In[13]:

model_json = clf.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
clf.save_weights("model.h5")
print("Saved model to disk")

