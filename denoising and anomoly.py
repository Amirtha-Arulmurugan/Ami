#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Denoising
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense,Flatten
from keras.models import Sequential


# In[5]:


(x_train,_),(x_test,_)=mnist.load_data()
x_train=x_train/255
x_test=x_test/255
noise=3


# In[7]:


x_train_noisy=x_train+noise*np.random.normal(loc=0,scale=1,size=x_train.shape)
x_test_noisy=x_test+noise*np.random.normal(loc=0,scale=1,size=x_test.shape)


# In[8]:


model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(36,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(36,activation="relu"))
model.add(Dense(28*28,activation="sigmoid"))


# In[9]:


model.compile(optimizer="adam",loss="mse",metrics=['accuracy'])


# In[10]:


model.fit(x_train_noisy,x_train.reshape(-1,784),epochs=10,batch_size=32,validation_data=(x_train_noisy,x_train.reshape(-1,784)))


# In[23]:


m=model.predict(x_test_noisy)
m


# In[24]:


m=np.array(m)
m


# In[25]:


m.shape


# In[27]:


plt.imshow(m.reshape(10000,28,28)[0])


# In[31]:


#Anomoly Detection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler


# In[36]:


df=pd.read_csv(r"C:\Users\DELL\Downloads\ecg.csv")
x_test=df.iloc[:,:-1]
y_test=df.iloc[:,-1]


# In[38]:


ss=StandardScaler()
x_test=ss.fit_transform(x_test)


# In[54]:


model = Sequential()
model.add(Dense(140, input_shape=(140,), activation="relu"))  # Change input shape to (140,)
model.add(Dense(120, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(120, activation="relu"))
model.add(Dense(140, activation="sigmoid"))
model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])
model.fit(x_test, x_test, epochs=10, batch_size=32, validation_data=(x_test, x_test))


# In[56]:


pre=model.predict(x_test)


# In[61]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.plot(x_test[i],label='Original')
    plt.plot(pre[i],label='Anamoly Detection')
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.fill_between(np.arange(140),x_test[i],pre[i],color='lightcoral')


# In[ ]:




