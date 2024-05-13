#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


books = pd.read_csv('bestsellers with categories.csv')


# In[3]:


books.head()


# In[4]:


books.shape


# In[5]:


books.columns


# In[6]:


books.head()


# In[7]:


books.rename(columns={
    "Name":"Title",
    "Author":'author',
    "Year":"year",
    "PRICE":"Price",
    "User Rating":"Rating"},inplace = True)


# In[8]:


book_pivot = books.pivot_table(index='Title',values='Rating')


# In[9]:


book_pivot


# In[10]:


book_pivot.shape


# In[11]:


book_pivot.fillna(0,inplace=True)


# In[12]:


book_pivot


# In[13]:


from scipy.sparse import csr_matrix


# In[14]:


book_sparse = csr_matrix(book_pivot)


# In[15]:


book_sparse


# In[16]:


from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(algorithm='brute')


# In[17]:


model.fit(book_sparse)


# In[18]:


distance,suggestion=model.kneighbors(book_pivot.iloc[237,:].values.reshape(1,-1),n_neighbors=6)


# In[19]:


distance


# In[20]:


suggestion


# In[21]:


for i in range(len(suggestion)):
    print(book_pivot.index[suggestion[i]])


# In[22]:


book_pivot.index[237]


# In[23]:


book_pivot.index


# In[24]:


books_name=book_pivot.index


# In[25]:


def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )
    
    for i in range(len(suggestion)):
            books = book_pivot.index[suggestion[i]]
            for j in books:
                
                    print(j)


# In[26]:


book_name ='12 Rules for Life: An Antidote to Chaos'
recommend_book(book_name)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




