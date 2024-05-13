#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[77]:


books = pd.read_csv('bestsellers with categories.csv')


# In[78]:


books.head()


# In[79]:


books.shape


# In[80]:


books.columns


# In[81]:


books.head()


# In[82]:


books.rename(columns={
    "Name":"Title",
    "Author":'author',
    "Year":"year",
    "PRICE":"Price",
    "User Rating":"Rating"},inplace = True)


# In[83]:


book_pivot = books.pivot_table(index='Title',values='Rating')


# In[84]:


book_pivot


# In[85]:


book_pivot.shape


# In[86]:


book_pivot.fillna(0,inplace=True)


# In[87]:


book_pivot


# In[88]:


books["ids"]=[i for i in range(0,books.shape[0])]


# In[89]:


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
vec=TfidfVectorizer()


# In[90]:


vecs=vec.fit_transform(books["Title"].apply(lambda x: np.str_(x)))


# In[91]:


vecs.shape


# In[ ]:





# In[92]:


from sklearn.metrics.pairwise import cosine_similarity


# In[93]:


sim=cosine_similarity(vecs) 


# In[94]:


sim.shape


# In[95]:


sim[100][100]


# In[96]:


def recommend(title):
    book_id=books[books.Title==title]["ids"].values[0]
    scores=list(enumerate(sim[book_id]))
    sorted_scores=sorted(scores,key=lambda x:x[1],reverse=True)
    sorted_scores=sorted_scores[1:]
    bookes=[books[bookes[0]==books["ids"]]["Title"].values[0] for bookes in sorted_scores]
    return bookes


# In[97]:


def recommend_ten(book_list):
    first_ten=[]
    count=0
    for book in book_list:
        if count > 9:
            break
        count+=1
        first_ten.append(book)
    return first_ten


# In[98]:


lst=recommend('10-Day Green Smoothie Cleanse')
m=recommend_ten(lst)


# In[99]:


m


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




