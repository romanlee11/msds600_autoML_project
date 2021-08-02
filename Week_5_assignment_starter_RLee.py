#!/usr/bin/env python
# coding: utf-8

# # DS Automation Assignment

# Using our prepared churn data from week 2:
# - use pycaret to find an ML algorithm that performs best on the data
#     - Choose a metric you think is best to use for finding the best model; by default, it is accuracy but it could be AUC, precision, recall, etc. The week 3 FTE has some information on these different metrics.
# - save the model to disk
# - create a Python script/file/module with a function that takes a pandas dataframe as an input and returns the probability of churn for each row in the dataframe
#     - your Python file/function should print out the predictions for new data (new_churn_data.csv)
#     - the true values for the new data are [1, 0, 0, 1, 0] if you're interested
# - test your Python module and function with the new data, new_churn_data.csv
# - write a short summary of the process and results at the end of this notebook
# - upload this Jupyter Notebook and Python file to a Github repository, and turn in a link to the repository in the week 5 assignment dropbox
# 
# *Optional* challenges:
# - return the probability of churn for each new prediction, and the percentile where that prediction is in the distribution of probability predictions from the training dataset (e.g. a high probability of churn like 0.78 might be at the 90th percentile)
# - use other autoML packages, such as TPOT, H2O, MLBox, etc, and compare performance and features with pycaret
# - create a class in your Python module to hold the functions that you created
# - accept user input to specify a file using a tool such as Python's `input()` function, the `click` package for command-line arguments, or a GUI
# - Use the unmodified churn data (new_unmodified_churn_data.csv) in your Python script. This will require adding the same preprocessing steps from week 2 since this data is like the original unmodified dataset from week 1.

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('updated_churn_data.csv', index_col='customerID')
df


# In[10]:


conda install -c conda-forge pycaret -y


# In[11]:


from pycaret.classification import setup, compare_models, predict_model, save_model, load_model


# In[12]:


automl = setup(df, target='Churn')


# In[39]:


automl[6]


# In[16]:


best_model = compare_models()


# In[25]:


best_model


# In[26]:


best = compare_models(sort = 'Kappa')


# In[ ]:


I selected Kappa as a metric I thought could be the best to use for the model, knowing it produced 
the second lowest percentage. Interesting that after I did that, the automl pulls the ada as
the best possible model for my dataset. I will try again by sorting to the metric AUC instead this time. 


# In[28]:


best = compare_models(sort = 'AUC')


# In[ ]:


Using AUC as the best metric for my test, the logistic regression again returns as the best model to use 
for my data set. Now if I want to return the top three models based on my data, I can run this code below, based
on the default metric of accuracy I can run this code below. 


# In[29]:


top3 = compare_models(n_select = 3)


# In[30]:


df.iloc[-1].shape


# In[31]:


df.iloc[-2:-1].shape


# In[ ]:


You can see that they differ because df.iloc[-1].shape only returns the total number of columns since
we only specified an indexing of a 1D array. 


# In[32]:


predict_model(best_model, df.iloc[-2:-1])


# In[ ]:


We can see this line of code creates a score column with the probability class of 1. It also
creates a 'label' column with the predicted label, where it rounds up the score if the score
is greater than or equal to 0.5. 


# In[33]:


save_model(best_model, 'LR')


# In[ ]:


We save our trained model based on our best model comparison code ran earlier so we can use 
it in a python file later. 


# In[56]:


import pickle

with open('LR_model.pk', 'wb') as f:
    pickle.dump(best_model, f)


# In[57]:


with open('LR_model.pk', 'rb') as f:
    loaded_model = pickle.load(f)


# In[58]:


new_data = df.iloc[-2:-1].copy()
new_data.drop('Churn', axis=1, inplace=True)
loaded_model.predict(new_data)


# In[59]:


loaded_lr = load_model('LR')


# In[60]:


predict_model(loaded_lr, new_data)


# In[ ]:


I saved my pycaret model and test it with loading it and making predictions to make sure it works which it does. 


# In[68]:


from IPython.display import Code

Code('predict_churn.py')


# In[69]:


get_ipython().run_line_magic('run', 'predict_churn.py')


# In[ ]:


I created a separate python module to take in new data and make a prediction. 
I then import the code and test the code to make sure it reads and pulls up necesssary data. 
I can see that we have binary data returning for churn and no churn so the model is working ok but it is 
not perfect. I need to lookout for false positives and false negatives to ensure we understand our new data correctly. 


# # Summary

# In[ ]:





# Write a short summary of the process and results here.
