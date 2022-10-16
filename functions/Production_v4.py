#!/usr/bin/env python
# coding: utf-8

# In[3]:


# importing libraries 
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import pickle 
from os import path
#import 
from wordcloud import WordCloud
import Modeling_functions as mf
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

currdir = path.dirname("cloud.png")

from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

import plotly.express as px


# In[4]:


#import and authentication of linkedIn API
get_ipython().system('pip install git+https://github.com/tomquirk/linkedin-api.git')
get_ipython().system('pip install linkedin-api~=2.0.0a')

from linkedin_api import Linkedin
mail = open('LinkedIn.txt').readlines()[0].split()[0]
password = open('LinkedIn.txt').readlines()[1]
api = Linkedin(f'{mail}',f'{password}')


# In[125]:


def get_prediction ():
    '''
    
    '''
    filename = 'output/jobs_data.pkl'
    infile = open(filename,'rb')
    jobs_data = pickle.load(infile)
    infile.close()
    
    filename = 'best_model_3.pkl'
    infile = open(filename,'rb')
    final_model = pickle.load(infile)
    infile.close()
    
    try: 
        
        print('Write your LinkedIn URL:')
        URL = input()
        if URL[-1] == '/':
            url_name=URL.split('in/')[-1].split('/')[0]
        else:
            url_name=URL.split('in/')[-1]
        

        # Calling the Linkedin API
        profile = api.get_profile(url_name)
        skills_profile = api.get_profile_skills(urn_id=profile['entityUrn'].split('urn:li:fs_profile:')[1])
        skill_list = [dicc['name'] for dicc in skills_profile]
        dicc_skill= {key: 0 for key in list(jobs_data.columns)}

        #generating the wordcloud
        print('these are your skills:')
        text = ' '.join(map(str,skill_list ))

        d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

        cloud_mask = np.array(Image.open(path.join(d, "oval.png")))

        mask = np.array(Image.open("oval.png"))
        wordcloud = WordCloud( background_color="white", max_words=1000, mask=mask).generate(text)
        
        plt.savefig('N.png')

        plt.figure(figsize=(15,7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig('wordcloud.png')
        plt.show()

        # Preprocessig the data
        print(' ')
        print('-----------------------------------')
        print('Your skills among the most popular in the data industry:')
        for i in list(jobs_data.columns):
            if i.lower() in text.lower():
                dicc_skill[i] = 1
                
        # Individually find R and C skills as one word 
        dicc_skill['R']= np.where(mf.find_only_whole_word("r", text), int(1), int(0))
        dicc_skill['R'] = int(dicc_skill['R'])
        dicc_skill['C']= np.where(mf.find_only_whole_word("c", text), int(1), int(0))
        dicc_skill['C'] = int(dicc_skill['C'])
        dicc_skill['ML']= np.where(mf.find_only_whole_word("ml", text), int(1), int(0))
        dicc_skill['ML'] = int(dicc_skill['ML'])
        
        for key, val in dicc_skill.items():
            if val==1:
                print('•',key)
        
        if sum(list(dicc_skill.values())) < 4 :
            print('Sorry!')
            print("it looks like you don't have enough data skills yet. Check your profile and try again.")
            return
        
        else: 
            jobs_data = jobs_data.append(dicc_skill, ignore_index=True)
            production = jobs_data.tail(1)

            #Using the model to predict
            target = 'Job_Category'
            X_prod = production.drop(target, axis=1)
            y = production[target]

            y_prod= final_model.predict(X_prod)
            final_model.predict_proba(X_prod)

            results = pd.DataFrame(final_model.predict_proba(X_prod), columns = ['Data Analyst', 'Data Engineer', 'Data Scientist'],)
            results = results.apply(lambda x: round(x,4)*100)

            final_pred = y_prod[0]

            #print(results)
            print(' ')
            print('---------------------------')
            print('This is your data role prediction according to your skills:')
            print(final_pred)
            print(results)

            # Showing the results

            fig = px.bar(x=list(results.columns), y=list(results.loc[0]), color=list(results.columns), text_auto=True, 
                        labels={'x':'Job role', 'y':'fit percentage'}, title='Your skillscaner result:')
            #fig.show()
        
    except IndexError as e:
        print('Link not found, try again')

    except KeyError as a:
        print('Please include your full linkedIn url')

    return  fig
    

