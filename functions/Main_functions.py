# Functions

# General libraries
from linkedin_api import Linkedin
import pandas as pd
import pickle
import regex as re
from glob import glob

# API authentication:
mail = open('LinkedIn.txt').readlines()[0].split()[0]
password = open('LinkedIn.txt').readlines()[1]
api = Linkedin(f'{mail}',f'{password}')

# Function to get skills for each Linkedin profile 'urn_id':
def get_skills(df):
    '''
    Given a dataframe column with urn_id, make request to API to get profile skills
    Parameters
    ----------
    df : dataframe
    
    Output
    ----------
    Return a list with the profile skills
    '''    
    urn_list = pd.Series.tolist(df['urn_id'])
    skills_people = []
    for i in urn_list:
        skills_people.append(api.get_profile_skills(urn_id = i))
    return skills_people

# Function to create a column to categorize all the profile into a label according their job title

def create_category (data, title_column, category_column):
    '''
    Given a dataframe a column create a new column with a job category according if the job title have the keywords
    
    Parameters
    ----------
    data : dataframe
    title_column : string, column name in the dataframe 
    category_column : string, new column to be created as an output
    
    Output
    ----------
    The categories that will be use to map the first column given are:
    'Data Analyst'
    'Data Engineer'
    'Data Scientist'
    The categories keywords must be present in teh title as strings in lowercase or uppercase.
    If there are missing values the function will return a null value for that row
    '''
    
    data[title_column].fillna('Unknown', inplace = True)
    
    for i in data[data[title_column].str.contains('Analy', na=False, case=False)].index:
        data.loc[i,category_column] = 'Data Analyst'
        
    for i in data[data[title_column].str.contains('Intelligen',na=False, case=False)].index:
        data.loc[i,category_column] = 'Data Analyst'

    for i in data[data[title_column].str.contains('Engineer', na=False, case=False)].index:
        data.loc[i,category_column] = 'Data Engineer'
        
    for i in data[data[title_column].str.contains('Data\s*\w*\s*\w*\s*Architect',na=False, case=False)].index:
        data.loc[i,category_column] = 'Data Engineer'        
        
    for i in data[data[title_column].str.contains('Scien', na=False, case=False)].index:
        data.loc[i,category_column] = 'Data Scientist'
        
    for i in data[data[title_column].str.contains('ML', na=False, case=False)].index:
        data.loc[i,category_column] = 'Data Scientist'
        
    for i in data[data[title_column].str.contains('Machine\s*\w*\s*\w*\s*Learning', na=False, case=False)].index:
        data.loc[i,category_column] = 'Data Scientist'
        
    return data

# function to extract all the skills in a single list for each category
# create new dataframes with the category and the skills for each one

def top_skills_category (data, category_column, list_of_skills):
    '''
    Given a dataframe a category name and a list of skills 
    Return a dataframe with the top 50 skills as index and the count of the times the skills are present in the category profiles
    
    Parameters
    ----------
    data : dataframe
    category_column : string, new column to be created as an output
    list_of_skills: list, with strings values

    Output
    ----------
    First unnest the column of the original dataframe with list with sublists and nested dictionaries and crete a single list
    Then map the column with the list of keyword and find if they are present in the column
    Sustitude the value for the keyword given
    And finally create a dataframe with the 50 most repeated skills/values
    
    '''
    
    skill_list = [dicc['name'] for lst in data['skills'] for dicc in lst]
    skills_category_df= pd.DataFrame(skill_list)
    
    for i in list_of_skills:
        for e in skills_category_df[skills_category_df[0].str.contains(i)][0].index:
            skills_category_df.loc[e, 0] = i

    skills_category_df= pd.DataFrame(skills_category_df[0].value_counts().sort_values(ascending=False).head(50))
    skills_category_df.rename(columns={0: category_column}, inplace=True)
    
    return skills_category_df

# Reading csv and merging in one dataframe
def combine_csv(file_directory, column_drop_list, source_column):
    """
    Read all CSVs in route specified and combine all files in a single pandas dataframe
    
    Parameters
    ----------
    file_directory : string, file directory where csv files are located
    column_drop_list : list of columns to drop
    source_column: string, new column to be created indicateing source csv file

    Output
    ----------
    Pandas dataframe with all csv information
    """
    
    csv_files = glob(file_directory+"/*.csv")
    dfs = []
    for csv in csv_files:
        print(csv)
        df = pd.read_csv(csv)
        df.drop(labels=column_drop_list, axis=1, inplace=True, errors='ignore')  
        df[source_column] = csv
        dfs.append(df)
    dfs = pd.concat(dfs).reset_index(drop=True)
    return dfs

# Processing 'Salary Estimate' column:
def salary_min_max (data, column):
    '''
    Split salary column and return two value for the numeric pattern in the original column
    
    Parameters
    ----------
    data : dataframe
    column : string, salary column in the dataframe 
    
    Output
    ----------
    Return two lists with minimum and maximum salaries per job title
    '''    
    
    salary_min = []
    salary_max = []

    for i in data[column]:
        salary_min.append(re.findall('[0-9]+', i)[0])
        salary_max.append(re.findall('[0-9]+', i)[-1])
    return salary_min, salary_max

# Function to extract years of experience from a job description
def years_experience (data, column):
    '''
    Find the years of experience within a job description, given a column of 'job description'
    Parameters
    ----------
    data : dataframe
    column : string, job description column in the dataframe 
    
    Output
    ----------
    A list containing the years of experience for each row (job)
    ''' 
    
    exp_row = []
    years_list = []
    for i in data[column]:
        experience = re.findall('\w+\+*\s*years\s*\w*\s*\w*\s*experience', str(i))
        if len(experience) == 1:
            years_list.append(experience[0].split(' ')[0])
        elif len(experience) > 1:
            for e in range(len(experience)):
                exp_row.append(experience[e].split(' ')[0])
            years_list.append(exp_row)
            exp_row= []
        else:
            years_list.append('NaN')
    return years_list

