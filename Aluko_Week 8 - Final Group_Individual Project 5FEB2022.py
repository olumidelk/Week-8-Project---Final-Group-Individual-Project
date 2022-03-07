#!/usr/bin/env python
# coding: utf-8

# Importing all the necessary libraries

# Title: Week 8 - Final Group/Individual Project
# 
# Date: 5FEB2022
# 
# Author: Olumide Aluko
# 
# Purpose: Create a data visualization project that i am proud of with creativity

# Input:
# 
# Output:
# 
# Notes: Week 8 - Final Project

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import iplot
import plotly.graph_objs as go
from plotly import tools
plt.rcParams['axes.grid'] = False
from plotly import express as px


# # Import Dataset
# 
# Data visualization on Movies - Python: Movies on Netflix, Prime Video, Hulu and Disney+
# 
# Source: https://www.kaggle.com/linci04/data-visualization-on-movies

# In[2]:


df = pd.read_csv("MoviesOnStreamingPlatforms_updated.csv")


# Data exploration

# In[3]:


df.head(3)


# In[4]:


df.shape


# In[5]:


# examine data info
df.info()


# Missing Values

# In[6]:


total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# Drop irrelavent columns

# In[7]:


df.drop(['Unnamed: 0','Type'],axis=1, inplace=True)


# Data Visualization

# # 1. Total Number of movies released each year.
# 
# Type: Line chart
# 
# Line graphs are used to track changes over short and long periods of time. A line chart uses points connected by line segments from left to right to demonstrate changes in value.

# In[8]:


yearly_movie_count = df.groupby('Year')['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})


# In[9]:


fig = px.line(yearly_movie_count, x='Year', y='MovieCount',width=600, height=300, template="plotly_white")
fig.show()


# Inference:
# 
# - Data ranges from the year 1902 to 2020
# 
# 
# - More number of movies were released in the year 2017, which is a total of 1401 movies.
# 
# 
# - From the year 1995, a significant raise in the number of movies released was observed.

# # 2. Genres which have a greater number of movies.
# 
# Type: Bar chart / Column chart – vertical
# 
# Bar chart represents categorical data with rectangular bars with heights or lengths proportional to the values that they represent. A vertical bar chart is sometimes called a column chart.

# In[10]:


Genre_count = df.groupby('Genres')['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})
a=Genre_count.sort_values(by='MovieCount', ascending=False).head(20)
b=a.sort_values(by='Genres', ascending=False).head(20)


# In[11]:


fig = px.bar(b, x="Genres", y="MovieCount",color='MovieCount',
             width=600, height=500, 
             title='Genres which have a greater number of movies.',
             template="plotly_white")
fig.show()


# Inference:
# 
# • The genre “drama” was observed to have the most number of movies which is a total of 1314.
# 
# • The genre of” documentary” was found to have the second most number of movies.
# 
# • Following Documentary, the comedy and horror genres were found to have the highest numbers.

# # 3. Average IMDb rating of top 5 genres
# 
# Type: Bar chart with animation frame.
# 
# To visualize large number of data animation frame was used

# In[12]:


top_5_genres = ['Drama','Documentary','Comedy', 'Comedy,Drama','Horror']
table = df.loc[:,['Year','Genres','IMDb']]
table['AvgRating'] = table.groupby([table.Genres,table.Year])['IMDb'].transform('mean')
table.drop('IMDb', axis=1, inplace=True)
table = table[(table.Year>1995) & (table.Year<2020)]
table = table.loc[table['Genres'].isin(top_5_genres)]
table = table.sort_values('Year')


# In[13]:


fig=px.bar(table,x='Genres', y='AvgRating', animation_frame='Year', 
           animation_group='Genres', color='Genres',height=500,width=700, hover_name='Genres', range_y=[0,10])
fig.update_layout(showlegend=False)
fig.show()


# Inference:
# 
# • The top 5 genres were observed to be drama, comedy, comedy+drama, horror and documentary.
# 
# • Average rating for each genre can be seen with respect to the year in the animation presented above.

# # 4. Average rating in different countries
# 
# CREATING A NEW DATA SET WITH COUNTRY CODE

# In[14]:


df['Country'] = df['Country'].str.split(',')

df = (df
 .set_index(['ID','Title','Year','Age', 'IMDb', 'Rotten Tomatoes', 'Netflix', 'Hulu',
             'Prime Video', 'Disney+','Directors' ,'Genres','Language', 'Runtime'])['Country']
 .apply(pd.Series)
 .stack()
 .reset_index()
 .drop('level_14', axis=1)
 .rename(columns={0:'Country'}))


# In[15]:


import pycountry
def do_fuzzy_search(country):
    try:
        result = pycountry.countries.search_fuzzy(country)
        return result[0].alpha_3
    except:
        return np.nan
df["country_code"] = df["Country"].apply(lambda country: do_fuzzy_search(country))


# In[16]:


df.to_csv('new.csv')


# Type: Choropleth map
# 
# Choropleth map is a thematic map where geographic regions are coloured, shaded, or patterned in relation to a value. This type of map is particularly useful when visualizing a variable and how it changes across defined regions.

# In[17]:


df1 = pd.read_csv("new.csv")


# In[18]:


li = df1.groupby(['country_code','Country'])['IMDb'].mean().reset_index().rename(columns = {'Title':'MovieCount'})


# In[19]:


fig = px.choropleth(li, locations="country_code",
                    color="IMDb", 
                    hover_name="Country", 
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()


# Inference:
# 
# • Somalia region was observed to have the highest average rating of 8.2
# 
# • Korea presented the lowest average rating of 4.7.

# # 5. Movie count in different countries
# 
# Type: 3d choropleth map
# 
# It is of the type orthographic, which is basically 3d choropleth map.

# In[20]:


a = df1.groupby(['country_code','Country'])['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})


# In[21]:


colors=[[0, 'rgb(102,194,165)'], [0.05, 'rgb(102,194,165)'],
              [0.15, 'rgb(171,221,164)'], [0.2, 'rgb(230,245,152)'],
              [0.25, 'rgb(255,255,191)'], [0.35, 'rgb(254,224,139)'],
              [0.45, 'rgb(253,174,97)'], [0.55, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]


# In[22]:


plotmap = [ dict(
        type = 'choropleth',
        locations = a['Country'],
        locationmode = 'country names',
        z = a['MovieCount'],
        text = a['Country'],
        colorscale = colors,
       
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(102,194,165)',
                width = 0.10
            ) ),
        colorbar = dict(
            title = "NUMBER OF MOVIES RELEASED IN DIFFERENT COUNTRIES"),
      ) ]


# In[23]:


layout = dict(
    title = "",
    geo = dict(
        showframe = True,
        
        showcoastlines = True,
        showocean = True,
        
        oceancolor = '#26466D',
        projection = dict(type = 'orthographic')
    ),
    height=700,
    width=900
)

fig = dict( data=plotmap, layout= layout )
iplot(fig)


# Inference:
# 
# • Total number of movies released in the different region of the world can be observed from the choropleth map above.
# 
# • USA was observed to be the country that released the highest number of movies.

# # 6. Distribution of Runtime
# 
# Type: Distplot
# 
# The distplot represents the univariate distribution of data i.e., data distribution of a variable against the density distribution.

# In[24]:


lk=df.dropna()
import plotly.figure_factory as ff
x = lk["Runtime"]
hist_data = [x]
group_labels = ['Runtime'] # name of the dataset
colors = ['rgb(0, 0, 100)', 'rgb(0, 200, 200)']
fig = ff.create_distplot(hist_data, group_labels,colors=colors)
fig.show()


# Inference:
# 
# • The distribution of runtime was found to be normal.
# 
# • Most of the movies were found to have duration of 90 minutes.
# 
# • Duration was observed to vary from a range of 11 minutes to 259 minutes.

# # 7. Number of movies present in specific age group with respect to year
# 
# Type: Sunburst plot
# 
# Sunburst plots visualize hierarchical data spanning outwards radially from root to leaves. The root starts from the centre and children are added to the outer rings.

# In[25]:


age = df.groupby(['Age','Year'])['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})
age_year=age[age['Year']>=2015]
fig = px.sunburst(age_year, path=['Age', 'Year', 'MovieCount'],width=600,height=450)
fig.show()


# Inference:
# 
# • Most number of movies were found to be released under the age group of 18+
# 
# • About 374 movies were released in the year 2017 under the age group of 18+.

# # 8.Number of movies released in different languages.
# 
# Type: Horizontal bar chart.
# 
# Horizontal bar graphs represent the data horizontally. It is a graph whose bars are drawn horizontally. The data categories are shown on the vertical axis and the data values are shown on the horizontal axis.

# In[26]:


df['Language'] = df['Language'].str.split(',')

df2 = (df
 .set_index(['ID','Title','Year','Age', 'IMDb', 'Rotten Tomatoes', 'Netflix', 'Hulu',
             'Prime Video', 'Disney+','Directors' ,'Genres','Country', 'Runtime'])['Language']
 .apply(pd.Series)
 .stack()
 .reset_index()
 .drop('level_14', axis=1)
 .rename(columns={0:'language'}))


# In[27]:


lang = df2.groupby(['language'])['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})
the=lang.sort_values(by='MovieCount',ascending=False).head(15)


# In[28]:


fig = px.bar(the, x="MovieCount", y="language", orientation='h',height=450,width=600)
fig.show()


# Inference:
# 
# • Highest number of movies were found to be released in English, which is nearly 17k movies.
# 
# • Second highest was found to be Spanish with about 1367 movies.
# 
# • Portuguese had the least number of movies which is 177.

# # 9. Number of movies present in different age groups
# 
# Type: Funnel Area
# 
# Each row of the Data Frame is represented as a stage of the funnel. Each stage is illustrated as a percentage of the total of all values.

# In[29]:


age_count = df.groupby(['Age'])['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})
ac=age_count.sort_values(by='MovieCount',ascending=False)


# In[30]:


fig = px.funnel_area(names=ac['Age'],
                    values=ac['MovieCount'],
                     height=500,width=500,
                     color_discrete_sequence=px.colors.sequential.RdBu,
                     template="plotly_dark")
fig.update_traces(textinfo="percent+label", title='Movie Count per Age')
fig.show()


# Inference:
# 
# • The age group 18+ was observed to have the highest number of movies.
# 
# • The age group 7+ follows with nearly 19.9% of the movies.
# 
# • The age group 13+ and “all” was found to have nearly 17.1% and 11.5% of the movies respectively.
# 
# • The age group 16+ was observed to have the least number of movies i.e., 4.3%.

# # 10.Top 15 Directors who released the greatest number of movies
# 
# Type: Funnel plot
# 
# Funnel charts are mostly used for representing a sequential process, to compare and see how the numbers change through the stages.

# In[31]:


n = df1.groupby(['Directors'])['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})


# In[32]:


jn=n.sort_values(by='MovieCount',ascending=False).head(15)


# In[33]:


data = dict(
    number=jn['MovieCount'],
    stage=jn['Directors'])
fig = px.funnel(data, x='number', y='stage',height=450,width=600)
fig.show()


# Inference:
# 
# • Jay Chapman was found to have released the highest number of movies i.e., 36 movies.
# 
# • Samuel Rich, Gabriella Fritz and Werner Herzog have released the least number of movies which is 18.

# # 11. Highest number of movies released by directors in different age group
# 
# Type: Tree map
# 
# The tree map chart is used for representing hierarchical data in a tree-like structure. The size of the rectangle is s proportional to the corresponding data value.

# In[34]:


age_directors = df.groupby(['Age','Directors'])['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})


# In[35]:


age_13=age_directors[age_directors["Age"]=='13+'].sort_values(by='MovieCount',ascending=False).head(10)
age_16=age_directors[age_directors["Age"]=='16+'].sort_values(by='MovieCount',ascending=False).head(10)
age_18=age_directors[age_directors["Age"]=='18+'].sort_values(by='MovieCount',ascending=False).head(10)
age_7=age_directors[age_directors["Age"]=='7+'].sort_values(by='MovieCount',ascending=False).head(10)
age_all=age_directors[age_directors["Age"]=='all'].sort_values(by='MovieCount',ascending=False).head(10)

frames = [age_13,age_16,age_18,age_7,age_all]

result = pd.concat(frames)


# In[36]:


fig = px.treemap(result, path=['Age', 'Directors','MovieCount'], values='MovieCount')
fig.show()


# Inference:
# 
# • Hierarchy structure: Age group -> Directors -> Movie Count
# 
# • The size of 18+ group’s rectangle is larger compared to the other group’s. Therefor a greater number of movies were released under the age group 18+.
# 
# • Under 18+ age group Cheh Chang has released a total of 18 movies which is the highest.
# 
# • Details about the other age groups can be inferred based on the size of the rectangle in the chart above. .

# # 12. Which age group movie has Average highest IMDb rating in each year
# 
# Type: Area plot
# 
# Area Graphs are Line Graphs but with the area below the line filled in with a certain colour or texture. An area plot displays quantitative data visually.

# In[37]:


b = df.groupby(['Age','Year'])['IMDb'].mean().reset_index().rename(columns = {'Title':'MovieCount'})
b1=b.sort_values(by='IMDb',ascending=True)


# In[38]:


fig = px.area(b1, x="Year", y="IMDb", color="Age",
	      line_group="Age",height=400,width=700)
fig.show()


# Inference:
# 
# • The age group “ALL” had the highest average rating (i.e., 8) during the year – 1983.
# 
# • The age group “7+” had the highest average rating (i.e., 6.69) during the year – 1984
# 
# • The age group “13+” had the highest average rating (i.e., 6.15) during the year – 1994
# 
# • The age group “16+” had the highest average rating (i.e.,7.2 ) during the year – 2003
# 
# • The age group “18+” had the highest average rating (i.e., 6.004) during the year – 1981

# # 13. Content released on different platform over the years.
# 
# Type: Multiple line plot
# 
# Multiple line graph, there are two or more lines in the graph connecting two or more sets of data points. The independent variable is listed along the horizontal, or x, axis and the quantity or value of the data is listed along the vertical, or y, axis.

# In[39]:


netflix = df.groupby(['Netflix','Year'])['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})
netflix_=netflix[netflix['Netflix']==1]


hulu = df.groupby(['Hulu','Year'])['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})
hulu_= hulu[hulu['Hulu']==1]


# In[40]:


prime = df.groupby(['Prime Video','Year'])['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})
prime_=prime[prime['Prime Video']==1]
prime_

disney = df.groupby(['Disney+','Year'])['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})
disney_=disney[disney['Disney+']==1]


# In[41]:


import plotly.graph_objects as go

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=netflix_['Year'], y=netflix_['MovieCount'],name='Netflix'))
fig1.add_trace(go.Scatter(x=hulu_['Year'], y=hulu_['MovieCount'],name='Hulu'))
fig1.add_trace(go.Scatter(x=prime_['Year'], y=prime_['MovieCount'],name='Prime'))
fig1.add_trace(go.Scatter(x=disney_['Year'], y=disney_['MovieCount'],name='Disney'))
fig1.show()


# Inference
# 
# Netflix: in the year 2017 , 696 movies were released
# 
# Hulu: in the year 2018, 222 movies were released
# 
# Prime Video: in the year 2013 ,1000 movies were released
# 
# Disney+: in the year 2003 ,31 movies were released

# # 14. Number of movies in Netflix and prime based on comedy, drama, horror genres:
# 
# Type: Mosaic plot
# 
# Mosaic plot is a graphical display of the cell frequencies of a contingency table in which the area of boxes of the plot are proportional to the cell frequencies of the contingency table.

# In[42]:


dv= df.groupby(['Genres','Netflix','Hulu','Prime Video','Disney+'])['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})


# In[43]:


th = ['Drama','Comedy','Horror']
tab = dv.loc[dv['Genres'].isin(th)]
pd.crosstab(tab['Genres'], [tab['Netflix'], tab['Prime Video']], rownames=['Genres'], colnames=['Netflix', 'Prime Video'])


# In[44]:


from statsmodels.graphics.mosaicplot import mosaic
mosaic(tab,['Genres','Netflix','Prime Video'],gap=0.01,axes_label=True)
plt.show()


# # 15. Relationship between IMDb, Runtime and Rotten Tomatoes:
# 
# Type: Scatter matrix
# 
# Type of multiple scatterplots to determine the correlation (if any) between a series of variables.

# In[45]:


fig = px.scatter_matrix(df, dimensions=['IMDb','Rotten Tomatoes','Runtime'],width=600,height=450)
fig.show()


# # 16. Number of Movies based on different language in each country:
# 
# Type: Grouped bar chart – using animation frame
# 
# A grouped bar chart extends the bar chart, plotting numeric values for levels of two categorical variables instead of one. Bars are grouped by position for levels of one categorical variable, with colour indicating the secondary category level within each group.

# In[46]:


df1 = pd.read_csv("new.csv")

df1['Language'] = df1['Language'].str.split(',')

df3 = (df1
 .set_index(['ID','Title','Year','Age', 'IMDb', 'Rotten Tomatoes', 'Netflix', 'Hulu',
             'Prime Video', 'Disney+','Directors' ,'Genres','Country', 'Runtime'])['Language']
 .apply(pd.Series)
 .stack()
 .reset_index()
 .drop('level_14', axis=1)
 .rename(columns={0:'language'}))


# In[47]:


er= df3.groupby(['language','Country'])['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})
ert=er[er['MovieCount']>=60]


# In[48]:


fig = px.bar(ert, x="Country", y="MovieCount",
             color='language', barmode='group',
             width=700,
    height=400,animation_frame='Country')
fig.show()


# Inference:
# 
# • Animation frame was used to visualize large amount of data.
# 
# • United states: nearly 10k movies were released in English ,305 movies in French,430 movies in Spanish, and 114 movies in Russian.

# # 17. Age wise Movie count on Prime video over the years:
# 
# Type: Bubble Chart
# 
# Bubble chart is a scatter plot in which a third dimension of the data is shown through the size of markers. The size of the markers shows the proportion of the data.

# In[49]:


pri = df.groupby(['Prime Video','Year','Age'])['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})
pri_=pri[pri['Prime Video']==1]
fig = px.scatter(pri_.query("Year>2014"), x="Year", y="Age",
	         size="MovieCount", color="MovieCount", color_discrete_sequence=px.colors.sequential.RdBu,title='Age Wise Movie count on Prime Video',
                 hover_name="MovieCount", log_x=True, size_max=60,height=500,width=700)

fig.show()


# Inference:
# 
# • It is observed from the size of marker that the 18+ age group has the highest number of movies.
# 
# • The age group ‘all’ has the least number of movies.
# 
# • In the year 2015, nearly 135 movies were released under the age group 18+.
# 
# • The year with the least number of movies released was observed to be 2020.

# # 18. PARALLEL CATEGORIES OF AGE VS PLATFORMS

# In[50]:


df10 = df.groupby(['Age','Year','Netflix','Hulu','Prime Video','Disney+'])['Title'].count().reset_index().rename(columns = {'Title':'MovieCount'})
fig = px.parallel_categories(df10, color="MovieCount")
fig.show()


# # 19. Movies that has the highest rating greater than 9.3

# In[51]:


top_movies=df[df['IMDb']>=9][['Title','Directors']]


# In[52]:


from wordcloud import WordCloud, ImageColorGenerator
text = ",".join(review for review in top_movies.Title)
wordcloud = WordCloud(max_words=50,collocations=False,background_color="black").generate(text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.figure(1,figsize=(20, 20))
plt.show()

