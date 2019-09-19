
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


import matplotlib.pyplot as plt
import pandas as pd


# # Context
# 
# Using Watson Analytics, you can predict behavior to retain your customers. You can analyze all relevant customer data and develop focused customer retention programs.
# 
# # Inspiration
# 
# Understand customer demographics and buying behavior. Use predictive analytics to analyze the most profitable customers and how they interact. Take targeted actions to increase profitable customer response, retention, and growth.

# In[6]:


df = pd.read_csv('C:/Users/KATHY/Desktop/df.csv')
df


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df.head()


# We are going to analyze it to understand how different customers behave and react to different marketing strategies.
# 
# #Overall Engagement Rate
# 
# The Response field contains information about whether a customer responded to the marketing efforts.

# In[9]:


# Get the total number of customers who have responded
df.groupby('Response').count()['Customer']


# In[11]:


# Visualize this in a bar plot

ax=df.groupby('Response').count()['Customer'].plot(
    kind='bar',
    color='orchid',
    grid=True,figsize=(10,7),
    title='Marketing Engagement'
)
ax.set_xlabel('Engaged')
ax.set_ylabel('Count')
plt.show()


# In[12]:


# Calculate the percentages of the engaged and non-engaged customers
df.groupby('Response').count()['Customer']/df.shape[0]


# From this output and from the plot, we can see that only about 14% of the customers respondedto the marketing calls.

# #Engagement Rates by Offer Type
# The Renew Offer Type column in this DataFrame contains the type of the renewal offer presentedto the customers. We are going to look into what types of offers worked best for the engaged customers.

# In[7]:


# Get the engagement rates per renewal offer type
by_offer_type_df=df.loc[
    df['Response']=='Yes',# count only engaged customers
].groupby([
    'Renew Offer Type'# engaged customers grouped by renewal offer type
]).count()['Customer']/df.groupby('Renew Offer Type').count()['Customer']

by_offer_type_df


# In[9]:


# Visualize it in a bar plot
ax=(by_offer_type_df*100.0).plot(
    kind='bar',figsize=(7,7),
    color='dodgerblue',
    grid=True
)    


# --> As we can see,Offer2had the highest engagement rate among the customers.

# #Offer Type & Vehicle Class
# 
# We are going to understand how customers with different attributes respond differently to differ-ent marketing messages. We start looking at the engagements rates by each offer type and vehicleclass.

# In[10]:


by_offer_type_df=df.loc[
    df['Response']=='Yes'# engaged customers
].groupby([
    'Renew Offer Type','Vehicle Class'# grouping the data by these two columns
]).count()['Customer']/df.groupby('Renew Offer Type').count()['Customer']


# In[11]:


by_offer_type_df


# # Make the previous output more readable using unstack function
# # to pivot the data and extract and transform the inner-level groups to columns

# In[13]:


by_offer_type_df=by_offer_type_df.unstack().fillna(0)
by_offer_type_df


# In[15]:


# Visualize this data in bar plot
ax=(by_offer_type_df*100.0).plot(
    kind='bar',
    figsize=(10,7),
    grid=True)
ax.set_ylabel('Engagement Rate (%)')
plt.show()


# We already knew from the previous section “Engagement Rates by Offer Type” that Offer2had the highest response rate among customers. Now we can add more insights by having brokendown the customer attributes with the category “Vehicle class”: we can notice that customers withFour-Door Car respond more frequently for all offer types and that those with “Luxury SUV”respond with a higher chance to Offer1 than to Offer2.If we have significantly difference in theresponse rates among different customer rates, we can fine-tune who to target for different setof offers.

# #Engagement Rates by Sales Channel
# 
# We are going to analyze how engagement rates differ by different sales channels.

# In[26]:


by_sales_channel_df=df.loc[
    df['Response']=='Yes'
].groupby(['Sales Channel'
          ]).count()['Customer']/df.groupby('Sales Channel').count()['Customer']
by_sales_channel_df


# In[24]:


Sales Channel


# In[27]:


ax=(by_sales_channel_df*100.0).plot(
    kind='bar',
    figsize=(7,7),
    color='palegreen',
    grid=True
)
ax.set_ylabel('Engagement Rate (%)')
plt.show()


# As we can notice, Agent works better in term of getting responses from the customers, andthen sales through Web works the second best. Let’s go ahead in breaking down this result deeperwith different customers’ attributes

# #Sales Channel & Vehicle Size
# 
# We are going to see whether customers with various vehicle sizes respond differently to different sales channels.

# In[29]:


by_sales_channel_df=df.loc[
    df['Response']=='Yes'
].groupby([
    'Sales Channel','Vehicle Size'
]).count()['Customer']/df.groupby('Sales Channel').count()['Customer']
by_sales_channel_df


# In[30]:


# Unstack the data into a more visible format

by_sales_channel_df=by_sales_channel_df.unstack().fillna(0)
by_sales_channel_df


# In[32]:


ax=(by_sales_channel_df*100.0).plot(
    kind='bar',
    figsize=(10,7),
        grid=True
)
ax.set_ylabel('Engagement Rate (%)')
plt.show()


# As we can see, customers with medium size vehicles respond the best to all sales channelswhereas the other customers differs slightly in terms of engagement rates across different saleschannels.

# In[35]:


#Engagement Rates by Months Since Policy Inception

by_months_since_inception_df=df.loc[
    df['Response']=='Yes'
].groupby(
    by='Months Since Policy Inception'
)['Response'].count()/df.groupby(
    by='Months Since Policy Inception'
)['Response'].count()*100.0
by_months_since_inception_df.fillna(0)


# In[38]:


ax=by_months_since_inception_df.fillna(0).plot(
    figsize=(10,7),
    title='Engagement Rates by Months Since Inception',
    grid=True,color='skyblue'
)

ax.set_xlabel('Months Since Policy Inception')
ax.set_ylabel('Engagement Rate (%)')
plt.show()


# In[37]:


#Customer Segmentation by CLV & Months Since Policy Inception

We are going to segment our customer base by Customer Lifetime Value and Months Since Policy Inception.


# In[39]:


# Take a look at the distribution of the CLV

df['Customer Lifetime Value'].describe()


# For the previous output, we are going to define those customers with a CLV higher than the median as high-CLV customers, and those with a CLV lower than the median aslow-CLV cus-tomers.

# In[41]:


df['CLV Segment']=df['Customer Lifetime Value'].apply(
    lambda x:'High'if x > df['Customer Lifetime Value'].median()else'Low'
)


# In[42]:


# Do the same procedure for Months Since Policy Inception

df['Months Since Policy Inception'].describe()


# In[43]:


df['Policy Age Segment']=df['Months Since Policy Inception'].apply(
    lambda x:'High'if x >df['Months Since Policy Inception'].median()else'Low'
)


# In[44]:


df.head()


# In[45]:


# Visualize these segments


# In[47]:


ax=df.loc[
    (df['CLV Segment']=='High')&(df['Policy Age Segment']=='High')
].plot.scatter(
    x='Months Since Policy Inception',
    y='Customer Lifetime Value',
    logy=True,color='red'
)



df.loc[
    (df['CLV Segment']=='Low')&(df['Policy Age Segment']=='High')
].plot.scatter(
    ax=ax,x='Months Since Policy Inception',
    y='Customer Lifetime Value',
    logy=True,
    color='blue'
)


df.loc[
    (df['CLV Segment']=='High')&(df['Policy Age Segment']=='Low')
].plot.scatter(
    ax=ax,x='Months Since Policy Inception',
    y='Customer Lifetime Value',
    logy=True,color='orange'

)

df.loc[
    (df['CLV Segment']=='Low')&(df['Policy Age Segment']=='Low')
].plot.scatter(
    ax=ax,x='Months Since Policy Inception',
    y='Customer Lifetime Value',
    logy=True,color='green',
    grid=True,figsize=(10,7)
)


# In[49]:


ax.set_ylabel('CLV (in log scale)')
ax.set_xlabel('Months Since Policy Inception')
ax.set_title('Segments by CLV and Policy Age')
plt.show()


# logy=True transform the scale to log scale and it is often used for monetary values as they often have high skewness in their values. We have repeated the code for the plot.scatter 4 times because we have created 4 segments.

# In[50]:


# See whether there is any noticeable difference in the engagement rates among these groups

engagement_rates_by_segment_df=df.loc[
    df['Response']=='Yes'
].groupby([
    'CLV Segment','Policy Age Segment'
]).count()['Customer']/df.groupby([
    'CLV Segment','Policy Age Segment'
]).count()['Customer']
engagement_rates_by_segment_df


# In[51]:


# Look at these differences in a chart
ax=(engagement_rates_by_segment_df.unstack()*100.0).plot(kind='bar',figsize=(10,7),grid=True)


# In[54]:


ax.set_ylabel('Engagement Rate (%)')
ax.set_title('Engagement Rates by Customer Segments')
plt.show()


# As we can notice,High Policy Age Segment has higher engagement than the Low Policy AgeSegment.This suggests that those customers who have been insured by this company longer re-spond better.Moreover,the High Policy Age and Low CLV segment has the highest engagementrate among the four segments.By creating different customer segments based on customer attributes, we can better under-stand how different groups of customers behave differently, and consequently, use this informa-tion to customize the marketing messages.

# In[56]:


#Add the package from the project
get_ipython().system('pip3 install scrapy')


# In[61]:


import scrapy as sp

