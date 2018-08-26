
# coding: utf-8

# # Music streaming fraud detection analysis
# 
# ### Use case
# The scenario is detecting fraudulent streaming behavior from a Music Service report. Fraudulent streaming might look like this: a person releases music, then automates streaming of the released track(s) in the Music Service using a botnet. Based on usage data received from a Music Service, we want to be able to automatically detect users and releases engaging in fraud.
# 
# ### Data
# This report consists of data for streaming events for releases on a particular day. The files are stored in the provided Google Cloud Storage bucket. The data is partially based on real data, which is why many fields are hashed or integer-coded, and names are randomly generated. The fields "track_id" and "user_id" are unique identifiers for tracks and users, respectively, in the Music Service.
# 
# * **Streams** Filename: streams/2017/09/09/allcountries. (device_type, length, os, timestamp, track_id, user_id) One entry in this file corresponds to one stream in the Music Service. The field timestamp refers to the time when the stream was recorded by the Music Service, and length is the duration of the stream in seconds.
# 
# * **Users** Filename: users/2017/09/09. (access, birth_year, country, gender, user_id) One entry for each Music Service user present in the Streams file.
# 
# * **Tracks** Filename: tracks/2017/09/09. (album_artist, album_code, album_name, track_id, track_name) One entry for each track present in the Streams file.

# Import packages

# In[143]:


import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import datetime
from sklearn import preprocessing
get_ipython().magic('matplotlib inline')


# Set working directory

# In[144]:


os.chdir('/Users/ahnate/Files/DataScience/RecordUnionFraudDetection/working')


# Define function for importing JSON data to pandas

# In[145]:


def import_dat(input):
    dat = open(input, "r").read()
    # Replace \n with commas and format as list for json.loads
    dat = json.loads("[" + dat.replace("}\n{", "},\n{") + "]")   
    dat = pd.DataFrame.from_dict(dat, orient='columns')  # convert from json to pandas
    return dat


# Import data

# In[146]:


tracks = import_dat("../input/tracks-2017-09-09")
users = import_dat("../input/users-2017-09-09")
streams = import_dat("../input/streams-2017-09-09-allcountries")


# ## Look at the data and do some preprocessing

# In[147]:


tracks.head()


# In[148]:


users.head()


# Preprocessing for `users`

# In[149]:


users.dtypes


# In[150]:


# set birth_year to numeric and country to categorical
users['birth_year'] = pd.to_numeric(users['birth_year'], errors='coerce')
users['country'] = users['country'].astype(object)
# compute age column
users['age'] = 2017 - users['birth_year']


# Let's look at `streams` next

# In[151]:


streams.head()


# Preprocessing for `streams`

# In[152]:


# Convert from milliseconds to seconds
streams['timestamp'] = streams['timestamp']/1000.0
# Convert from UNIX epoch to datetime format
streams['timestamp'] = streams['timestamp'].apply(datetime.datetime.fromtimestamp)
streams['timestamp'].head()


# ## Merge data into a single dataframe and do exploratory analysis

# In[153]:


dat = pd.merge(streams, tracks, how='left', on='track_id')
dat = pd.merge(dat, users, how='left', on='user_id')
dat.info()


# Descriptives for numerical variables
# * Note: 50% corresponds to the median, and is the best measure of central tendency in this dataset, since the data is highly skewed

# In[154]:


dat.describe()


# Look at counts and unique values

# In[155]:


dat[:].astype('object').describe()


# ### Let's look at some histograms

# In[156]:


dat['length'].hist()


# In[157]:


dat['timestamp'].dt.hour.hist()


# In[158]:


dat['device_type'].hist()


# In[159]:


dat['os'].hist()


# In[160]:


dat['access'].hist()


# In[161]:


dat['age'].hist()


# In[162]:


dat['gender'].hist()


# ## Feature engineering: aggregate by `user_id`
# 
# Now we'll engineer some key features that can help us flag fraudulent (bot) streaming activity. We'll do this by creating aggregate statistics grouped by `user_id`

# In[163]:


datByUser = dat.groupby('user_id').agg({'timestamp':["count", lambda x: max(x) - min(x)],
                            'length': ["nunique", sum, min, max, lambda x: max(x) - min(x)],
                            'track_id': "nunique",
                            'album_code': "nunique",
                            'device_type': "first",
                            'os': "first",
                            'access': "first",
                            'country': "first",
                            'gender': "first",
                            'age': "first"
                           })
# Drop one level of column index
datByUser.columns = datByUser.columns.droplevel(level=0)
# Rename columns
datByUser.columns = ["time.n", "time.range", "len.nuniq", "len.sum", "len.min", "len.max", "len.range",
                    "track.nuniq", "album.nuniq", "dev_type", "os", "access", "country", "gender", "age"                     
                    ]
datByUser.head()


# The new engineered features include:
# * `time.n`  Total number of streams initiated by user in this dataset (24hrs)
# * `time.range` Time window between the first and last stream for each user (note 15min temporal resolution)
# * `len.nuniq` Count of unique listening times (e.g. if a bot streams 5 songs, but only listens to each song for a length of 60s this value will be 1. If it streams all songs for either 55s or 60s this value will be 2)
# * `len.sum` Total streaming time
# * `len.min` Shortest streaming time 
# * `len.max` Longest streaming time
# * `len.range` Range between shortest and longest streaming time
# * `track.nuniq` Count of total unique tracks (e.g. if a bot streamed the same track 100 times, this value will be 1)
# * `album.nuniq` Count of total unique albums

# ## What's a "normal" amount of streaming per day?
# 
# Let's look statistics for number of tracks streamed (`time.n`) and total streaming time per user (`len.sum`). We'll look at the median and quantiles (instead of mean and standard deviation) since the data is highly skewed.

# In[164]:


datByUser[["time.n","len.sum"]].describe().loc[["min","25%","50%","75%","max"]]


# Very interesting that 75% of users stream 2 songs or less per day, and stream only 354 seconds or less. Up to the 99 percentile of users stream only 16 tracks or less per day:

# In[165]:


datByUser["time.n"].quantile(.99)


# Let's look at it from a more extreme case: for example, how much would a really heavy user stream on average?
# * Assuming an active listening window (maximum) of 12 hours (43200s) per day, how many songs would a human user stream on average per day?
# * We'll say the average song length is: 230s
# https://www.statcrunch.com/5.0/viewreport.php?groupid=948&reportid=28647
# * So non-stop listening within a 12hr window could be about 43200/230 = 188 songs.
# * Of course, not all streaming bots would neccessarily maximize their bandwith (streaming for 24hrs a day, for example). More nuanced bots might only stream for 4 to 6 hours a day but use different user IDs (see below for data supporting this).

# ## Flagging for fraud
# After doing some exploration on the high streaming users (I ended up using a flagging criteria for upwards of 150 songs per user per day), I made a few observations on certain "bot" signatures:
# * There appears to be a bot that streams 10 albums and 100 tracks (±1) per day. It also tends to stream only in durations between 51 and 73 seconds (give or take a few). This bot operates on a desktop browser and tends to run somewhere between 4 to 6 hours per day on the free tier. Example:

# In[166]:


datByUser[datByUser["time.n"].ge(150) & datByUser["album.nuniq"].isin(range(9,11)) & datByUser["track.nuniq"].isin(range(99,101))].head(6)


# * A bot that streams in one of two durations from only one album, usually under 60 or 75 seconds (see below)
# * A bot that streams all the songs in only one album repeatedly (see below, row 7 & 8)

# In[167]:


datByUser[datByUser["time.n"].ge(150) & datByUser["album.nuniq"].isin(range(1,2))].head(10)


# # Final query
# I ended up using the final query below, which catches all of the above cases (possibly others as well). It outputs 630 unique user IDs for potential fraud.

# In[168]:


flagged_dat = datByUser[datByUser["time.n"].ge(150)]  # upwards of 150 songs per user per day
flagged_dat = flagged_dat[
    flagged_dat["len.nuniq"].le(10) |  # total unique streaming durations less than 10 counts
    flagged_dat["len.max"].le(80) |  # max streaming duration less than 80 seconds
    (flagged_dat["len.nuniq"] == flagged_dat["track.nuniq"])  # unique duration counts same as unique track counts
    ]
flagged_dat.head(20)


# In[169]:


# first 10 flagged user IDs for output below
flagged_dat.index.values[1:10]


# ## Output
# Write the flagged user IDs to `flagged_user_id.txt` in current working directory

# In[177]:


outfile = open('flagged_user_id_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt', 'w')
for item in flagged_dat.index.values:
  outfile.write("%s\n" % item)
outfile.close()


# ## Notes and limitations
# * Given that this analysis yields us a **labeled dataset** at the end (pending perhpas further confirmation of fraud from the music companies, etc.), there is a possibility of now training a machine learning model (e.g. lightGBM or XGBoost) on this labeled dataset, and then using this model on the dataset again (in lieu of the query). The benefit here is that a machine learning model may pick up on other predictive features in the dataset that have not yet been noted.
# * There is also the question of whether we want to flag users that stream tracks for longer periods of time (e.g. for 24hrs, perhaps as background music?). Is this acceptable use? For example: User_id `102a0cd1bc18dffd6e8462c6bf0af60860b19dda` listened to 19 tracks from 19 albums (perhaps a compilation) for the entire day (23.5 hrs, 172 total streams):

# In[171]:


datByUser[datByUser.index == "102a0cd1bc18dffd6e8462c6bf0af60860b19dda"]


# In[172]:


dat[dat["user_id"] == "102a0cd1bc18dffd6e8462c6bf0af60860b19dda"]

