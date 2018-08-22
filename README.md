# Record Union Instructions:

### Use case
The scenario we are trying to solve is detecting fraudulent streaming behavior from a Music Service report. Fraudulent streaming might look like this: a person releases music through Record Union, then automates streaming of the released track(s) in the Music Service using a botnet. Based on usage data received from a Music Service, we want to be able to automatically detect users and releases engaging in fraud.

### Data
A report consists of three gzip compressed files, containing data for streaming events for Record Union releases on a particular day. The files are stored in the provided Google Cloud Storage bucket. The data is partially based on real data, which is why many fields are hashed or integer-coded, and names are randomly generated. The fields "track_id" and "user_id" are unique identifiers for tracks and users, respectively, in the Music Service.

### Streams
Filename: streams/2017/09/09/allcountries
(device_type, length, os, timestamp, track_id, user_id)
One entry in this file corresponds to one stream in the Music Service. The field timestamp refers to the time when the stream was recorded by the Music Service, and length is the duration of the stream in seconds.

### Users
Filename: users/2017/09/09
(access, birth_year, country, gender, user_id)
One entry for each Music Service user present in the Streams file.

### Tracks
Filename: tracks/2017/09/09
(album_artist, album_code, album_name, track_id, track_name) One entry for each track present in the Streams file.

### Expected outcomes
Version-controlled code
Visualizations and discussion of the solution
Technical constraints
The data needs to be read from, and any output from algorithms written into Google Cloud Storage. All code should be versioned with Git.
DataFrames in Pandas should be used for data transformations.
The implementation should use Apache Airflow to orchestrate the processing from data to results.
