import numpy as np
import pandas as pd
import difflib    # to get closet match if user enter wrong data
from sklearn.feature_extraction.text import TfidfVectorizer # used to convert text /string(title,director) into numeric value
from sklearn.metrics.pairwise import cosine_similarity # to find the similarity beteen data by using similarity score
import pickle
#Data Collection and Pre-Processing

# loading the data from the csv file to apandas dataframe
movies_data = pd.read_csv('movies.csv')

# printing the first 5 rows of the dataframe
print(movies_data.head())

# number of rows and columns in the data frame
print(movies_data.shape)

# selecting the relevant features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)

# replacing the null valuess with null string

for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('') # filling null value with empty string


# combining all the 5 selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
print(combined_features)

#important  converting the text data to feature vectors(numerical value)
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
print(feature_vectors)

#important getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)
print(similarity) # compare itself to every other
print(similarity.shape)

#Getting the movie name from the user

#Movie Recommendation Sytem

movie_name = input(' Enter your favourite movie name : ')
#Enter your favourite movie name : iron man

# creating a list with all the movie names given in the dataset
list_of_all_titles = movies_data['title'].tolist()
# print(list_of_all_titles)

#finding the close match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
# print(find_close_match) #['Iron Man', 'Iron Man 3', 'Iron Man 2']

close_match = find_close_match[0]
# print(close_match) # Iron Man


# finding the index of the movie with title
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
# print(index_of_the_movie) # this done because we need to find similarity with index no.

# getting a list of similar movies
similarity_score = list(enumerate(similarity[index_of_the_movie])) # enumerate - loop
# print(similarity_score)# (index,similarity score)
# len(similarity_score)

# sorting the movies based on their similarity score
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) # sorting by 2nd value
# print(sorted_similar_movies)

# print the name of similar movies based on the index

print('Movies suggested for you : \n')

i = 1
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1

'''Movies suggested for you : 

1 . Iron Man
2 . Iron Man 2
3 . Iron Man 3
4 . Avengers: Age of Ultron
5 . The Avengers
6 . Captain America: Civil War
7 . Captain America: The Winter Soldier
8 . Ant-Man
9 . X-Men
10 . Made
11 . X-Men: Apocalypse
12 . X2
13 . The Incredible Hulk
14 . The Helix... Loaded
15 . X-Men: First Class
16 . X-Men: Days of Future Past
17 . Captain America: The First Avenger
18 . Kick-Ass 2
19 . Guardians of the Galaxy
20 . Deadpool
21 . Thor: The Dark World
22 . G-Force
23 . X-Men: The Last Stand
24 . Duets
25 . Mortdecai
26 . The Last Airbender
27 . Southland Tales
28 . Zathura: A Space Adventure
29 . Sky Captain and the World of Tomorrow
'''    

# #Movie Recommendation Sytem

# movie_name = input(' Enter your favourite movie name : ')

# list_of_all_titles = movies_data['title'].tolist()

# find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

# close_match = find_close_match[0]

# index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

# similarity_score = list(enumerate(similarity[index_of_the_movie]))

# sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

# print('Movies suggested for you : \n')

# i = 1

# for movie in sorted_similar_movies:
#   index = movie[0]
#   title_from_index = movies_data[movies_data.index==index]['title'].values[0]
#   if (i<30):
#     print(i, '.',title_from_index)
#     i+=1

# '''
# Enter your favourite movie name : bat man
# Movies suggested for you : 

# 1 . Batman
# 2 . Batman Returns
# 3 . Batman & Robin
# 4 . The Dark Knight Rises
# 5 . Batman Begins
# 6 . The Dark Knight
# 7 . A History of Violence
# 8 . Superman
# 9 . Beetlejuice
# 10 . Bedazzled
# 11 . Mars Attacks!
# 12 . The Sentinel
# 13 . Planet of the Apes 
# 14 . Man of Steel
# 15 . Suicide Squad
# 16 . The Mask
# 17 . Salton Sea
# 18 . Spider-Man 3
# 19 . The Postman Always Rings Twice
# 20 . Hang 'em High
# 21 . Spider-Man 2
# 22 . Dungeons & Dragons: Wrath of the Dragon God
# 23 . Superman Returns
# 24 . Jonah Hex
# 25 . Exorcist II: The Heretic
# 26 . Superman II
# 27 . Green Lantern
# 28 . Superman III
# 29 . Something's Gotta Give
# '''    

