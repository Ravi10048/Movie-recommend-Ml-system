import numpy as np
import pandas as pd
import difflib    # to get closet match if user enter wrong data
from sklearn.feature_extraction.text import TfidfVectorizer # used to convert text /string(title,director) into numeric value
from sklearn.metrics.pairwise import cosine_similarity # to find the similarity beteen data by using similarity score
import streamlit as st
#Data Collection and Pre-Processing

# loading the data from the csv file to apandas dataframe
movies_data = pd.read_csv('movies.csv')
# st.write(movies_data)
# st.dataframe(movies_data)
# st.table(movies_data)

# selecting the relevant features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']

# replacing the null valuess with null string

for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('') # filling null value with empty string

# combining all the 5 selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

#important  converting the text data to feature vectors(numerical value)
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

#important getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)
 # compare itself to every other


#Getting the movie name from the user


#Movie Recommendation Sytem
st.markdown("# Movie Recommendation ML Model ")
movie_name = st.text_input(' Enter your favourite movie name : ')
# movie_name = input(' Enter your favourite movie name : ')
if st.button("Click Here"):

   list_of_all_titles = movies_data['title'].tolist()

   find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
   if(len(find_close_match)==0):
     st.markdown("Movie Not Available Try Something Different:")
   else:  
      close_match = find_close_match[0]

      index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

      similarity_score = list(enumerate(similarity[index_of_the_movie]))

      sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

      st.markdown('# Movies suggested for you : \n')

      i = 1

      for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index==index]['title'].values[0]
        if (i<31):
          # print(i, '.',title_from_index)
          st.markdown(str(i)+ '.'+title_from_index)
          i+=1

