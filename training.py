import discord
import os
from dotenv import load_dotenv
from chatbot import Chatbot
import pandas as pd
import random as rd
import requests #used in get_youtube
import numpy as np 
import datetime
import matplotlib.pyplot as plt

import nltk
from nltk.stem import WordNetLemmatizer

def clean_up_data(df):

    #We get rid of the useless parameters 
    df = df.drop(['acousticness','instrumentalness','key','liveness','loudness','mode','speechiness','obtained_date','valence'], axis=1)
    df = df.dropna()

    # instance_id should be an integer
    df["instance_id"] = [int(x) for x in df["instance_id"]]

    #modifie the df["danceability"] to be in %
    df["danceability"] = round(df["danceability"]*100)

    #drop the  line if df["duration_ms"] is Nan and modifie the df["duration_ms"] to be in minutes
    df = df[df["duration_ms"].notnull()]
    df["duration_ms"] = df["duration_ms"].apply(lambda x: f'{round(x/1000)//60}:{round(x/1000)%60}')
    df = df.rename(columns={"duration_ms": "duration"})

    #modifie the df["energy"] to be in %
    df["energy"] = round(df["energy"]*100)

    #replace all ? by 0 in the df['tempo']
    df['tempo'] = df['tempo'].replace('?',0)
    #convert the tempo to float and round it
    df['tempo'] = df['tempo'].apply(lambda x: round(float(x)))

    return df

def get_youtube(research):
    load_dotenv()
    API_KEY = os.getenv("YOUTUBE_API_KEY_2")
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=2" \
                f"&q={research}&type=video" \
                f"&key={API_KEY}"
                
    response = requests.get(url).json()
    videoId = response['items'][0]['id']['videoId']
    url = f"https://www.youtube.com/watch?v={videoId}"
    description = response['items'][0]['snippet']['description']
    date = response['items'][0]['snippet']['publishedAt']
    image = response['items'][0]['snippet']['thumbnails']['high']['url']#if you want to change the size of the image put medium or high instead of default
    title = response['items'][0]['snippet']['title']
    return {'url': url, 'description': description, 'image' : image, 'date' : date, 'title' : title}

def clean_up_string(string):
    #clean string drop the potential question mark at the end of the music name
    if ' ?' in string:
        string = string.replace(' ?','')
    elif '?' in string:
        string = string.replace('?','')
    return string

def artist(**kwargs):
    music_name = kwargs.get('music_name',None)
    music_name = clean_up_string(music_name)
    text, embed = None, None
    music_artist = df[df['track_name'].str.lower() == music_name.lower()][['artist_name','track_name']].values
    #shuffle the artist name
    rd.shuffle(music_artist)
    if len(music_artist) == 0:
        text =  "Could you be more precise about the music name?"#autre error possible to code
    elif len(music_artist) == 1:
        youtube = get_youtube(f"{music_artist[0][1]} by {music_artist[0][0]}")
        embed = discord.Embed(title=youtube['title'],  url=youtube['url'])
        embed.set_image(url=youtube['image'])
        embed.add_field(name="artist", value=music_artist[0][0])
        embed.add_field(name="music", value=music_artist[0][1])

    else:
        text = f"As there are several **{music_name}** music the artist that you are "\
                f"looking for could be {music_artist[0][0]}* \n\n"\
                f"If you want to know more about this music ask for information, "\
                f"otherwise here is the link to the video of the music:"
        youtube = get_youtube(f"{music_artist[0][1]} by {music_artist[0][0]}")
        embed = discord.Embed(title=youtube['title'],  url=youtube['url'])
        embed.set_image(url=youtube['url'])
        embed.add_field(name="artist", value=music_artist[0][0])
        embed.add_field(name="music", value=music_artist[0][1])

    return {'text': text, 'embed': embed}

def genre(**kwargs):
    
    music_name = kwargs.get('music_name',None)
    music_name = clean_up_string(music_name)

    music_genres_artist = df[df['track_name'].str.lower() == music_name.lower()][['music_genre','artist_name']].values
    text = ""
    text, embed = None, None
    #shuffle the music_genres_artist
    rd.shuffle(music_genres_artist)
    if len(music_genres_artist) == 0:
        text =  "Could you be more precise about the music name?"
    elif len(music_genres_artist) == 1:
        youtube = get_youtube(f"{music_name} by {music_genres_artist[0][1]}")      
        embed = discord.Embed(title=youtube['title'],  url=youtube['url'])
        embed.set_image(url=youtube['image'])
        embed.add_field(name="genre", value=music_genres_artist[0][0])
    else:
        text = f"As there are several **{music_name}** musics the one that you are looking for "\
                f"could be wrote by *{music_genres_artist[0][1]}* and its genre "\
                f"is **{music_genres_artist[0][0]}** \n\nIf you want to know more about this music "\
                f"ask for information, otherwise here is the link to the video of the music:"

        youtube = get_youtube(f"{music_name} by {music_genres_artist[0][1]}")
        embed = discord.Embed(title=youtube['title'],  url=youtube['url'])
        embed.set_image(url=youtube['image'])
        embed.add_field(name="genre", value=music_genres_artist[0][0])

    return {'text': text, 'embed': embed}

def information(**kwargs):
    text, embed = None, None

    music_name = kwargs.get('music_name',None)
    artist_name = kwargs.get('artist_name',None)
    #clean words
    print('music_name', music_name)
    print('artist_name', artist_name)
    music_name = clean_up_string(music_name)
    artist_name = clean_up_string(artist_name) if artist_name != None else None
    print('music_name', music_name)
    print('artist_name', artist_name)
    music_info = df[df['track_name'].str.lower() == music_name.lower()]
    print('music_info\n', music_info)
    music_info = music_info[music_info['artist_name'].str.lower() == artist_name.lower()] if artist_name != None else music_info
    print('music_info\n', music_info)
    music_information = music_info.values
    print('music_information', music_information)
    #shuffle the music_information  
    rd.shuffle(music_information)
    if len(music_information) == 0:
        text =  "Could you be more precise about music name? and if you need help write 'help'"
    elif len(music_information) == 1:
        youtube = get_youtube(f"{music_information[0][2]} by {music_information[0][1]}")
        text = f"here are some information and a video link about **{music_name}** by **{artist_name}**:"

        embed = discord.Embed(title=youtube['title'], url=youtube['url'])
        embed.set_image(url=youtube['image'])
        embed.add_field(name="artist", value=music_information[0][1])
        embed.add_field(name="music", value=music_information[0][2])
        embed.add_field(name="popularity", value=music_information[0][3])
        embed.add_field(name="danceability", value=music_information[0][4])
        embed.add_field(name="duration", value=music_information[0][5])
        embed.add_field(name="energy", value=music_information[0][6])
        embed.add_field(name="tempo", value=music_information[0][7])
        embed.add_field(name="genre", value=music_information[0][8])
        embed.add_field(name="date", value=youtube['date'])


    else:
        youtube = get_youtube(f"{music_information[0][2]} by {music_information[0][1]}")
        text = f"As there are several **{music_name}** the information that you are "\
                f"looking for could be {music_information[0][1]}* \n\n"\
                f"here is the link to the video of the music:"
        embed = discord.Embed(title=youtube['title'], url=youtube['url'])
        embed.set_image(url=youtube['image'])
        embed.add_field(name="artist", value=music_information[0][1])
        embed.add_field(name="music", value=music_information[0][2])
        embed.add_field(name="popularity", value=music_information[0][3])
        embed.add_field(name="danceability", value=music_information[0][4])
        embed.add_field(name="duration_ms", value=music_information[0][5])
        embed.add_field(name="energy", value=music_information[0][6])
        embed.add_field(name="tempo", value=music_information[0][7])
        embed.add_field(name="genre", value=music_information[0][8])
        embed.add_field(name="date", value=youtube['date'])

    return {'text': text, 'embed': embed}

def recommandation(userID):
    text = None
    embed = None
    df_user = df[df['user_id'] == userID]
    if len(df_user) <= 3:
        text =  "You have not yet looked for enough music, so I can't recommend you anything"
    else:
        df_user = df_user.sort_values(by=['popularity'], ascending=False)
        youtube = get_youtube(f"{df_user['track_name'].values[0]} by {df_user['artist_name'].values[0]}")
        text = "Here are some musics that I think you would like:"
        embed = discord.Embed(title=youtube['title'], url=youtube['url'])
        embed.set_image(url=youtube['image'])
        embed.add_field(name="artist", value=df_user['artist_name'].values[0])
        embed.add_field(name="music", value=df_user['track_name'].values[0])
        embed.add_field(name="genre", value=df_user['music_genre'].values[0])
        embed.add_field(name="energy", value=df_user['energy'].values[0])
        embed.add_field(name="danceability", value=df_user['danceability'].values[0])
        embed.add_field(name="popularity", value=df_user['popularity'].values[0])
        embed.add_field(name="duration", value=df_user['duration_ms'].values[0])
        embed.add_field(name="tempo", value=df_user['tempo'].values[0])
        embed.add_field(name="date", value= youtube['date'])

    return {'text': text, 'embed': embed}

def set_up_random_feature(features, feature_name):
    feature = {'min': 0, 'max': 100}
    if features[f'{feature_name}'] == 'high':
       feature['min'] = df[f'{feature_name}'].mean() 
       feature['max'] = df[f'{feature_name}'].max()
    elif features[f'{feature_name}'] == 'low':
        feature['min'] = df[f'{feature_name}'].min()
        feature['max'] = df[f'{feature_name}'].mean()
    elif features[f'{feature_name}'] == 'medium':
        feature['min'] = df[f'{feature_name}'].quantile(0.25)
        feature['max'] = df[f'{feature_name}'].quantile(0.75)

    return feature

def clean_up_random_infos(infos):
    features = {}
    if infos != None:
        #tokemize the string
        infos_words = nltk.word_tokenize(infos)
        # if energy is in the string look for the word just before and return add to the energy key the value of that word into features
        if 'energy' in infos_words:
            energy_index = infos_words.index('energy')
            if energy_index > 0:
                features['energy'] = infos_words[energy_index - 1]
        # if tempo is in the string look for the word just before and return add to the tempo key the value of that word into features
        if 'tempo' in infos_words:
            tempo_index = infos_words.index('tempo')
            if tempo_index > 0:
                features['tempo'] = infos_words[tempo_index - 1]
        # if popularity is in the string look for the word just before and return add to the pupolarity key the value of that word into features
        if 'popularity' in infos_words:
            popularity_index = infos_words.index('popularity')
            if popularity_index > 0:
                features['popularity'] = infos_words[popularity_index - 1]
        # if danceability is in the string look for the word just before and return add to the danceability key the value of that word into features
        if 'danceability' in infos_words:
            danceability_index = infos_words.index('danceability')
            if danceability_index > 0:
                features['danceability'] = infos_words[danceability_index - 1]


    danceability, energy, tempo, popularity = None, None, None, None

    

    # if the value in the features is equal to high modifie the min value to df[feature].mean()
    energy = set_up_random_feature(features, 'energy') if 'energy' in features else None
    tempo = set_up_random_feature(features, 'tempo') if 'tempo' in features else None
    popularity = set_up_random_feature(features, 'popularity') if 'popularity' in features else None
    danceability = set_up_random_feature(features, 'danceability') if 'danceability' in features else None

    return {'danceability': danceability, 'energy': energy, 'tempo': tempo, 'popularity': popularity}

def clean_up_genre(genre): # clean up the genre by tokenizing it to words
    genres = []
    genre_words = nltk.word_tokenize(genre) # tokenize the genre into words
    #list of all music_genre from df
    music_genre = df['music_genre'].unique()
    #list of all music_genre from df in lower case
    music_genre_lower = [x.lower() for x in music_genre]
    genre_words = [x.lower() for x in genre_words]
    #look the word inside the genre_words list and if it is in the music_genre_lower list then return the genre
    for word in genre_words:
        if word.lower() in music_genre_lower:
            genres.append(word)

    return genres # return the genres' list

def random_music(**kwargs):
    text, embed = None, None
    genre = kwargs.get('genre',None)
    print('genre : ', genre)
    infos = kwargs.get('information',None)
    print('infos : ', infos)
    danceability = clean_up_random_infos(infos)['danceability']
    energy = clean_up_random_infos(infos)['energy']
    tempo = clean_up_random_infos(infos)['tempo']
    popularity = clean_up_random_infos(infos)['popularity']
    genres = clean_up_genre(genre) if genre != None else None


    if infos == None and genres == None:
        df_random = df.sample(n=1)
        text = "Here is a random music:"
    elif genres != None and infos == None:
        df_random = df[df['music_genre'].str.lower().isin(genres)]
        df_random = df_random.sample(n=1)
        music_genre_chosen = df_random["music_genre"].values[0]
        text = f"Here is a random {music_genre_chosen} music:"
    else:
        text = "Here is a random music with all the specification that you asked:"
        df_random = df
        df_random = df_random[df_random['music_genre'].str.lower().isin(genres)] if genres != None else df_random
        df_random = df_random[df_random['danceability'] >= danceability['min']][df_random['danceability']<= danceability['max']] if danceability != None else df_random
        df_random = df_random[df_random['energy'] >= energy['min']][df_random['energy']<= energy['max']] if energy != None else df_random
        df_random = df_random[df_random['tempo'] >= tempo['min']][df_random['tempo']<= tempo['max']] if tempo != None else df_random
        df_random = df_random[df_random['popularity'] >= popularity['min']][df_random['popularity']<= popularity['max']] if popularity != None else df_random
        df_random = df_random.sample(n=1)

    
    youtube = get_youtube(f"{df_random['track_name'].values[0]} by {df_random['artist_name'].values[0]}")

    embed = discord.Embed(title=youtube['title'], url=youtube['url'])
    embed.set_image(url=youtube['image'])
    embed.add_field(name="artist", value=df_random['artist_name'].values[0])
    embed.add_field(name="music", value=df_random['track_name'].values[0])
    embed.add_field(name="genre", value=df_random['music_genre'].values[0])
    embed.add_field(name="energy", value=df_random['energy'].values[0])
    embed.add_field(name="danceability", value=df_random['danceability'].values[0])
    embed.add_field(name="popularity", value=df_random['popularity'].values[0])
    embed.add_field(name="duration", value=df_random['duration'].values[0])
    embed.add_field(name="tempo", value=df_random['tempo'].values[0])
    embed.add_field(name="date", value= youtube['date'])

    return {'text': text, 'embed': embed}

client = discord.Client()

df = pd.read_csv('music_genre.csv')
df = clean_up_data(df)
print(df.head())

chatbot = Chatbot('intents.json', 
                    intent_methods = {'artist': artist, 
                                        'genre': genre, 
                                        'information': information,
                                        'random_music': random_music, 
                                        'recommandation': recommandation})
chatbot.train_model()

rating_emojis = ['0️⃣', '1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣']

def emoji_to_number(emoji):
    if emoji not in rating_emojis:
        return None
    return rating_emojis.index(emoji)

async def display_reaction(message):
    #display all the emojis' number on the message
    for emoji in rating_emojis:
        await message.add_reaction(emoji)

@client.event
async def on_message(message):
    if message.author == client.user:
        # if the message contains an embed add emoji to it

        if message.embeds:
            for emoji in rating_emojis:
                await message.add_reaction(emoji)
        return    

    if isinstance(chatbot.request(message.content), dict):
        response = chatbot.request(message.content)['text']
        print('response: ', response)
        embed = chatbot.request(message.content)['embed']
        print('embed: ', embed)
        await message.channel.send(response, embed=embed)
        #send emoji to the embed message
        await message.add_reaction('👍')
    else:
        response = chatbot.request(message.content)
        await message.channel.send(response, embed=None)


df_users = pd.DataFrame({'username':[], 
                        'instance_id':[],	
                        'artist_name':[],	
                        'track_name':[],
                        'music_genre':[],
                        'energy':[],
                        'danceability':[],
                        'popularity':[],
                        'duration':[],
                        'tempo':[], 
                        'rating':[]})  

async def add_rating(user_id, music_information, rating):
    print('music_information: ', music_information)
    #add a new row to the dataframe
    df_users.loc[len(df_users)] = [user_id,
                                    music_information[0]['instance_id'],
                                    music_information[0]['artist_name'],
                                    music_information[0]['track_name'],
                                    music_information[0]['music_genre'],
                                    music_information[0]['energy'],
                                    music_information[0]['danceability'],
                                    music_information[0]['popularity'],
                                    music_information[0]['duration_ms'],
                                    music_information[0]['tempo'],
                                    rating]
    df_users.to_csv('users.csv', index=False)

async def add_ratings(username, rating):
    df_users.loc[len(df_users)] = [username,0,0,0,0,0,0,0,0,0, rating]
    df_users.to_csv('users.csv', index=False)


@client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):#recupère le message pour avoir les infos sur la music pour ajouter les infos dans le df_users

    if payload.user_id == client.user.id:
        return
    if payload.emoji.name in rating_emojis:
        print('AAAAAAAA----------AAAAAAAAA')
        print('payload: ', payload)
        print('AAAAAAAA----------AAAAAAAAA')
        await add_ratings(username = f'{payload.member.name}#{payload.member.discriminator}', rating = emoji_to_number(payload.emoji.name))

    
@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')


load_dotenv()
client.run(os.getenv('DISCORD_TOKEN'))

