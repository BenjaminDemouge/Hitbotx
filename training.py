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


def clean_up_data(df):
    #We get rid of the useless parameters 
    df = df.drop(['acousticness','instrumentalness','key','liveness','loudness','mode','speechiness','obtained_date','valence'], axis=1)
    df = df.dropna()

    # instance_id should be an integer
    df["instance_id"] = [int(x) for x in df["instance_id"]]

    #modifie the df["danceability"] to be in %
    df["danceability"] = round(df["danceability"]*100)

    #drop the  line if df["duration_ms"] is Nan
    df = df[df["duration_ms"].notnull()]
    df["duration_ms"] = df["duration_ms"].apply(lambda x: f'{round(x/1000)//60}:{round(x/1000)%60}')

    #modifie the df["energy"] to be in %
    df["energy"] = round(df["energy"]*100)

    #replace all ? by 0 in the df['tempo']
    df['tempo'] = df['tempo'].replace('?',0)
    #convert the tempo to float and round it
    df['tempo'] = df['tempo'].apply(lambda x: round(float(x)))
    df.head(10)
    df = pd.read_csv('music_genre.csv')
    df = df.drop(['acousticness','instrumentalness','key','liveness','loudness','mode','speechiness','obtained_date','valence'], axis=1)

    return df

def get_youtube(research):
    load_dotenv()
    API_KEY = os.getenv("YOUTUBE_API_KEY_3")
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=2" \
                f"&q={research}&type=video" \
                f"&key={API_KEY}"
                
    response = requests.get(url).json()
    videoId = response['items'][0]['id']['videoId']
    url = f"https://www.youtube.com/watch?v={videoId}"
    description = response['items'][0]['snippet']['description']
    date = response['items'][0]['snippet']['publishedAt']
    image = response['items'][0]['snippet']['thumbnails']['default']['url']#if you want to change the size of the image put medium or high instead of default
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
    embed = discord.Embed()
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
    embed = None

    music_name = kwargs.get('music_name',None)
    artist_name = kwargs.get('artist_name',None)
    #clean words
    print('music_name', music_name)
    print('artist_name', artist_name)
    music_name = clean_up_string(music_name)
    artist_name = clean_up_string(artist_name) if artist_name != None else None
    print('music_name', music_name)
    print('artist_name', artist_name)

    music_information = df[df['track_name'].str.lower() == music_name.lower()][df['artist_name'] == artist_name if artist_name != None else df['artist_name'] == df['artist_name']][['artist_name','track_name','music_genre','energy','danceability','popularity','duration_ms','tempo']].values
    print('music_information', music_information)
    #shuffle the music_information  
    rd.shuffle(music_information)
    if len(music_information) == 0:
        text =  "Could you be more precise about music name? and if you need help write 'help'"
    elif len(music_information) == 1:
        youtube = get_youtube(f"{music_information[0][1]} by {music_information[0][0]}")
        text = f"here are some information and a video link about **{music_name}**  wrote by **{artist_name}**:"

        embed = discord.Embed(title=youtube['title'], url=youtube['url'])
        embed.set_image(url=youtube['image'])
        embed.add_field(name="artist", value=music_information[0][0])
        embed.add_field(name="music", value=music_information[0][1])
        embed.add_field(name="genre", value=music_information[0][2])
        embed.add_field(name="energy", value=music_information[0][3])
        embed.add_field(name="danceability", value=music_information[0][4])
        embed.add_field(name="popularity", value=music_information[0][5])
        embed.add_field(name="duration", value=music_information[0][6])
        embed.add_field(name="tempo", value=music_information[0][7])
        embed.add_field(name="date", value= youtube['date'])

    else:
        youtube = get_youtube(f"{music_information[0][1]} by {music_information[0][0]}")
        text = f"As there are several **{music_name}** the information that you are "\
                f"looking for could be {music_information[0][0]}* \n\n"\
                f"here is the link to the video of the music:"
        embed = discord.Embed(title=youtube['title'], url=youtube['url'])
        embed.set_image(url=youtube['image'])
        embed.add_field(name="artist", value=music_information[0][0])
        embed.add_field(name="music", value=music_information[0][1])
        embed.add_field(name="genre", value=music_information[0][2])
        embed.add_field(name="energy", value=music_information[0][3])
        embed.add_field(name="danceability", value=music_information[0][4])
        embed.add_field(name="popularity", value=music_information[0][5])
        embed.add_field(name="duration", value = music_information[0][6])
        embed.add_field(name="tempo", value=music_information[0][7])
        embed.add_field(name="date", value= youtube['date'])

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


client = discord.Client()

df = pd.read_csv('music_genre.csv')
df = clean_up_data(df)

chatbot = Chatbot('intents.json', intent_methods = {'artist': artist, 'genre': genre, 'information': information, 'recommandation': recommandation})
chatbot.train_model()


@client.event
async def on_message(message):
    if message.author == client.user:
        return    

    if isinstance(chatbot.request(message.content), dict):
        print('IIIIIIIIIIIIIIIFFFFFFFFFFFFF')
        response = chatbot.request(message.content)['text']
        print('response: ', response)
        embed = chatbot.request(message.content)['embed']
        print('embed: ', embed)
        await message.channel.send(response, embed=embed)
    else:
        print('EEEEEEEEEEEESSSSSSSSSSSSSSSSLLLLLLLLLLLLLEEEEEEEEEE')
        response = chatbot.request(message.content)
        await message.channel.send(response) 
    

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')


load_dotenv()
client.run(os.getenv('DISCORD_TOKEN'))