import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.feature_extraction.text import TfidfVectorizer
import openai

#sns.set_palette(sns.color_palette(['#FF7F50']))
sns.set_palette(sns.color_palette(['#009682'])) 
color = sns.color_palette()[0]

# Title and description
st.set_page_config(page_title="BASF Job Review Dashboard", layout="wide")
st.title("BASF Job Review Dashboard")
#st.write("This dashboard presents an analysis of job reviews.")


# Load the data
df = pd.read_csv('./data/processed_data.csv')

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Create columns
col1, col2 = st.columns(2)

min_year = int(df['date'].dt.year.min())
max_year = int(df['date'].dt.year.max())
year_range = col1.slider("Select a Time Range", min_year, max_year, (min_year, max_year))

# Filter the data for the selected time range
df_time_range = df[(df['date'].dt.year >= year_range[0]) & (df['date'].dt.year <= year_range[1])]

# Generate unique job categories
job_categories = df['job_category'].unique().tolist()

# Append 'all' option to the list
job_categories = ['Alle'] + job_categories

# Generate the select box for the job categories
selected_job_category = col2.selectbox(
    "Select Job Category",
    options=job_categories
)


# Filter the dataframe for the selected job category
if selected_job_category != 'Alle':
    df_filtered = df_time_range[df_time_range['job_category'] == selected_job_category]
else:
    df_filtered = df_time_range

# Calculate average rating and review count for the current filter
average_rating = round(df_filtered['rating'].mean(), 2)
average_sentiment =round(df_filtered['polarity'].mean(), 2)
average_subjectivity = round(df_filtered['subjectivity'].mean(), 2)
review_count = len(df_filtered)

complete_subjectivity = round(df['subjectivity'].mean(), 2)

delta_review_count = round((review_count-len(df))/len(df)*100, 2)
delta_subjectivity = round((average_subjectivity-complete_subjectivity)/complete_subjectivity*100, 2)


# Assign an emoji based on the average rating
if average_rating >= 4.5:
    emoji = ":heart_eyes:"
elif average_rating >= 4.0:
    emoji = ":blush:"
elif average_rating >= 3.5:
    emoji = ":smiley:"
elif average_rating >= 3.0:
    emoji = ":slightly_smiling_face:"
elif average_rating >= 2.5:
    emoji = ":neutral_face:"
elif average_rating >= 2:
    emoji = ":worried:"
else:
    emoji = ":angry:"

# Display labels and KPIs
col2_1, col2_2, col2_3, col2_4= col2.columns(4)

col2_1.metric(label=f"**Average Rating** {emoji}", value=f"{average_rating}", delta=round(average_rating-df['rating'].mean(), 2))
col2_2.metric(label="**Review Count**", value=review_count, delta=f"{delta_review_count}%")
col2_3.metric(label="**Average Sentiment**", value=average_sentiment, delta=round(average_sentiment-df['polarity'].mean(), 2))
col2_4.metric(label="**Average Subjectivity**", value=average_subjectivity, delta=f"{delta_subjectivity}%", delta_color="inverse")


# Generate the plot for sentiment over time
df_filtered = df_filtered.sort_values(by='date')
if not df_filtered.empty:
    df_resampled = df_filtered.resample('M', on='date').mean()
    fig2, ax2 = plt.subplots(figsize=(8, 2))
    sns.lineplot(data=df_resampled, x='date', y='polarity', color = color)
    plt.xlabel('Date')
    plt.ylabel('Polarity')
else:
    fig2, ax2 = plt.subplots(figsize=(8, 2))
    plt.xlabel('Date')
    plt.ylabel('Polarity')

# Display the plot in col1
col1.subheader("Sentiment Development")
col1.pyplot(fig2)


# Generate the plot
fig, ax = plt.subplots(figsize=(8, 2))
sns.countplot(data=df_filtered, x='rating', order=np.arange(1, 6), color=color, edgecolor='black')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')

# Ensure x-axis always shows rating 1 to 5
plt.xticks(ticks=np.arange(5), labels=np.arange(1, 6))

# Ensure y-axis always shows integers
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Display the plot in col1
col1.subheader("Rating Distribution")
col1.pyplot(fig)

# Create TF-IDF Values
col2.subheader("Most Important Terms", help='Extract the most important words of the \'Pros\' and \'Cons\' of the reviews based on TF-IDF. You can switch between Unigrams and Bigrams')

def get_unigrams(corpus, n=5):
    vec = TfidfVectorizer(ngram_range=(1,1)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_bigrams(corpus, n=5):
    vec = TfidfVectorizer(ngram_range=(2,2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

col2_1, col2_2, col2_3, col2_4 = col2.columns(4)
unigram_button = col2_1.button("Unigrams")
bigram_button = col2_2.button("Bigrams")

def create_table(unigrams=True):
    if not df_filtered.empty:
        if unigrams:
            pros = get_unigrams(df_filtered['processed_pros'])
            cons = get_unigrams(df_filtered['processed_cons'])
        else:
            pros = get_bigrams(df_filtered['processed_pros'])
            cons = get_bigrams(df_filtered['processed_cons'])
        # Create dataframes
        df_pros = pd.DataFrame(pros, columns=['Pros', 'TF-IDF'])
        df_cons = pd.DataFrame(cons, columns=['Cons', 'TF-IDF'])
        # Concatenate the dataframes horizontally
        df = pd.concat([df_pros['Pros'], df_cons['Cons']], axis=1)
        return df
    else:
        col2.write("The selected data contains too less data.")
        return pd.DataFrame() # return empty DataFrame


if unigram_button:
    important_words = create_table()
    col2.table(important_words)
if bigram_button:
    important_words = create_table(unigrams=False)
    col2.table(important_words)

# Automated Responses to Reviews
with open('OPENAI_API_KEY.txt', 'r') as file:
    openai_api_key = file.read().strip()
    
openai.api_key = openai_api_key

def generate_response(review_text):
    if review_text == "":
        return "You have not passed any text. Please provide a review in order to generate an answer."
    else:
        prompt = f"Ein Mitarbeiter hat eine Bewertung über das Unternehmen abgegeben. Der Kommentar des Mitarbeiters lautet wie folgt:\n\n'{review_text}'\n\n Bitte geben Sie eine kurze Antwort als Vertreter des Unternehmens."
        # Use OpenAI's GPT-3 to generate a response
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=250, # Set a maximum of tokens of the response
            n=1, # How often should a response be generated
            temperature=0.2, # Randomness, the higher the weirder
        )
        return response.choices[0].text.strip()

col2.subheader("Automated Response System", help='Your given text input will be feed via an API to OpenAI\'s GPT3 model in order to generate an automatic response')
text_input = col2.text_area(label='',
                            max_chars = 250,
                            placeholder='Insert a review ...',
                            label_visibility='collapsed'
                            )

if col2.button("Generate"):
    col2.write(f"Submitted Review: {text_input}")
    message = col2.chat_message("assistant")
    message.write(generate_response(text_input))
    #message.write(f"Das war der Input: {text_input}") # Use this for testing purposes instead
