import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Set page config
st.set_page_config(
    page_title="Book & Movie Recommender",
    page_icon="📚",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 20px;
    }
    
    /* Header animations */
    .title-animation {
        animation: fadeInDown 1.5s ease-in-out;
    }
    
    /* Card styling */
    .stSelectbox, .stRadio {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .stSelectbox:hover, .stRadio:hover {
        transform: translateY(-5px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #2b5876 0%, #4e4376 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Recommendation card styling */
    .recommendation-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        animation: fadeInUp 0.5s ease-in-out;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0px);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0px);
        }
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px;
        color: #000;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Emoji animations */
    .emoji-float {
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
</style>
""", unsafe_allow_html=True)

def get_recommendations(title, data, n=5):
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Combine genre and description for better recommendations
    data['combined_features'] = data['genre'] + ' ' + data['description']
    
    # Create TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get index of the title
    idx = data[data['title'] == title].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N most similar items (excluding the item itself)
    sim_scores = sim_scores[1:n+1]
    
    # Get indices of recommended items
    item_indices = [i[0] for i in sim_scores]
    
    return data['title'].iloc[item_indices], [round(score[1] * 100, 1) for score in sim_scores]

def analyze_emotions(text):
    """Analyze the emotional content of text using a more sophisticated approach"""
    try:
        # Create TextBlob object with error handling
        blob = TextBlob(str(text))
        
        # Basic sentiment analysis
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Enhanced emotion detection using lemmatization
        words = word_tokenize(str(text).lower())
        emotion_scores = {
            'joy': 0,
            'sadness': 0,
            'anger': 0,
            'fear': 0,
            'surprise': 0,
            'love': 0
        }
        emotion_words = {
            'joy': ['happy', 'joy', 'delight', 'excite', 'wonderful', 'fantastic', 'pleasure', 'laugh', 'smile', 'cheer', 'glad', 'enjoy', 'celebrate', 'thrilled', 'bliss', 'great', 'fun', 'good', 'positive', 'awesome'],
            'sadness': ['sad', 'tragic', 'depress', 'heartbreak', 'grief', 'sorrow', 'melancholy', 'despair', 'miserable', 'gloom', 'cry', 'tear', 'upset', 'lost', 'alone', 'suffer', 'hurt', 'pain', 'distress', 'regret'],
            'anger': ['angry', 'furious', 'rage', 'hate', 'violent', 'fight', 'outrage', 'irritate', 'hostile', 'bitter', 'mad', 'frustrate', 'annoy', 'aggressive', 'hate', 'resent', 'disgust', 'vengeful', 'fury', 'wrath'],
            'fear': ['fear', 'scary', 'terrify', 'horror', 'dangerous', 'threat', 'dread', 'panic', 'nightmare', 'frighten', 'afraid', 'anxious', 'worried', 'nervous', 'scare', 'terror', 'phobia', 'alarm', 'shock', 'concern'],
            'surprise': ['surprise', 'unexpected', 'amaze', 'astonish', 'shock', 'startle', 'stun', 'incredible', 'wonder', 'awe', 'wow', 'sudden', 'unbelievable', 'strange', 'extraordinary', 'remarkable', 'stunning', 'spectacular', 'mindblowing', 'impressive'],
            'love': ['love', 'romantic', 'passion', 'desire', 'affection', 'heart', 'tender', 'adore', 'cherish', 'warmth', 'care', 'fond', 'devoted', 'admire', 'sweet', 'kind', 'gentle', 'compassion', 'embrace', 'attachment']
        }
        for word in words:
            word_lemma = lemmatizer.lemmatize(word)
            word_lemma_verb = lemmatizer.lemmatize(word, pos='v')
            
            for emotion, keywords in emotion_words.items():
                if (word in keywords or 
                    word_lemma in keywords or 
                    word_lemma_verb in keywords or 
                    any(keyword in word for keyword in keywords) or
                    any(keyword in word_lemma for keyword in keywords)):
                    emotion_scores[emotion] += 1
                    
                word_variations = [
                    word + 'ing', word + 'ed', word + 'es', word + 's',
                    word_lemma + 'ing', word_lemma + 'ed', word_lemma + 'es', word_lemma + 's'
                ]
                if any(var in keywords for var in word_variations):
                    emotion_scores[emotion] += 1
        
        total_emotions = sum(emotion_scores.values()) or 1
        emotion_scores = {k: (v / total_emotions) * 100 for k, v in emotion_scores.items()}
        
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        emotion_colors = {
            'joy': '#FFD700',
            'sadness': '#4682B4',
            'anger': '#FF4500',
            'fear': '#800080',
            'surprise': '#32CD32',
            'love': '#FF69B4'
        }
        
        return {
            'primary_emotion': primary_emotion,
            'emotion_scores': emotion_scores,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'color': emotion_colors.get(primary_emotion, '#808080')
        }
    except Exception as e:
        st.error(f"Error in emotion analysis: {str(e)}")
        return {
            'primary_emotion': 'neutral',
            'emotion_scores': {k: 0 for k in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']},
            'polarity': 0,
            'subjectivity': 0,
            'color': '#808080'
        }

def create_similarity_chart(titles, similarity_scores):
    """Create a bar chart to show similarity scores between items"""
    import plotly.graph_objects as go
    
    # Create color gradient based on similarity scores
    colors = [f'rgba(43, 88, 118, {score/100})' for score in similarity_scores]
    
    fig = go.Figure()
    
    # Add bars with enhanced styling
    fig.add_trace(go.Bar(
        x=list(titles),
        y=list(similarity_scores),
        marker=dict(
            color=colors,
            line=dict(color='rgba(43, 88, 118, 1)', width=1)
        ),
        text=[f'{score:.1f}%' for score in similarity_scores],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>' +
                     'Similarity: %{y:.1f}%<br>' +
                     '<extra></extra>'
    ))
    
    # Enhanced layout
    fig.update_layout(
        title=dict(
            text='Similarity Scores',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='',
            tickangle=45,
            showgrid=False
        ),
        yaxis=dict(
            title='Similarity Score (%)',
            gridcolor='rgba(43, 88, 118, 0.1)',
            range=[0, 100]
        ),
        plot_bgcolor='rgba(255, 255, 255, 0.95)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        height=400,
        margin=dict(l=50, r=20, t=80, b=120),
        bargap=0.2,
        showlegend=False
    )
    
    return fig

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Sample data (you can replace this with your own dataset)
def load_sample_data():
    # Define base Tamil movies list
    tamil_movies = [
        {'title': 'Vikram', 'genre': 'Action,Thriller', 'description': 'A special agent investigates a murder spree linked to a drug syndicate...'},
        {'title': 'Master', 'genre': 'Action,Drama', 'description': 'An alcoholic professor is sent to a juvenile school where he confronts his past...'},
        {'title': 'Ponniyin Selvan', 'genre': 'Historical,Drama', 'description': 'A historical epic based on Kalki\'s novel about the Chola dynasty...'},
        {'title': 'Vaaranam Aayiram', 'genre': 'Romance,Drama', 'description': 'A man reminisces about his life and love through different stages...'},
        {'title': 'Kaithi', 'genre': 'Action,Thriller', 'description': 'A prison transport driver must battle criminals to save police officers...'},
        # Adding more Tamil movies...
        {'title': 'Theeran Adhigaaram Ondru', 'genre': 'Action,Crime', 'description': 'A police officer tracks down a gang of ruthless dacoits terrorizing highways...'},
        {'title': 'Asuran', 'genre': 'Drama,Action', 'description': 'A farmer fights against social injustice while protecting his family...'},
        {'title': 'Super Deluxe', 'genre': 'Drama,Thriller', 'description': 'Multiple storylines intersect in this surreal drama about life and destiny...'},
        {'title': 'Karnan', 'genre': 'Drama,Action', 'description': 'A young man fights for the rights of his oppressed community...'},
        {'title': '96', 'genre': 'Romance,Drama', 'description': 'Two high school sweethearts meet after decades and reminisce about their past...'},
        {'title': 'Pariyerum Perumal', 'genre': 'Drama,Social', 'description': 'A law student faces caste discrimination while pursuing his education...'},
        {'title': 'Soorarai Pottru', 'genre': 'Drama,Biography', 'description': 'A man dreams of launching a low-cost airline for the common people...'},
        {'title': 'Ratsasan', 'genre': 'Thriller,Crime', 'description': 'A police officer hunts down a psychopathic serial killer targeting young girls...'},
        {'title': 'Vada Chennai', 'genre': 'Crime,Drama', 'description': 'A skilled carrom player gets pulled into the world of gang warfare...'},
        {'title': 'Maanagaram', 'genre': 'Thriller,Drama', 'description': 'Multiple lives intersect in the urban jungle of Chennai...'},
        {'title': 'Visaranai', 'genre': 'Crime,Drama', 'description': 'Working-class men become victims of police brutality and corruption...'},
        {'title': 'Jigarthanda', 'genre': 'Crime,Comedy', 'description': 'An aspiring filmmaker researches a gangster for his next movie...'},
        {'title': 'Jai Bhim', 'genre': 'Drama,Legal', 'description': 'A lawyer fights for justice for a tribal woman whose husband disappeared in custody...'},
        {'title': 'Kaala', 'genre': 'Action,Drama', 'description': 'A powerful don protects his people in Mumbai\'s largest slum...'},
        {'title': 'Aandavan Kattalai', 'genre': 'Comedy,Drama', 'description': 'A man\'s attempts to go abroad lead to complicated situations...'},
        {'title': 'Joker', 'genre': 'Drama,Satire', 'description': 'A farmer takes on the system through satirical protests...'},
        {'title': 'Kadaisi Vivasayi', 'genre': 'Drama', 'description': 'The last farmer in a village fights to preserve traditional farming...'},
        {'title': 'Managaram', 'genre': 'Thriller,Drama', 'description': 'A small-town guy faces various challenges in the big city...'},
        {'title': 'Kaakkaa Muttai', 'genre': 'Drama', 'description': 'Two slum children dream of tasting pizza from a new restaurant...'},
        {'title': 'Aruvi', 'genre': 'Drama,Social', 'description': 'A young woman rebels against societal norms and injustice...'},
        {'title': 'Vikram Vedha', 'genre': 'Action,Crime', 'description': 'A police officer pursues a mysterious gangster who challenges his beliefs...'},
        {'title': 'Merku Thodarchi Malai', 'genre': 'Drama', 'description': 'The life of workers in the Western Ghats region...'},
        {'title': 'Dhuruvangal Pathinaaru', 'genre': 'Thriller,Crime', 'description': 'A retired police officer recounts a complex case from his past...'},
        {'title': 'Mandela', 'genre': 'Comedy,Drama', 'description': 'A barber becomes a deciding vote in local elections...'},
        {'title': 'Nayakan', 'genre': 'Crime,Drama', 'description': 'A boy becomes a powerful don while fighting for his community...'},
        {'title': 'Bombay', 'genre': 'Romance,Drama', 'description': 'A Hindu-Muslim love story set against communal riots...'},
        {'title': 'Hey Ram', 'genre': 'Historical,Drama', 'description': 'A man\'s journey through India\'s partition and independence...'},
        {'title': 'Thalapathi', 'genre': 'Action,Drama', 'description': 'A man rises in the criminal world while seeking his identity...'},
        {'title': 'Anbe Sivam', 'genre': 'Drama,Comedy', 'description': 'Two contrasting personalities learn about humanity during a journey...'},
        {'title': 'Irudhi Suttru', 'genre': 'Sports,Drama', 'description': 'A boxing coach finds a talented female boxer from fishing community...'},
        {'title': 'Thani Oruvan', 'genre': 'Action,Crime', 'description': 'A police officer targets a sophisticated criminal mastermind...'},
        {'title': 'Kadhal', 'genre': 'Romance,Drama', 'description': 'A love story facing the challenges of caste and class barriers...'},
        {'title': 'Vinnaithaandi Varuvaayaa', 'genre': 'Romance,Drama', 'description': 'A Hindu boy falls in love with a Christian girl aspiring to be an actress...'},
        {'title': 'OK Kanmani', 'genre': 'Romance,Drama', 'description': 'A modern love story about live-in relationships...'},
        {'title': 'Enthiran', 'genre': 'Sci-Fi,Action', 'description': 'A scientist creates a robot that develops human emotions...'},
        {'title': 'Kaththi', 'genre': 'Action,Social', 'description': 'A thief fights against corporate exploitation of farmers...'},
        {'title': 'Mozhi', 'genre': 'Romance,Drama', 'description': 'A musician falls in love with a deaf and mute woman...'},
        {'title': 'Pudhupettai', 'genre': 'Crime,Drama', 'description': 'A young man\'s rise in the violent world of politics and crime...'},
        {'title': 'Alaipayuthey', 'genre': 'Romance,Drama', 'description': 'A couple faces challenges after their marriage...'},
        {'title': 'Kannathil Muthamittal', 'genre': 'Drama,War', 'description': 'An adopted girl searches for her biological mother in war-torn Sri Lanka...'},
        {'title': 'Pithamagan', 'genre': 'Drama', 'description': 'The friendship between a graveyard caretaker and a conman...'},
        {'title': 'Raavanan', 'genre': 'Drama,Action', 'description': 'A modern interpretation of the Ramayana...'},
        {'title': 'Autograph', 'genre': 'Drama,Romance', 'description': 'A man reminisces about his past relationships...'},
        {'title': 'Peranbu', 'genre': 'Drama', 'description': 'A father learns to care for his differently-abled daughter...'},
        {'title': 'Virumaandi', 'genre': 'Action,Drama', 'description': 'A man wrongly imprisoned narrates his life story...'},
        {'title': 'Sethu', 'genre': 'Romance,Drama', 'description': 'A college rowdy\'s life changes after falling in love...'},
        {'title': 'Subramaniapuram', 'genre': 'Crime,Drama', 'description': 'A period film about friendship and betrayal in the 1980s...'},
        {'title': 'Aayirathil Oruvan', 'genre': 'Adventure,Mystery', 'description': 'An expedition to find a lost Chola civilization...'},
        {'title': 'Vettaiyaadu Vilaiyaadu', 'genre': 'Crime,Thriller', 'description': 'A police officer pursues a serial killer across continents...'},
        {'title': 'Chennai 600028', 'genre': 'Comedy,Sports', 'description': 'Street cricket teams compete in local tournaments...'},
        {'title': 'Polladhavan', 'genre': 'Action,Drama', 'description': 'A young man gets involved with criminals while searching for his stolen bike...'},
        {'title': 'Anjathey', 'genre': 'Crime,Thriller', 'description': 'Police trainees get involved in a dangerous investigation...'},
        {'title': 'Mayakkam Enna', 'genre': 'Drama,Romance', 'description': 'An aspiring photographer struggles for recognition...'},
        {'title': 'Kaadhal', 'genre': 'Romance,Drama', 'description': 'A pure love story facing societal pressures...'},
        {'title': 'Theeran', 'genre': 'Action,Crime', 'description': 'Based on true events about highway robbery cases...'},
        {'title': 'Goli Soda', 'genre': 'Action,Drama', 'description': 'Four young boys fight against a powerful businessman...'},
        {'title': 'Madras', 'genre': 'Action,Drama', 'description': 'Local politics and wall graffiti rights lead to violence...'},
        {'title': 'Onaayum Aattukkuttiyum', 'genre': 'Thriller,Drama', 'description': 'A mysterious man helps an injured policeman...'},
        {'title': 'Paradesi', 'genre': 'Drama', 'description': 'Tea plantation workers struggle under British rule...'},
        {'title': 'Kuttrame Thandanai', 'genre': 'Crime,Drama', 'description': 'A man with tunnel vision gets involved in a murder...'},
        {'title': 'Thegidi', 'genre': 'Thriller,Mystery', 'description': 'A private detective uncovers a dangerous conspiracy...'},
        {'title': 'Angadi Theru', 'genre': 'Drama', 'description': 'The lives of workers in a textile showroom...'},
        {'title': 'Naduvula Konjam Pakkatha Kaanom', 'genre': 'Comedy,Drama', 'description': 'A man loses his recent memory before his wedding...'},
        {'title': 'Aaranya Kaandam', 'genre': 'Crime,Thriller', 'description': 'Multiple characters intersect in the Chennai underworld...'},
        {'title': 'Mouna Guru', 'genre': 'Thriller,Crime', 'description': 'An introvert student gets framed for a crime...'},
        {'title': 'Soodhu Kavvum', 'genre': 'Comedy,Crime', 'description': 'A kidnapper who follows a moral code gets into trouble...'},
        {'title': 'Pizza', 'genre': 'Horror,Thriller', 'description': 'A pizza delivery boy experiences supernatural events...'},
        {'title': 'Attakathi', 'genre': 'Romance,Comedy', 'description': 'A young man keeps falling in love despite rejections...'},
        {'title': 'Marina', 'genre': 'Drama', 'description': 'The lives of various people working at Marina Beach...'},
        {'title': 'Kumki', 'genre': 'Drama,Romance', 'description': 'A mahout and his elephant protect a village...'},
        {'title': 'Mynaa', 'genre': 'Romance,Drama', 'description': 'A pure love story set in a rural backdrop...'},
        {'title': 'Engaeyum Eppothum', 'genre': 'Romance,Drama', 'description': 'Multiple love stories converge during a bus journey...'},
        {'title': 'Veyil', 'genre': 'Drama', 'description': 'A man reflects on his relationship with his brother...'},
        {'title': 'Thavamai Thavamirundhu', 'genre': 'Family,Drama', 'description': 'A father sacrifices everything for his sons...'},
        {'title': 'Raam', 'genre': 'Thriller,Drama', 'description': 'A teenager investigates his mother\'s murder...'}
    ]    # Define international movies
    international_movies = [
        {'title': 'The Shawshank Redemption', 'genre': 'Drama', 'description': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency...'},
        {'title': 'The Godfather', 'genre': 'Crime,Drama', 'description': 'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son...'},
        {'title': 'Inception', 'genre': 'Action,Sci-Fi', 'description': 'A thief who enters the dreams of others to steal secrets finds himself on a perilous mission to plant an idea into someone\'s mind...'},
        {'title': 'The Dark Knight', 'genre': 'Action,Crime,Drama', 'description': 'When the menace known as the Joker wreaks havoc on Gotham City, Batman must accept one of the greatest psychological tests of his ability to fight injustice...'},
        {'title': 'Pulp Fiction', 'genre': 'Crime,Drama', 'description': 'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption...'},
        {'title': 'Schindler\'s List', 'genre': 'Biography,Drama,History', 'description': 'A German businessman saves the lives of more than a thousand Jewish refugees during the Holocaust...'},
        {'title': 'Forrest Gump', 'genre': 'Drama,Romance', 'description': 'The presidencies, wars, and cultural events of the 20th century are seen through the eyes of a simple man from Alabama...'},
        {'title': 'The Matrix', 'genre': 'Action,Sci-Fi', 'description': 'A computer programmer discovers that reality as he knows it is a simulation created by machines, and joins a rebellion to break free...'},
        {'title': 'Fight Club', 'genre': 'Drama,Thriller', 'description': 'An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much more...'},
        {'title': 'Goodfellas', 'genre': 'Biography,Crime,Drama', 'description': 'The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen and his mob partners Jimmy Conway and Tommy DeVito...'},
        {'title': 'The Silence of the Lambs', 'genre': 'Crime,Drama,Thriller', 'description': 'A young FBI cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer...'},
        {'title': 'The Lord of the Rings: The Fellowship of the Ring', 'genre': 'Action,Adventure,Drama', 'description': 'A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth...'},
        {'title': 'Saving Private Ryan', 'genre': 'Drama,War', 'description': 'Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper whose brothers have been killed in action...'},
        {'title': 'Jurassic Park', 'genre': 'Action,Adventure,Sci-Fi', 'description': 'A pragmatic paleontologist visiting an almost complete theme park is tasked with protecting a couple of kids after a power failure causes the park\'s cloned dinosaurs to run loose...'},
        {'title': 'The Green Mile', 'genre': 'Crime,Drama,Fantasy', 'description': 'The lives of guards on Death Row are affected by one of their charges: a black man accused of child murder and rape, yet who has a mysterious gift...'},
        {'title': 'Gladiator', 'genre': 'Action,Adventure,Drama', 'description': 'A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery...'},
        {'title': 'The Departed', 'genre': 'Crime,Drama,Thriller', 'description': 'An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in South Boston...'},
        {'title': 'The Sixth Sense', 'genre': 'Drama,Mystery,Thriller', 'description': 'A boy who communicates with spirits seeks the help of a disheartened child psychologist...'},
        {'title': 'The Social Network', 'genre': 'Biography,Drama', 'description': 'As Harvard student Mark Zuckerberg creates the social networking site that would become known as Facebook, he is sued by the twins who claimed he stole their idea...'},
        {'title': 'Avatar', 'genre': 'Action,Adventure,Fantasy', 'description': 'A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home...'},
        {'title': 'Interstellar', 'genre': 'Adventure,Drama,Sci-Fi', 'description': 'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival...'},
        {'title': 'The Prestige', 'genre': 'Drama,Mystery,Thriller', 'description': 'After a tragic accident, two stage magicians engage in a battle to create the ultimate illusion while sacrificing everything they have to outwit each other...'},
        {'title': 'The Pursuit of Happyness', 'genre': 'Biography,Drama', 'description': 'A struggling salesman takes custody of his son as he\'s poised to begin a life-changing professional career...'},
        {'title': 'No Country for Old Men', 'genre': 'Crime,Drama,Thriller', 'description': 'Violence and mayhem ensue after a hunter stumbles upon a drug deal gone wrong and more than two million dollars in cash near the Rio Grande...'},
        {'title': 'The Pianist', 'genre': 'Biography,Drama,Music', 'description': 'A Polish Jewish musician struggles to survive the destruction of the Warsaw ghetto of World War II...'},
        {'title': 'The Usual Suspects', 'genre': 'Crime,Mystery,Thriller', 'description': 'A sole survivor tells of the twisty events leading up to a horrific gun battle on a boat...'},
        {'title': 'The Wolf of Wall Street', 'genre': 'Biography,Comedy,Crime', 'description': 'Based on the true story of Jordan Belfort, from his rise to a wealthy stock-broker living the high life to his fall involving crime, corruption and the federal government...'},
        {'title': 'Eternal Sunshine of the Spotless Mind', 'genre': 'Drama,Romance,Sci-Fi', 'description': 'When their relationship turns sour, a couple undergoes a medical procedure to have each other erased from their memories...'},
        {'title': 'The Grand Budapest Hotel', 'genre': 'Adventure,Comedy,Crime', 'description': 'A writer encounters the owner of an aging high-class hotel, who tells him of his early years serving as a lobby boy in the hotel\'s glorious years under an exceptional concierge...'},
        {'title': 'The Revenant', 'genre': 'Action,Adventure,Drama', 'description': 'A frontiersman on a fur trading expedition in the 1820s fights for survival after being mauled by a bear and left for dead by members of his own hunting team...'},
        {'title': 'Whiplash', 'genre': 'Drama,Music', 'description': 'A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student\'s potential...'},
        {'title': 'The Theory of Everything', 'genre': 'Biography,Drama,Romance', 'description': 'A look at the relationship between the famous physicist Stephen Hawking and his wife...'},
        {'title': 'La La Land', 'genre': 'Comedy,Drama,Music', 'description': 'While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future...'},
        {'title': 'The Shape of Water', 'genre': 'Adventure,Drama,Fantasy', 'description': 'At a top secret research facility in the 1960s, a lonely janitor forms a unique relationship with an amphibious creature that is being held in captivity...'},
        {'title': 'Black Swan', 'genre': 'Drama,Thriller', 'description': 'A committed dancer struggles to maintain her sanity after winning the lead role in a production of Tchaikovsky\'s "Swan Lake"...'},
        {'title': 'Gone Girl', 'genre': 'Drama,Mystery,Thriller', 'description': 'With his wife\'s disappearance having become the focus of an intense media circus, a man sees the spotlight turned on him when it\'s suspected that he may not be innocent...'},
        {'title': 'The Martian', 'genre': 'Adventure,Drama,Sci-Fi', 'description': 'An astronaut becomes stranded on Mars after his team assumes him dead, and must rely on his ingenuity to find a way to signal to Earth that he is alive...'},
        {'title': 'A Beautiful Mind', 'genre': 'Biography,Drama', 'description': 'After John Nash, a brilliant but asocial mathematician, accepts secret work in cryptography, his life takes a turn for the nightmarish...'},
        {'title': 'The Butterfly Effect', 'genre': 'Drama,Sci-Fi,Thriller', 'description': 'Evan Treborn suffers blackouts during significant events of his life. As he grows up, he finds a way to remember these lost memories and a supernatural way to alter his life by reading his journal...'},
        {'title': 'The Intouchables', 'genre': 'Biography,Comedy,Drama', 'description': 'After he becomes a quadriplegic from a paragliding accident, an aristocrat hires a young man from the projects to be his caregiver...'},
        {'title': 'The Curious Case of Benjamin Button', 'genre': 'Drama,Fantasy,Romance', 'description': 'Tells the story of Benjamin Button, a man who starts aging backwards with consequences...'},
        {'title': 'The Machinist', 'genre': 'Drama,Thriller', 'description': 'An industrial worker who hasn\'t slept in a year begins to doubt his own sanity...'},
        {'title': 'The Big Short', 'genre': 'Biography,Comedy,Drama', 'description': 'In 2006-2007 a group of investors bet against the US mortgage market. In their research they discover how flawed and corrupt the market is...'},
        {'title': 'Ex Machina', 'genre': 'Drama,Sci-Fi,Thriller', 'description': 'A young programmer is selected to participate in a ground-breaking experiment in synthetic intelligence by evaluating the human qualities of a highly advanced humanoid A.I...'},
        {'title': 'Inside Out', 'genre': 'Animation,Adventure,Comedy', 'description': 'After young Riley is uprooted from her Midwest life and moved to San Francisco, her emotions - Joy, Fear, Anger, Disgust and Sadness - conflict on how best to navigate a new city, house, and school...'},
        {'title': 'The Help', 'genre': 'Drama', 'description': 'An aspiring author during the civil rights movement of the 1960s decides to write a book detailing the African American maids\' point of view...'},
        {'title': 'The King\'s Speech', 'genre': 'Biography,Drama,History', 'description': 'The story of King George VI, his impromptu ascension to the throne of the British Empire in 1936, and the speech therapist who helped the unsure monarch overcome his stammer...'},
        {'title': 'Life of Pi', 'genre': 'Adventure,Drama,Fantasy', 'description': 'A young man who survives a disaster at sea is hurtled into an epic journey of adventure and discovery. While cast away, he forms an unexpected connection with another survivor: a fearsome Bengal tiger...'},
        {'title': 'The Perks of Being a Wallflower', 'genre': 'Drama,Romance', 'description': 'An introvert freshman is taken under the wings of two seniors who welcome him to the real world...'},
        {'title': 'Room', 'genre': 'Drama,Thriller', 'description': 'Held captive for 7 years in an enclosed space, a woman and her young son finally gain their freedom, allowing the boy to experience the outside world for the first time...'},
        {'title': 'The Imitation Game', 'genre': 'Biography,Drama,Thriller', 'description': 'During World War II, the English mathematical genius Alan Turing tries to crack the German Enigma code...'},
        {'title': 'The Truman Show', 'genre': 'Comedy,Drama', 'description': 'An insurance salesman discovers his whole life is actually a reality TV show...'},
        {'title': 'The Hurt Locker', 'genre': 'Drama,Thriller,War', 'description': 'During the Iraq War, a Sergeant recently assigned to an army bomb squad is put at odds with his squad mates due to his maverick way of handling his work...'}
    ]

    # Combine all movies
    all_movies = tamil_movies + international_movies

    # Create the movies DataFrame
    movies_df = pd.DataFrame(all_movies)
    movies_df['movie_type'] = ['Tamil' if movie in tamil_movies else 'International' for movie in all_movies]
      # Books data
    books_data = {
        'title': [
            '1984', 'To Kill a Mockingbird', 'The Great Gatsby', 'Pride and Prejudice', 'The Catcher in the Rye',
            'The Lord of the Rings', 'One Hundred Years of Solitude', 'Brave New World', 'The Hobbit', 'Crime and Punishment',
            'The Alchemist', 'Don Quixote', 'The Chronicles of Narnia', 'The Da Vinci Code', 'The Hunger Games',
            'The Girl with the Dragon Tattoo', 'The Book Thief', 'The Kite Runner', 'Life of Pi', 'The Road',
            'The Name of the Wind', 'American Gods', 'The Night Circus', 'The Shadow of the Wind', 'The Handmaid\'s Tale',
            'Dune', 'Fahrenheit 451', 'The Picture of Dorian Gray', 'The Count of Monte Cristo', 'Jane Eyre',
            'Wuthering Heights', 'The Brothers Karamazov', 'Les Misérables', 'The Little Prince', 'The Bell Jar',
            'A Tale of Two Cities', 'Anna Karenina', 'The Grapes of Wrath', 'The Old Man and the Sea', 'Lord of the Flies',
            'Slaughterhouse-Five', 'The Secret History', 'The Pillars of the Earth', 'The Giver', 'The Stand',
            'Neuromancer', 'The Foundation Trilogy', 'The Time Machine', 'The Wind in the Willows', 'A Game of Thrones',
            'The Hitchhiker\'s Guide to the Galaxy', 'The Color Purple', 'The Outsiders', 'Water for Elephants', 'Gone with the Wind'
        ],
        'genre': [
            'Dystopian,Political', 'Fiction,Classic', 'Fiction,Classic', 'Romance,Classic', 'Fiction,Classic',
            'Fantasy,Adventure', 'Magical Realism,Literary Fiction', 'Dystopian,Sci-Fi', 'Fantasy,Adventure', 'Psychological Fiction,Crime',
            'Fantasy,Philosophy', 'Classic,Adventure', 'Fantasy,Children', 'Mystery,Thriller', 'Dystopian,Young Adult',
            'Crime,Mystery', 'Historical Fiction,War', 'Fiction,Drama', 'Adventure,Fantasy', 'Post-Apocalyptic,Fiction',
            'Fantasy,Adventure', 'Fantasy,Mythology', 'Fantasy,Romance', 'Mystery,Historical', 'Dystopian,Feminist',
            'Sci-Fi,Fantasy', 'Dystopian,Sci-Fi', 'Gothic,Philosophy', 'Adventure,Romance', 'Gothic,Romance',
            'Gothic,Romance', 'Philosophy,Drama', 'Historical Fiction,Drama', 'Children,Fantasy', 'Semi-Autobiographical,Drama',
            'Historical Fiction,Drama', 'Literary Fiction,Romance', 'Historical Fiction,Drama', 'Literary Fiction,Adventure', 'Allegory,Fiction',
            'Sci-Fi,Satire', 'Mystery,Literary Fiction', 'Historical Fiction,Drama', 'Dystopian,Young Adult', 'Post-Apocalyptic,Horror',
            'Cyberpunk,Sci-Fi', 'Sci-Fi,Epic', 'Sci-Fi,Classic', 'Children,Fantasy', 'Fantasy,Epic',
            'Sci-Fi,Comedy', 'Literary Fiction,Drama', 'Young Adult,Drama', 'Historical Fiction,Romance', 'Historical Fiction,Romance'
        ],
        'description': [
            'A dystopian social science fiction novel where a totalitarian regime controls everything through surveillance and manipulation of language and history...',
            'The story of racial injustice and the loss of innocence in the American South through the eyes of young Scout Finch...',
            'The story of the mysteriously wealthy Jay Gatsby and his obsessive love for the beautiful Daisy Buchanan, set against the backdrop of the Jazz Age...',
            'Elizabeth Bennet\'s journey through matters of upbringing, morality, education, and marriage in Georgian England...',
            'Holden Caulfield\'s experiences in New York City after leaving his prep school, dealing with alienation and the loss of innocence...',
            'An epic high-fantasy novel about a hobbit\'s quest to destroy a powerful ring and defeat the dark lord who created it...',
            'Multi-generational saga of the Buendía family in the mythical town of Macondo, blending reality with magical elements...',
            'A dystopian novel envisioning a genetically engineered future where humanity is pacified through pleasure and conditioning...',
            'The adventure of Bilbo Baggins, who journeys with a group of dwarves to reclaim their mountain home from a dragon...',
            'A psychological novel about a poor ex-student who commits murder and then deals with the consequences and his conscience...',
            'A shepherd\'s journey to find his Personal Legend and understand the Soul of the World...',
            'The adventures of a man who loses his sanity reading chivalric romances and decides to revive chivalry...',
            'Children discover a magical world through a wardrobe and help defeat an evil witch who has trapped the land in eternal winter...',
            'A murder mystery involving cryptic codes, secret societies, and religious conspiracies...',
            'In a dystopian future, young people are forced to participate in a televised battle to the death...',
            'A complex murder mystery involving a brilliant hacker and a journalist investigating a wealthy family\'s dark secrets...',
            'A young girl in Nazi Germany finds solace in books while her family hides a Jewish man in their basement...',
            'A story of friendship, betrayal, and redemption set against the backdrop of Afghanistan\'s recent history...',
            'A young man survives a shipwreck and spends 227 days on a lifeboat with a Bengal tiger...',
            'A father and son journey through a post-apocalyptic America, trying to survive and maintain their humanity...',
            'A young man seeks to become the greatest magician in the world while unraveling the mystery of his parents\' death...',
            'An ex-convict becomes entangled in a war between ancient and modern gods in America...',
            'A magical competition between two young illusionists who don\'t realize they\'re falling in love...',
            'A boy discovers a mysterious book in a forgotten library, pulling him into a haunting mystery...',
            'In a dystopian future, women are stripped of their rights and forced into reproductive servitude...',
            'A science fiction epic about politics, religion, and ecology on a desert planet...',
            'A fireman whose job is to burn books begins to question his society\'s censorship...',
            'A man remains eternally young while his portrait ages, revealing the price of immortality and hedonism...',
            'A man wrongly imprisoned seeks revenge against those who betrayed him...',
            'An orphan girl\'s journey from a harsh childhood to finding love and independence...',
            'A passionate but tragic love story set on the Yorkshire moors...',
            'A philosophical novel exploring faith, doubt, and morality through three brothers...',
            'An epic tale of justice, love, and redemption in 19th-century France...',
            'A poetic tale about a young prince visiting Earth from his tiny asteroid...',
            'A young woman\'s descent into mental illness in 1950s America...',
            'A story of sacrifice and resurrection during the French Revolution...',
            'A complex exploration of love, marriage, and Russian society...',
            'A family\'s journey during the Great Depression, seeking dignity and a better life...',
            'An old fisherman\'s epic struggle with a giant marlin in the Gulf Stream...',
            'A group of British boys attempt to govern themselves after being stranded on an uninhabited island...',
            'A non-linear narrative about a soldier\'s experiences during WWII and his time-travel adventures...',
            'A group of elite college students become entangled in murder and ancient Greek rituals...',
            'An epic tale of power struggles, romance, and cathedral building in medieval England...',
            'A young boy discovers the dark truth behind his seemingly perfect society...',
            'A post-apocalyptic epic of good versus evil after a pandemic wipes out most of humanity...',
            'A groundbreaking cyberpunk novel about hackers, AI, and corporate power...',
            'A complex saga about psychohistory and the fall and rise of a galactic empire...',
            'A scientist invents a machine that allows him to travel through time...',
            'The adventures of various animal friends along a river in the English countryside...',
            'The first book in an epic fantasy series about political intrigue and dynastic struggles...',
            'A hilarious science fiction comedy about the meaning of life and the end of the world...',
            'A powerful story of abuse, resilience, and sisterhood in the American South...',
            'A coming-of-age story about youth gangs and class conflict in 1960s Oklahoma...',
            'A veterinary student joins a traveling circus during the Great Depression...',
            'An epic tale of love and loss in the American Civil War South...'
        ]
    }

    # Create the books DataFrame
    books_df = pd.DataFrame(books_data)

    return movies_df, books_df

def main():
    st.markdown('<h1 class="title-animation">📚 <span class="emoji-float">🎬</span> Book & Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="title-animation">Get personalized recommendations based on your favorite books and movies! ✨</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        movies_df, books_df = load_sample_data()
        st.session_state.movies_df = movies_df
        st.session_state.books_df = books_df
        st.session_state.data_loaded = True
    
    tab1, tab2 = st.tabs(["🎬 Movies", "📖 Books"])
    
    with tab1:
        st.markdown('<h2 class="title-animation">🎬 Movie Recommendations</h2>', unsafe_allow_html=True)
        
        st.markdown('<h3 class="title-animation">🎭 Mood-based Filtering</h3>', unsafe_allow_html=True)
        st.info("""
        How mood filtering works:
        1. Start with "Any Mood" to see ALL recommendations first
        2. Look at the emotional profiles shown in the radar charts
        3. Then select a specific mood to filter the recommendations you like
        
        For example:
        1. Keep "Any Mood" selected to see all similar movies
        2. Notice which emotions are strongest in the recommendations
        3. Then choose 'Joy' for uplifting movies, 'Love' for romantic stories, etc.
        4. You can always go back to "Any Mood" to see all recommendations again
        """)
        
        selected_mood = st.selectbox(
            "Select your current mood:",
            options=['Any Mood', 'Joy', 'Love', 'Surprise', 'Fear', 'Sadness', 'Anger'],
            help="Choose a mood to filter recommendations. Select 'Any Mood' to see all recommendations."
        )
        
        movie_type = st.radio(
            "Select Movie Category: 🎯",
            options=['🌟 All Movies', '🎭 Tamil Movies', '🌍 International Movies'],
            horizontal=True
        )
        
        if '🎭 Tamil Movies' in movie_type:
            filtered_movies = st.session_state.movies_df[st.session_state.movies_df['movie_type'] == 'Tamil']
        elif '🌍 International Movies' in movie_type:
            filtered_movies = st.session_state.movies_df[st.session_state.movies_df['movie_type'] == 'International']
        else:
            filtered_movies = st.session_state.movies_df
        
        movie_title = st.selectbox(
            "🎥 Select a movie you like:",
            options=filtered_movies['title'].tolist()
        )

        if st.button("🔍 Get Movie Recommendations"):
            recommended_titles, similarity_scores = get_recommendations(
                movie_title, 
                st.session_state.movies_df
            )
            
            st.markdown('<h3 class="title-animation">🎯 Movie Similarity Analysis</h3>', unsafe_allow_html=True)
            
            similarity_fig = create_similarity_chart(recommended_titles, similarity_scores)
            st.plotly_chart(similarity_fig, use_container_width=True)
            
            st.markdown('<h3 class="title-animation">📊 Detailed Recommendations</h3>', unsafe_allow_html=True)
            
            filtered_recommendations = []
            
            for title, score in zip(recommended_titles, similarity_scores):
                movie_data = st.session_state.movies_df[st.session_state.movies_df['title'] == title].iloc[0]
                emotion_data = analyze_emotions(movie_data['description'])
                
                if selected_mood == 'Any Mood' or emotion_data['primary_emotion'].capitalize() == selected_mood:
                    filtered_recommendations.append((title, score, movie_data, emotion_data))
            
            if not filtered_recommendations:
                st.warning(f"No movies matching the selected mood '{selected_mood}' were found. Try a different mood or 'Any Mood'.")
            else:
                for title, score, movie_data, emotion_data in filtered_recommendations:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>🎬 {title}</h4>
                            <p>✨ Similarity: {score}%</p>
                            <p>🎭 Genre: {movie_data['genre']}</p>
                            <p>💫 Primary Emotion: <span style="color: {emotion_data['color']}">{emotion_data['primary_emotion'].capitalize()}</span></p>
                            <p>📝 Description: {movie_data['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.progress(score/100, f"Similarity: {score}%")
                    
                    st.divider()
    
    with tab2:
        st.markdown('<h2 class="title-animation">📖 Book Recommendations</h2>', unsafe_allow_html=True)
        
        # Add mood filter with explanation
        st.markdown('<h3 class="title-animation">🎭 Mood-based Filtering</h3>', unsafe_allow_html=True)
        st.info("""
        How mood filtering works:
        1. Start with "Any Mood" to see ALL book recommendations first
        2. Look at the similarity scores for each recommendation
        3. Then select a specific mood to filter the recommendations you like
        
        For example:
        1. Keep "Any Mood" selected to see all similar books
        2. Notice the emotional themes in the descriptions
        3. Then choose 'Joy' for uplifting books, 'Love' for romantic stories, etc.
        4. You can always go back to "Any Mood" to see all recommendations again
        """)
        
        selected_book_mood = st.selectbox(
            "Select your current mood:",
            options=['Any Mood', 'Joy', 'Love', 'Surprise', 'Fear', 'Sadness', 'Anger'],
            key="book_mood",
            help="Choose a mood to filter book recommendations. Select 'Any Mood' to see all recommendations."
        )
        
        book_title = st.selectbox(
            "📚 Select a book you like:",
            options=st.session_state.books_df['title'].tolist()
        )
        
        if st.button("🔍 Get Book Recommendations"):
            recommended_titles, similarity_scores = get_recommendations(
                book_title, 
                st.session_state.books_df
            )
            
            st.markdown('<h3 class="title-animation">📚 Book Similarity Analysis</h3>', unsafe_allow_html=True)
            
            # Create and display the similarity chart
            similarity_fig = create_similarity_chart(recommended_titles, similarity_scores)
            st.plotly_chart(similarity_fig, use_container_width=True)
            
            st.markdown('<h3 class="title-animation">📊 Detailed Recommendations</h3>', unsafe_allow_html=True)
            
            # Filter recommendations by mood if specified
            filtered_recommendations = []
            
            for title, score in zip(recommended_titles, similarity_scores):
                book_data = st.session_state.books_df[st.session_state.books_df['title'] == title].iloc[0]
                emotion_data = analyze_emotions(book_data['description'])
                
                if selected_book_mood == 'Any Mood' or emotion_data['primary_emotion'].capitalize() == selected_book_mood:
                    filtered_recommendations.append((title, score, book_data, emotion_data))
            
            if not filtered_recommendations:
                st.warning(f"No books matching the selected mood '{selected_book_mood}' were found. Try a different mood or 'Any Mood'.")
            else:
                for title, score, book_data, emotion_data in filtered_recommendations:
                    # Create two columns for each recommendation
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>📖 {title}</h4>
                            <p>✨ Similarity: {score}%</p>
                            <p>📚 Genre: {book_data['genre']}</p>
                            <p>💫 Primary Emotion: <span style="color: {emotion_data['color']}">{emotion_data['primary_emotion'].capitalize()}</span></p>
                            <p>📝 Description: {book_data['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Display similarity score as a progress bar
                        st.progress(score/100, f"Similarity: {score}%")
                    
                    st.divider()

if __name__ == "__main__":
    main()