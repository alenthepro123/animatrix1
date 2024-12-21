import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import streamlit as st
import pandas as pd
import base64
import os

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="ANIMATRIX",
    page_icon="C:/Users/Senku Ishigami/Downloads/Red_Geass.png",
    layout="centered"
)

# Import custom font for title
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    </style>
    """,
    unsafe_allow_html=True
)

# --- Streamlit Centered Logo and Title ---
st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: center; text-align: center; margin-top: 20px;">
    <img src="data:image/png;base64,{base64.b64encode(open('C:/Users/Senku Ishigami/Downloads/Red_Geass.png', 'rb').read()).decode()}" 
         alt="ANIMATRIX Logo" 
         style="width: 200px; height: 100px; margin-right: 20px;">
    <h1 style="
        font-family: 'Orbitron', sans-serif; 
        font-size: 50px; 
        color: #ffffff; 
        background: linear-gradient(90deg,#1E90FF, #6A5ACD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 6px 6px rgba(0, 0, 0, 0.9);
        margin: 0;
    ">
        ANIMATRIX
    </h1>
</div>


    """,
    unsafe_allow_html=True
)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 100  # Should match the number of classes
ANIME_METADATA_PATH = "C:/Users/Senku Ishigami/Documents/Assignment-elective-main/anime-dataset-2023.csv"  # Path to your dataset
ANIME_TITLES = [
    'Akame ga Kill!', 'Angel Beats!', 'Ano Hi Mita Hana no Namae wo Bokutachi wa Mada Shiranai',
    'Another', 'Ansatsu Kyoushitsu', 'Ao no Exorcist', 'Bakemonogatari', 'BANANA FISH', 'Black Clover',
    'Bleach', 'Boku dake ga Inai Machi', 'Boku no Hero Academia', 'Bungou Stray Dogs', 'Chainsaw Man',
    'Charlotte', 'Chuunibyou demo Koi ga Shitai', 'CLANNAD', 'Code Geass_ Hangyaku no Lelouch',
    'Cowboy Bebop', 'Darling in the Franxx', 'Death Note', 'Death Parade', 'Devilman Crybaby',
    'Dororo', 'Dr. Stone', 'Dungeon ni Deai wo Motomeru no wa Machigatteiru Darou ka', 'Durarara!!',
    'Enen no Shouboutai', 'Fairy Tail', 'Fate Zero', 'Fruits Basket_ 1st Season', 'Fumetsu no Anata e',
    'Go-toubun no Hanayome', 'Goblin Slayer', 'Hagane no Renkinjutsushi_ Fullmetal Alchemist',
    'Haikyuu!!', 'Hataraku Maou-sama!', 'High School DxD', 'Horimiya', 'Howl no Ugoku Shiro',
    'Hunter x Hunter', 'Hyouka', 'JoJo', 'Jujutsu Kaisen', 'Kaguya-sama wa Kokurasetai_ Tensaitachi no Renai Zunousen',
    'Kakegurui', 'Kami no Tou_ Tower of God', 'Kanojo, Okarishimasu', 'Kill la Kill', 'Kimetsu no Yaiba',
    'Kimi no Na wa', 'Kimi no Suizou wo Tabetai', 'Kiseijuu_ Sei no Kakuritsu', 'Kobayashi-san Chi no Maidragon',
    'Koe no Katachi', 'Komi-san wa, Komyushou desu', 'Kono Subarashii Sekai ni Shukufuku wo!', 'Made in Abyss',
    'Mahou Shoujo Madoka Magica', 'Mirai Nikki', 'Mob Psycho 100', 'Monster', 'Mushoku Tensei_ Isekai Ittara Honki Dasu',
    'Nanatsu no Taizai', 'Naruto', 'No Game No Life', 'Noragami', 'One Piece', 'One Punch Man', 'Overlord',
    'Owari no Seraph', 'Psycho-Pass', 'Re_Zero kara Hajimeru Isekai Seikatsu', 'Saiki Kusuo no Psi-nan',
    'Seishun Buta Yarou wa Bunny Girl Senpai no Yume wo Minai', 'Sen to Chihiro no Kamikakushi', 'Shigatsu wa Kimi no Uso',
    'Shin Seiki Evangelion', 'Shingeki no Kyojin', 'Shokugeki no Souma', 'Sono Bisque Doll wa Koi wo Suru',
    'Soul Eater', 'Sousou no Frieren', 'Spy x Family', 'Steins Gate', 'Sword Art Online',
    'Tate no Yuusha no Nariagari', 'Tengen Toppa Gurren Lagann', 'Tenki no Ko', 'Tensei Shitara Slime Datta Ken',
    'The God of High School', 'Tokyo Ghoul', 'Tokyo Revengers', 'Toradora!', 'Vinland Saga', 'Violet Evergarden',
    'Wotaku ni Koi wa Muzukashii', 'Yahari Ore no Seishun Love Come wa Machigatteiru', 'Yakusoku no Neverland',
    'Youkoso Jitsuryoku Shijou Shugi no Kyoushitsu e'
]

# Load Anime Metadata Dataset
anime_metadata = pd.read_csv(ANIME_METADATA_PATH)

# Model Definition
class MyEfficientNetB0(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        from torchvision.models import efficientnet_b0
        self.network = efficientnet_b0(weights='IMAGENET1K_V1')
        self.network.classifier[1] = nn.Linear(self.network.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.network(x)

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Model
MODEL_PATH = "C:/Users/Senku Ishigami/Downloads/anime_model_full (1).pth"
model = MyEfficientNetB0(num_classes=len(ANIME_TITLES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Prediction Function
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    predicted_index = predicted_class.item()
    prediction_percentage = confidence.item() * 100

    if predicted_index < len(ANIME_TITLES):
        predicted_title = ANIME_TITLES[predicted_index]
    else:
        predicted_title = "Unknown Anime (Index Out of Range)"
    
    return predicted_title, prediction_percentage, probabilities, img



def add_bg_from_local(file_path):
    with open(file_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("data:image/png;base64,{encoded_string}") no-repeat center center fixed;
                background-size: cover;
            }}
            .stButton>button {{
                background-color: #31511E;
                color: #F6FCDF;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #859F3D;
            }}
            .stButton>button:hover {{
                background-color: #859F3D;
                border-color: #F6FCDF;
                color: #1A1A19;
            }}
            h1, h2, h3 {{
                color: #F6FCDF;
                text-align: center;
                font-family: 'Orbitron', sans-serif;
                text-shadow: 3px 3px #1A1A19;
            }}
            .stMarkdown p {{
                color: #F6FCDF;
                font-size: 14px;
                font-family: 'Orbitron', sans-serif;
                text-shadow: 1px 1px #1A1A19;
            }}
            .uploadedFile {{
                background-color: #31511E;
                color: #FFA500;
                border-radius: 8px;
            }}
            .stSidebar {{
                background-color: rgba(26, 26, 25, 0.9);
                color: #F6FCDF;
                padding: 20px;
                border-radius: 10px;
            }}
            .stSidebar .stFileUploader {{
                background-color: #1A1A19;
                border: 2px solid #859F3D;
                border-radius: 8px;
            }}
            .stSlider {{
                color: #F6FCDF;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

background_image_path = "C:/Users/Senku Ishigami/Downloads/1319754.jpeg"
if os.path.exists(background_image_path):
    add_bg_from_local(background_image_path)
else:
    st.warning("Background image not found. Ensure the file is in your dataset directory.")

# Streamlit App
st.title("Anime Classification App")
st.write("Upload an image of an anime to predict its title.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("Classifying...")
    try:
        # Save image temporarily
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        
        
        
        # Predict
        prediction, prediction_percentage, probabilities, img = predict("temp_image.png")
        st.image(img, caption="Processed Image", width=300)  # Resize the image to a smaller size
        st.success(f"**Predicted Title:** {prediction} ({prediction_percentage:.2f}%)")
        st.markdown(
    f"""
    <div style="
        display: flex; 
        align-items: center; 
        justify-content: center; 
        padding: 15px; 
        background: linear-gradient(90deg, #1E90FF, #87CEFA, #00FFFF); 
        border-radius: 10px; 
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5); 
        margin-top: 20px;">
        <h3 style="
            font-family: 'Orbitron', sans-serif; 
            font-size: 24px; 
            color: #FFFFFF; 
            text-shadow: 2px 2px rgba(0, 0, 0, 0.8);">
            Predicted Title: {prediction} ({prediction_percentage:.2f}%)
        </h3>
    </div>
    """,
    unsafe_allow_html=True
)
        # Get the corresponding anime metadata
        anime_info = anime_metadata[anime_metadata['Name'] == prediction]
        if not anime_info.empty:
            description = anime_info.iloc[0]['Synopsis']
            image_url = anime_info.iloc[0]['Image URL']
            rating = anime_info.iloc[0]['Rating']
            premiered = anime_info.iloc[0]['Premiered']
            
            # Limit the description length to 300 characters for a more compact view
            short_description = description[:300] + "..." if len(description) > 300 else description
            
            st.write(f"**Description:** {short_description}")
            st.write(f"**Rating:** {rating}")
            st.write(f"**Premiered:** {premiered}")
            st.image(image_url, caption="Anime Poster", width=200)  # Resize the image to a smaller size
        else:
            st.error(f"No information found for the anime: {prediction}")
    except Exception as e:
        st.error(f"Error: {e}")







