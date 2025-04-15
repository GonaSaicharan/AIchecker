import nltk
import os
from pathlib import Path


def initialize_nltk():
    try:
     
        nltk_dir = os.path.join(str(Path.home()), 'nltk_data')
        os.makedirs(nltk_dir, exist_ok=True)
        nltk.data.path.append(nltk_dir)

   
        nltk.download('punkt', download_dir=nltk_dir, quiet=True)
        nltk.download('punkt_tab', download_dir=nltk_dir, quiet=True)
        nltk.download('stopwords', download_dir=nltk_dir, quiet=True)

    except Exception as e:
        print(f"NLTK initialization error: {e}")


initialize_nltk()
