import nltk
import os
from pathlib import Path


def initialize_nltk():
    try:
        # Set the NLTK data directory
        nltk_dir = os.path.join(str(Path.home()), 'nltk_data')
        os.makedirs(nltk_dir, exist_ok=True)
        nltk.data.path.append(nltk_dir)

        # Download required data (only if missing)
        nltk.download('punkt', download_dir=nltk_dir, quiet=True)
        nltk.download('punkt_tab', download_dir=nltk_dir, quiet=True)
        nltk.download('stopwords', download_dir=nltk_dir, quiet=True)

    except Exception as e:
        print(f"NLTK initialization error: {e}")


# Initialize when the app loads
initialize_nltk()