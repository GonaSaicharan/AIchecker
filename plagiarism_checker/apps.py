from django.apps import AppConfig
import nltk

class PlagiarismCheckerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'plagiarism_checker'

    def ready(self):
     
        try:
            nltk.download('punkt_tab', download_dir='C:/Users/saich/nltk_data')
            nltk.download('punkt') 
        except Exception as e:
            print(f"NLTK download error: {e}")
