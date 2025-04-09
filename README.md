# AI Content Detector 🔍

A Django-based web application that analyzes text to detect AI-generated content (like ChatGPT) with multi-model verification.

## Screenshots 🖼️
| Home Page | 
![image alt](https://github.com/GonaSaicharan/AIchecker/blob/94bac9b02a5f9a3fca4ea3ae3cb9ba1747776112/home.png)
| Text Input |
![image alt](https://github.com/GonaSaicharan/AIchecker/blob/94bac9b02a5f9a3fca4ea3ae3cb9ba1747776112/textinput.png)
| Results |
![image alt](https://github.com/GonaSaicharan/AIchecker/blob/94bac9b02a5f9a3fca4ea3ae3cb9ba1747776112/results.png)

## Features ✨
- **Multi-model analysis** (5+ detection algorithms)
- **Confidence scoring** (0-100% accuracy rating)
- **Detailed reporting** (repetition, burstiness, perplexity metrics)
- **API endpoint** for programmatic access
- **Responsive UI** works on desktop/mobile

## Tech Stack 🛠️
- **Backend**: Django 4.2 + Python 3.10
- **Frontend**: Bootstrap 5 + Vanilla JS
- **NLP Models**: 
  - RoBERTa-base-openai-detector
  - ChatGPT-detector
  - DeBERTa-v3

## Installation 🚀

### Prerequisites
- Python 3.8+
- Git

### Local Setup
```bash
git clone https://github.com/GonaSaicharan/AIchecker.git
cd AIchecker

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Start dev server
python manage.py runserver
