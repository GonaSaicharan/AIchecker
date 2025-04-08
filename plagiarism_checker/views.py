import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import re
import logging
import traceback
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer  # New import

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='ai_detection.log'
)
logger = logging.getLogger(__name__)

# ======================
# SUPERCHARGED MODEL CONFIGURATION
# ======================
DETECTION_MODELS = [
    {
        'name': 'roberta-openai',
        'model_name': "openai-community/roberta-base-openai-detector",
        'weight': 0.30,  # Adjusted weight
        'type': 'transformers',
        'ai_class_index': 1,
        'trust_threshold': 0.80
    },
    {
        'name': 'chatgpt-detector',
        'model_name': "Hello-SimpleAI/chatgpt-detector-roberta",
        'weight': 0.30,  # Adjusted weight
        'type': 'transformers',
        'ai_class_index': 1,
        'trust_threshold': 0.85
    },
    {
        'name': 'xlm-roberta',
        'model_name': "elozano/ai-detector-xlm-roberta",
        'weight': 0.15,  # Reduced weight
        'type': 'transformers',
        'ai_class_index': 0
    },
    {
        'name': 'deberta-v3',
        'model_name': "microsoft/deberta-v3-base-openai-detector",
        'weight': 0.15,
        'type': 'transformers',
        'ai_class_index': 1,
        'trust_threshold': 0.90
    },
    {
        'name': 'gpt2-detector',
        'model_name': "roberta-base-openai-detector",
        'weight': 0.10,
        'type': 'transformers',
        'ai_class_index': 1
    }
]

# Initialize sentence transformer (loaded once)
SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize models with enhanced error handling
def initialize_models():
    """Safe model initialization with retry logic"""
    for model_cfg in DETECTION_MODELS:
        max_retries = 3  # Increased retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing {model_cfg['name']} (attempt {attempt + 1})...")
                
                model_cfg['tokenizer'] = AutoTokenizer.from_pretrained(
                    model_cfg['model_name'],
                    use_fast=True
                )
                
                model_cfg['model'] = AutoModelForSequenceClassification.from_pretrained(
                    model_cfg['model_name']
                )
                
                if torch.cuda.is_available():
                    model_cfg['model'] = model_cfg['model'].to('cuda')
                
                # Enhanced validation
                test_input = model_cfg['tokenizer'](
                    "Test input " * 50,  # Longer test input
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                if torch.cuda.is_available():
                    test_input = {k: v.to('cuda') for k, v in test_input.items()}
                
                with torch.no_grad():
                    outputs = model_cfg['model'](**test_input)
                    if not outputs.logits.any():
                        raise ValueError("Model returned empty logits")
                
                logger.info(f"{model_cfg['name']} initialized successfully")
                break
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {model_cfg['name']}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to initialize {model_cfg['name']} after {max_retries} attempts")
                    model_cfg['model'] = None
                else:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff

initialize_models()

# ======================
# MILITARY-GRADE PREDICTION
# ======================
def get_model_prediction(model_cfg, text):
    """Ultra-robust model prediction with confidence scoring"""
    if model_cfg['model'] is None:
        return None, 0  # Return confidence score
        
    try:
        # Enhanced tokenization with length checks
        inputs = model_cfg['tokenizer'](
            text[:2000],  # Increased buffer
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding='max_length',
            add_special_tokens=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Model inference with memory management
        try:
            with torch.no_grad():
                torch.cuda.empty_cache()
                outputs = model_cfg['model'](**inputs)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning(f"CUDA OOM for {model_cfg['name']}, retrying with CPU")
                inputs = {k: v.to('cpu') for k, v in inputs.items()}
                model_cfg['model'] = model_cfg['model'].to('cpu')
                with torch.no_grad():
                    outputs = model_cfg['model'](**inputs)
            else:
                raise
        
        logits = outputs.logits
        
        # Enhanced scoring with confidence
        if logits.shape[1] == 1:
            score = torch.sigmoid(logits)[0].item() * 100
            confidence = min(1.0, abs(score - 50) / 50)  # How far from 50%
        else:
            probs = torch.softmax(logits, dim=1)[0]
            ai_index = model_cfg.get('ai_class_index', 1)
            score = probs[ai_index].item() * 100
            confidence = probs.max().item()  # Max probability
            
        # Apply trust threshold
        min_confidence = model_cfg.get('trust_threshold', 0.75)
        if confidence < min_confidence:
            logger.warning(f"Low confidence {confidence:.2f} for {model_cfg['name']}")
            return None, confidence
            
        return score, confidence
        
    except Exception as e:
        logger.error(f"Prediction failed for {model_cfg['name']}: {str(e)}")
        logger.error(traceback.format_exc())
        return None, 0

# ======================
# ADVANCED TEXT ANALYSIS
# ======================
def calculate_perplexity(text):
    """Enhanced perplexity with n-gram support"""
    try:
        words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
        if len(words) < 50:  # Increased minimum
            return 50
            
        # Calculate for both unigrams and bigrams
        unigram_counts = Counter(words)
        bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        
        total_uni = len(words)
        total_bi = len(bigrams)
        
        uni_probs = [count/total_uni for count in unigram_counts.values()]
        bi_probs = [count/total_bi for count in bigram_counts.values()]
        
        # Combined entropy
        text_entropy = (entropy(uni_probs) + entropy(bi_probs)) / 2
        perplexity = min(100, max(0, 100 - (text_entropy * 8)))  # Adjusted scaling
        return perplexity
        
    except Exception as e:
        logger.error(f"Perplexity calculation failed: {str(e)}")
        return 50

def calculate_burstiness(text):
    """Enhanced burstiness with paragraph analysis"""
    try:
        paragraphs = [p for p in text.split('\n') if len(p.strip()) > 0]
        sentences = sent_tokenize(text)
        
        if len(sentences) < 8:  # Higher threshold
            return 50
            
        # Sentence length analysis
        lengths = [len(word_tokenize(s)) for s in sentences]
        med = np.median(lengths)
        mad = np.median([abs(l - med) for l in lengths])
        modified_zscores = [0.6745 * (l - med) / mad if mad != 0 else 0 for l in lengths]
        
        # Paragraph length analysis
        para_lengths = [len(word_tokenize(p)) for p in paragraphs]
        para_std = np.std(para_lengths) if len(para_lengths) > 1 else 0
        
        # Combined score
        outliers = sum(1 for z in modified_zscores if abs(z) > 3.5)
        burstiness = min(100, max(0, 
            100 - (outliers/len(sentences)) * 70 - (para_std * 30)
        ))
        return burstiness
        
    except Exception as e:
        logger.error(f"Burstiness calculation failed: {str(e)}")
        return 50

def calculate_repetition(text):
    """Enhanced repetition detection with sliding window"""
    try:
        words = [w.lower() for w in word_tokenize(text) if len(w) > 2]  # Lower threshold
        if len(words) < 50:
            return 0
            
        # Check multiple n-gram sizes
        ngram_sizes = [3, 4, 5]
        repetition_score = 0
        
        for n in ngram_sizes:
            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            counts = Counter(ngrams)
            repeated = sum(1 for cnt in counts.values() if cnt > 1)
            repetition_score += min(30, (repeated / len(counts)) * 100)  # Cap per n-gram
            
        return min(100, repetition_score)
        
    except Exception as e:
        logger.error(f"Repetition calculation failed: {str(e)}")
        return 0

def calculate_formality(text):
    """Enhanced formality with academic markers"""
    try:
        formal_phrases = [
            'in conclusion', 'furthermore', 'moreover', 'however', 
            'therefore', 'additionally', 'consequently', 'thus',
            'hence', 'nevertheless', 'accordingly', 'as evidenced by',
            'in light of', 'it can be argued', 'this suggests that'
        ]
        
        # Count both exact matches and partial matches
        text_lower = text.lower()
        exact_matches = sum(text_lower.count(phrase) for phrase in formal_phrases)
        partial_matches = sum(1 for phrase in formal_phrases if phrase in text_lower)
        
        return min(100, (exact_matches * 8 + partial_matches * 4))
    except Exception as e:
        logger.error(f"Formality calculation failed: {str(e)}")
        return 0

def calculate_semantic_coherence(text):
    """Detect unnatural flow using sentence embeddings"""
    try:
        sentences = [s for s in sent_tokenize(text) if len(s) > 10]
        if len(sentences) < 4:
            return 50
            
        embeddings = SEMANTIC_MODEL.encode(sentences)
        similarities = []
        
        for i in range(len(embeddings)-1):
            sim = np.dot(embeddings[i], embeddings[i+1])
            similarities.append(sim)
        
        # AI text tends to have higher average similarity
        avg_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # Score based on combination of factors
        coherence_score = min(100, max(0,
            (avg_sim * 60) + (std_sim * 40)
        ))
        return coherence_score
    except Exception as e:
        logger.error(f"Semantic coherence failed: {str(e)}")
        return 50

def detect_ai_patterns(text):
    """Advanced pattern detection"""
    patterns = [
        (r'\b(very|quite|rather|somewhat|extremely)\b', 5),  # Excessive qualifiers
        (r'^(in today\'s world|throughout history|it is important)', 10),  # Generic openings
        (r'\b(may|might|could|potentially|possibly)\b', 3),  # Hedging
        (r'\b(as an ai|as a language model)\b', 20),  # AI disclosures
        (r'\b(in summary|to conclude|overall)\b', 7)  # Formulaic conclusions
    ]
    
    score = 0
    text_lower = text.lower()
    for pattern, weight in patterns:
        matches = len(re.findall(pattern, text_lower))
        score += min(20, matches * weight)
        
    return min(100, score)

# ======================
# NUCLEAR DETECTION CORE
# ======================
def analyze_text(text):
    """Ultimate AI detection with all enhancements"""
    result = {
        'score': 0,
        'confidence': 0,
        'features': {},
        'indicators': [],
        'model_details': [],
        'error': None,
        'warnings': []
    }
    
    # Enhanced input validation
    if not text or not isinstance(text, str):
        result['error'] = 'Invalid text input'
        return result
        
    text = text.strip()
    if len(text) < 200:  # Increased minimum
        result['error'] = 'Text too short (minimum 200 characters required)'
        return result
    
    try:
        # Basic text statistics
        try:
            words = word_tokenize(text)
            sentences = sent_tokenize(text)
            paragraphs = [p for p in text.split('\n') if len(p.strip()) > 0]
            
            result['word_count'] = len(words)
            result['sentence_count'] = len(sentences)
            result['paragraph_count'] = len(paragraphs)
        except Exception as e:
            logger.error(f"Tokenization failed: {str(e)}")
            result['error'] = 'Text processing error'
            return result
        
        # Feature analysis
        features = {
            'perplexity': calculate_perplexity(text),
            'burstiness': calculate_burstiness(text),
            'repetition': calculate_repetition(text),
            'formality': calculate_formality(text),
            'semantic_coherence': calculate_semantic_coherence(text),
            'ai_patterns': detect_ai_patterns(text)
        }
        result['features'] = features
        
        # Model predictions with confidence
        model_scores = []
        confidences = []
        
        for model_cfg in DETECTION_MODELS:
            score, confidence = get_model_prediction(model_cfg, text)
            if score is not None:
                weighted_score = score * model_cfg['weight']
                model_scores.append(weighted_score)
                confidences.append(confidence)
                
                result['model_details'].append({
                    'name': model_cfg['name'],
                    'raw_score': score,
                    'weighted_score': weighted_score,
                    'confidence': round(confidence * 100, 1),
                    'status': 'success'
                })
            else:
                result['model_details'].append({
                    'name': model_cfg['name'],
                    'status': 'failed',
                    'confidence': round(confidence * 100, 1) if confidence else 0
                })
                result['warnings'].append(f"{model_cfg['name']} returned low confidence")
        
        # Calculate final score with nuclear fusion
        if model_scores:
            # Weighted average with confidence
            total_weight = sum(
                model_cfg['weight'] * conf 
                for model_cfg, conf in zip(DETECTION_MODELS, confidences)
                if conf > 0
            )
            
            if total_weight > 0:
                model_avg = sum(
                    score * conf 
                    for score, conf in zip(model_scores, confidences)
                ) / total_weight
            else:
                model_avg = sum(model_scores) / sum(m['weight'] for m in DETECTION_MODELS if m['model'])
            
            feature_score = (
                0.25 * features['perplexity'] +
                0.20 * features['burstiness'] + 
                0.15 * features['repetition'] +
                0.10 * features['formality'] +
                0.15 * features['semantic_coherence'] +
                0.15 * features['ai_patterns']
            )
            
            # Apply correlation boosting
            boost_factor = 1.0
            if features['perplexity'] > 75 and features['burstiness'] > 80:
                boost_factor *= 1.25
            if features['repetition'] > 70 and features['formality'] > 70:
                boost_factor *= 1.15
            if features['semantic_coherence'] > 80:
                boost_factor *= 1.10
                
            final_score = min(100, ((model_avg * 0.80) + (feature_score * 0.20)) * boost_factor)
        else:
            # Fallback to feature-only analysis
            feature_score = (
                0.30 * features['perplexity'] +
                0.25 * features['burstiness'] +
                0.20 * features['repetition'] +
                0.15 * features['semantic_coherence'] +
                0.10 * features['ai_patterns']
            )
            final_score = feature_score
            result['warnings'].append("Using fallback analysis (models unavailable)")
        
        # Calculate overall confidence
        active_models = sum(1 for m in result['model_details'] if m['status'] == 'success')
        model_confidence = (active_models / len(DETECTION_MODELS)) * 100
        
        feature_agreement = sum(
            1 for f in ['perplexity', 'burstiness', 'repetition', 'semantic_coherence'] 
            if features[f] > 70
        ) / 4 * 100
        
        result['score'] = min(100, max(0, final_score))
        result['confidence'] = min(100, (model_confidence * 0.6 + feature_agreement * 0.4))
        
        # Generate detailed indicators
        indicators = []
        if features['perplexity'] > 75:
            indicators.append(f"Low lexical diversity (score: {features['perplexity']:.1f})")
        if features['burstiness'] > 80:
            indicators.append(f"Uniform sentence structure (score: {features['burstiness']:.1f})")
        if features['repetition'] > 70:
            indicators.append(f"Phrase repetition detected (score: {features['repetition']:.1f})")
        if features['formality'] > 70:
            indicators.append(f"Overly formal language (score: {features['formality']:.1f})")
        if features['semantic_coherence'] > 75:
            indicators.append(f"Unnatural semantic flow (score: {features['semantic_coherence']:.1f})")
        if features['ai_patterns'] > 60:
            indicators.append(f"Common AI writing patterns (score: {features['ai_patterns']:.1f})")
            
        result['indicators'] = indicators if indicators else ["No strong AI indicators detected"]
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        logger.error(traceback.format_exc())
        result['error'] = 'System error during analysis'
        
    return result

# ======================
# DJANGO VIEW (ENHANCED)
# ======================
@csrf_exempt
def check_plagiarism(request):
    """Military-grade detection endpoint"""
    context = {
        'text': '',
        'error': None,
        'results': None,
        'detailed_report': None
    }
    
    if request.method == 'POST':
        try:
            text = request.POST.get('text', '').strip()
            context['text'] = text
            
            if not text:
                context['error'] = 'Please enter text to analyze'
                return render(request, 'plagiarism_checker/check.html', context)
                
            if len(word_tokenize(text)) < 100:  # Higher minimum
                context['error'] = 'Minimum 100 words required for accurate analysis'
                return render(request, 'plagiarism_checker/check.html', context)
            
            # Perform analysis
            analysis = analyze_text(text)
            
            if analysis.get('error'):
                context['error'] = analysis['error']
                return render(request, 'plagiarism_checker/check.html', context)
            
            # Enhanced risk classification
            score = analysis['score']
            confidence = analysis['confidence']
            
            if confidence < 60:
                risk = ('Unreliable', 'secondary', f'{score:.1f}% score', 'Low confidence in result')
            elif score >= 90:
                risk = ('Certain AI', 'danger', '>90% AI probability', 'Virtual certainty of AI generation')
            elif score >= 80:
                risk = ('High Risk', 'danger', '80-89% AI probability', 'Strong evidence of AI generation')
            elif score >= 65:
                risk = ('Likely AI', 'warning', '65-79% AI probability', 'Probable AI generation')
            elif score >= 50:
                risk = ('Suspicious', 'info', '50-64% AI probability', 'Possible AI assistance')
            else:
                risk = ('Likely Human', 'success', '<50% AI probability', 'Probably human-written')
            
            # Prepare detailed report
            model_details = []
            for model in analysis.get('model_details', []):
                model_details.append({
                    'name': model['name'],
                    'score': f"{model.get('raw_score', 'N/A')}%",
                    'weight': f"{model.get('weighted_score', 'N/A')}",
                    'confidence': f"{model.get('confidence', 0)}%",
                    'status': model['status']
                })
            
            context['results'] = {
                'score': round(score, 1),
                'confidence': round(confidence, 1),
                'risk_level': risk[0],
                'risk_class': risk[1],
                'risk_probability': risk[2],
                'risk_description': risk[3],
                'word_count': analysis['word_count'],
                'sentence_count': analysis['sentence_count'],
                'paragraph_count': analysis.get('paragraph_count', 1),
                'indicators': analysis['indicators']
            }
            
            context['detailed_report'] = {
                'model_details': model_details,
                'feature_scores': {
                    'perplexity': round(analysis['features'].get('perplexity', 0), 1),
                    'burstiness': round(analysis['features'].get('burstiness', 0), 1),
                    'repetition': round(analysis['features'].get('repetition', 0), 1),
                    'formality': round(analysis['features'].get('formality', 0), 1),
                    'semantic_coherence': round(analysis['features'].get('semantic_coherence', 0), 1),
                    'ai_patterns': round(analysis['features'].get('ai_patterns', 0), 1)
                },
                'warnings': analysis.get('warnings', [])
            }
            
        except Exception as e:
            logger.error(f"Endpoint error: {str(e)}")
            logger.error(traceback.format_exc())
            context['error'] = 'System error during analysis. Please try again.'
    
    return render(
        request, 
        'plagiarism_checker/results.html' if context.get('results') else 'plagiarism_checker/check.html', 
        context
    )
def home(request):
    """Simple home view"""
    return render(request, 'plagiarism_checker/home.html')

def results_view(request):
    """Handle results page view"""
    if request.method == 'POST':
        return check_plagiarism(request)
    return redirect('check_plagiarism')