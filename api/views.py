from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from django.shortcuts import render

def home(request):
    return render(request, 'plagiarism_checker/home.html')

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def check_plagiarism(request):
    """Enhanced plagiarism check view with proper error handling"""
    if request.method == 'POST':
        try:
            # Handle both form data and JSON payloads
            if request.content_type == 'application/json':
                data = json.loads(request.body)
                text = data.get('text', '').strip()
            else:
                text = request.POST.get('text', '').strip()

            if not text:
                return JsonResponse({
                    "error": "Text input is required",
                    "status": "failed"
                }, status=400)

            # Your actual plagiarism detection logic here
            results = enhanced_check_plagiarism(text)  # Use your existing function
            
            # Example response structure - modify with your actual data
            return JsonResponse({
                "status": "success",
                "results": {
                    "similarity_score": results.get('similarity_score', 0),
                    "ai_probability": results.get('ai_confidence', 0),
                    "is_plagiarized": results.get('similarity_score', 0) > 0.7,
                    "word_count": results.get('word_count', 0),
                    "writing_characteristics": results.get('stylometrics', {})
                }
            })

        except json.JSONDecodeError:
            return JsonResponse({
                "error": "Invalid JSON format",
                "status": "failed"
            }, status=400)
        except Exception as e:
            return JsonResponse({
                "error": str(e),
                "status": "error"
            }, status=500)

    return JsonResponse({
        "error": "Only POST requests are allowed",
        "status": "failed"
    }, status=405)
