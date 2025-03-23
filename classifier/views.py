from django.http import JsonResponse
from rest_framework.decorators import api_view
from model import classify_image

@api_view(['POST'])
def classify(request):
    if 'image' not in request.FILES:
        return JsonResponse({"error": "No image uploaded"}, status=400)
    
    image = request.FILES['image']

    try:
        predicted_class = classify_image(image)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    
    return JsonResponse({"class": predicted_class})
