from django.shortcuts import render, HttpResponse
import os
from .main_1 import pipeline
import tempfile

# Create your views here.

def home(request):
    return render(request, "home.html")

def run_pipeline(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image_file = request.FILES['image']

            # Create a temporary file to save the uploaded image
            with tempfile.NamedTemporaryFile(delete=False) as temp_image:
                temp_image.write(image_file.read())
                temp_image_path = temp_image.name

            # Process the image with the pipeline function
            number_plate = pipeline(temp_image_path)

            # Check if the pipeline function returned a result
            if number_plate:
                return render(request, "home.html", {'result': number_plate})
            else:
                return render(request, "home.html", {'error_message': 'Valid number plate not found'})
        except Exception as e:
            return render(request, "home.html", {'error_message': str(e)})
        finally:
            # Clean up by removing the temporary file
            os.unlink(temp_image_path)
    else:
        return render(request, "home.html")