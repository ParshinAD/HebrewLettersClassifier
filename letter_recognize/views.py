from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from .functions import predict_on_one_image, one_step_train
import base64
import json



def main(request):
    template = loader.get_template('letter_recognize/main.html')
    context = {

    }
    return HttpResponse(template.render(context, request))


@csrf_exempt
def hook2(request):
    if request.method == 'POST':
        image_b64 = request.POST['imageBase64']
        image_encoded = image_b64.split(',')[1]
        image = base64.decodebytes(image_encoded.encode('utf-8'))
        prediction = predict_on_one_image(image)

    return JsonResponse(prediction)


@csrf_exempt
def hook3(request):
    if request.method == 'POST':
        image_b64 = request.POST['imageBase64']
        digit = request.POST['digit']
        image_encoded = image_b64.split(',')[1]
        image = base64.decodebytes(image_encoded.encode('utf-8'))
        result = one_step_train(image, digit)

    return HttpResponse(result)