from django.http import HttpResponse
from django.shortcuts import render
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.template import loader
import base64
from .functions import Model
import logging
from django.contrib import messages



def main(request):
    template = loader.get_template('drawletters/main.html')
    context = {

    }
    messages.info(request, 'main view!')
    return HttpResponse(template.render(context, request))


@csrf_exempt
def hook(request):
    model = Model()
    save = 'Done'

    if request.method == 'POST':
        image_b64 = request.POST['imageBase64']
        drawn_digit = request.POST['digit']
        image_encoded = image_b64.split(',')[1]
        image = base64.decodebytes(image_encoded.encode('utf-8'))
        save = model.save_image(drawn_digit, image)

    return HttpResponse(save)
