from django.shortcuts import render
from django.http import HttpResponse
import json
from .models import agent_response

def index(request):
    return render(request, 'index.html', {})

def send_message(request):
    user_input = request.GET['user_input']
    response = {
            'user_input': user_input,
            'agent_response': agent_response(user_input)
            }
    return HttpResponse(json.dumps(response))
