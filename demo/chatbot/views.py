from django.shortcuts import render
from django.http import HttpResponse
import json
from .models import agent_response, reset_agent, get_timetable


def reset(request):
    sid = request.session.session_key
    reset_agent(sid)
    print(get_timetable(sid))
    return HttpResponse('reset success.')

def index(request):
    if not request.session.session_key:
        request.session.save()
    reset(request)
    sid = request.session.session_key
    return render(request, 'index.html', get_timetable(sid))

def send_message(request):
    user_input = request.GET['user_input']
    sid = request.session.session_key
    return HttpResponse(json.dumps(agent_response(user_input, sid)))
