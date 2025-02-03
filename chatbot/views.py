import json
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.contrib.sessions.models import Session
from .chatbot_nlp import chatbot_response

@csrf_exempt
def chatbot_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "")

            # Ensure session exists
            if not request.session.session_key:
                request.session.create()
            
            session_id = request.session.session_key
            conversation_memory = request.session.get("conversation_memory", {})

            # Process chatbot response
            bot_reply = chatbot_response(session_id, user_message, conversation_memory)

            # Save updated memory back to session
            request.session["conversation_memory"] = conversation_memory
            request.session.modified = True

            return JsonResponse({"response": bot_reply})

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)

    return JsonResponse({"error": "Invalid request"}, status=400)

# View to render chatbot HTML
def chatbot_page(request):
    return render(request, "chatbot.html")