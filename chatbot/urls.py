from django.urls import path
from .views import chatbot_api, chatbot_page

urlpatterns = [
    path("chat/", chatbot_api, name="chatbot_api"),
    path("", chatbot_page, name="chatbot_page")
]
