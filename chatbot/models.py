from django.db import models

class ChatbotResponse(models.Model):
    question = models.CharField(max_length=255)
    answer = models.TextField()

    def __str__(self):
        return self.question

class Booking(models.Model):
    name = models.CharField(max_length=100)
    stay_period = models.CharField(max_length=50)
    guests = models.IntegerField()
    breakfast = models.BooleanField(default=False)
    payment_method = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.name}'s booking"
