from django.db import models
from authentication.models import User
# Create your models here.

class Conversation(models.Model):
    conversation_id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=255)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class Message(models.Model):
    message_id = models.AutoField(primary_key=True)
    question = models.TextField()
    answer = models.TextField()
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.text