from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    email = models.EmailField(unique=True)
    google_id = models.CharField(max_length=100, unique=True)
    avatar_url = models.URLField(max_length=255, blank=True, null=True)
    groups = None
    user_permissions = None

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']
    
    def __str__(self):
        return self.email