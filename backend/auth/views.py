from django.shortcuts import render
from rest_framework import status, views
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework_simplejwt.tokens import RefreshToken
from google.oauth2 import id_token
from google.auth.transport import requests
from django.conf import settings
from .serializers import UserSerializer
from django.contrib.auth import get_user_model

User = get_user_model()

class GoogleAuthView(views.APIView):
    permission_classes = (AllowAny,)

    def post(self, request):
        token = request.data.get('token')
        
        try:
            idinfo = id_token.verify_oauth2_token(
                token,
                requests.Request(),
                settings.GOOGLE_OAUTH2_CLIENT_ID
            )

            email = idinfo['email']
            google_id = idinfo['sub']
            
            user, created = User.objects.get_or_create(
                google_id=google_id,
                defaults={
                    'email': email,
                    'username': email.split('@')[0],
                    'avatar_url': idinfo.get('picture', '')
                }
            )

            refresh = RefreshToken.for_user(user)
            
            return Response({
                'user': UserSerializer(user).data,
                'tokens': {
                    'refresh': str(refresh),
                    'access': str(refresh.access_token),
                }
            })

        except ValueError:
            return Response(
                {'error': 'Invalid token'},
                status=status.HTTP_400_BAD_REQUEST
            )