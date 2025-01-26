from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from .models import Conversation, Message
from .serializers import ConversationSerializer, ConversationListSerializer, MessageSerializer
from .rag_service import RAGService

class ConversationView(generics.RetrieveAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = ConversationSerializer

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user)

    def get_object(self):
        conversation_id = self.kwargs.get('id')
        return get_object_or_404(self.get_queryset(), id=conversation_id)

class ConversationsHistoryView(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = ConversationListSerializer

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user)

class DeleteConversationView(generics.DestroyAPIView):
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user)

    def get_object(self):
        conversation_id = self.kwargs.get('id')
        return get_object_or_404(self.get_queryset(), id=conversation_id)

class DeleteMessageView(generics.DestroyAPIView):
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Message.objects.filter(conversation__user=self.request.user)

    def get_object(self):
        message_id = self.kwargs.get('id')
        return get_object_or_404(self.get_queryset(), id=message_id)

class SendMessageView(generics.CreateAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = MessageSerializer
    rag_service = RAGService()

    def create(self, request, *args, **kwargs):
        message_content = request.data.get('message')
        conversation_id = request.data.get('conversation_id')

        if not message_content:
            return Response(
                {'error': 'Message content is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            if conversation_id:
                conversation = get_object_or_404(
                    Conversation.objects.filter(user=request.user), 
                    conversation_id=conversation_id
                )
            else:
                conversation = Conversation.objects.create(
                    user=request.user,
                    title=message_content[:50]  # Use first 50 chars of message as title
                )

            response = self.rag_service.get_response(message_content, conversation)
            return Response({
                'conversation_id': conversation.conversation_id,
                'response': response
            })
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )