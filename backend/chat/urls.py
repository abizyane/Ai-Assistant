from django.urls import path
from .views import ConversationView, ConversationsHistoryView, DeleteConversationView, SendMessageView, DeleteMessageView

urlpatterns = [
    path('conversation/<int:id>/', ConversationView.as_view(), name='conversation'),
    path('history/', ConversationsHistoryView.as_view(), name='conversations_history'),
    path('delconversation/<int:id>/', DeleteConversationView.as_view(), name='delete_conversation'),
    path('delmessage/<int:id>/', DeleteMessageView.as_view(), name='delete_message'),
    path('sendmessage/', SendMessageView.as_view(), name='send_message'),
]

