"use client"

import { useState, useEffect, useRef } from "react"
import { Trash2 } from "lucide-react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import axios from "axios"

interface Message {
  message_id: number
  question: string
  answer: string
  created_at: string
  updated_at: string
}

interface Conversation {
  conversation_id: number
  title: string
  created_at: string
  updated_at: string
  messages: Message[]
}

const TypingAnimation = () => (
  <div className="typing-animation">
    <span></span>
    <span></span>
    <span></span>
  </div>
)

const API_URL = "http://localhost:8080"

const fetchConversations = async (): Promise<Conversation[]> => {
  const token = localStorage.getItem("authToken")
  const response = await axios.get(`${API_URL}/chat/history/`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  return response.data
}

const ChatArea = () => {
  const [input, setInput] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const [currentConversationId, setCurrentConversationId] = useState<number | null>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)
  const queryClient = useQueryClient()

  const { data: conversations = [] } = useQuery({
    queryKey: ["conversations"],
    queryFn: fetchConversations,
  })

  const sendMessageMutation = useMutation({
    mutationFn: async ({ message, conversation_id }: { message: string; conversation_id: number | null }) => {
      const token = localStorage.getItem("authToken")
      const response = await axios.post(
        `${API_URL}/chat/sendmessage/`,
        { message, conversation_id },
        { headers: { Authorization: `Bearer ${token}` } },
      )
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["conversations"] })
    },
  })

  const deleteConversationMutation = useMutation({
    mutationFn: (id: number) => {
      const token = localStorage.getItem("authToken")
      return axios.delete(`${API_URL}/chat/delconversation/${id}/`, {
        headers: { Authorization: `Bearer ${token}` },
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["conversations"] })
    },
  })

  const deleteMessageMutation = useMutation({
    mutationFn: (id: number) => {
      const token = localStorage.getItem("authToken")
      return axios.delete(`${API_URL}/chat/delmessage/${id}/`, {
        headers: { Authorization: `Bearer ${token}` },
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["conversations"] })
    },
  })

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim()) {
      setIsTyping(true)
      try {
        await sendMessageMutation.mutateAsync({ message: input.trim(), conversation_id: currentConversationId })
        setInput("")
      } catch (error) {
        console.error("Error sending message:", error)
      } finally {
        setIsTyping(false)
      }
    }
  }

  const handleDeleteConversation = (id: number) => {
    deleteConversationMutation.mutate(id)
  }

  const handleDeleteMessage = (id: number) => {
    deleteMessageMutation.mutate(id)
  }

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }, [chatContainerRef])

  return (
    <div className="flex flex-col h-full">
      <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4">
        {conversations.map((conversation) => (
          <div key={conversation.conversation_id} className="space-y-2">
            <h3 className="font-bold text-lg">{conversation.title}</h3>
            {conversation.messages.map((message) => (
              <div key={message.message_id} className="space-y-2">
                <div className="flex justify-end">
                  <div className="relative max-w-xl px-4 py-2 rounded-lg bg-primary text-primary-foreground">
                    <button
                      onClick={() => handleDeleteMessage(message.message_id)}
                      className="absolute top-1 right-1 text-xs opacity-50 hover:opacity-100"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                    <p>{message.question}</p>
                    <span className="block text-xs mt-1 opacity-50">
                      {new Date(message.created_at).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
                <div className="flex justify-start">
                  <div className="relative max-w-xl px-4 py-2 rounded-lg bg-secondary text-secondary-foreground">
                    <p>{message.answer}</p>
                    <span className="block text-xs mt-1 opacity-50">
                      {new Date(message.created_at).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              </div>
            ))}
            <button
              onClick={() => handleDeleteConversation(conversation.conversation_id)}
              className="text-destructive hover:underline"
            >
              Delete Conversation
            </button>
          </div>
        ))}
        {isTyping && (
          <div className="flex justify-start">
            <div className="bg-secondary text-secondary-foreground px-4 py-2 rounded-lg">
              <TypingAnimation />
            </div>
          </div>
        )}
      </div>
      <form onSubmit={handleSendMessage} className="p-4 bg-card border-t border-border">
        <div className="flex space-x-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 bg-background border border-input rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary text-foreground"
          />
          <button
            type="submit"
            className="bg-primary text-primary-foreground px-4 py-2 rounded-lg hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-background"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  )
}

export default ChatArea

