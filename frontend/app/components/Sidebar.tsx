import { useState } from "react"
import { Search, PlusCircle, Trash2 } from "lucide-react"

interface Conversation {
  id: string
  title: string
}

const Sidebar = () => {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [searchTerm, setSearchTerm] = useState("")

  const handleNewConversation = () => {
    const newConversation: Conversation = {
      id: Date.now().toString(),
      title: `New Conversation ${conversations.length + 1}`,
    }
    setConversations([newConversation, ...conversations])
  }

  const handleDeleteConversation = (id: string) => {
    setConversations(conversations.filter((conv) => conv.id !== id))
  }

  const filteredConversations = conversations.filter((conv) =>
    conv.title.toLowerCase().includes(searchTerm.toLowerCase()),
  )

  return (
    <div className="w-64 bg-card border-r border-border flex flex-col">
      <div className="p-4">
        <div className="relative">
          <input
            type="text"
            placeholder="Search conversations"
            className="w-full pl-10 pr-4 py-2 bg-background border border-input rounded-lg text-foreground"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <Search className="absolute left-3 top-3 h-5 w-5 text-muted-foreground" />
        </div>
      </div>
      <button
        onClick={handleNewConversation}
        className="flex items-center justify-center px-4 py-2 bg-primary text-primary-foreground rounded-lg mx-4 mb-4"
      >
        <PlusCircle className="h-5 w-5 mr-2" />
        New Conversation
      </button>
      <div className="flex-1 overflow-y-auto">
        {filteredConversations.length === 0 ? (
          <div className="text-center text-muted-foreground mt-4">No conversations yet</div>
        ) : (
          filteredConversations.map((conv) => (
            <div key={conv.id} className="flex items-center justify-between px-4 py-2 hover:bg-muted">
              <span>{conv.title}</span>
              <button onClick={() => handleDeleteConversation(conv.id)} className="text-destructive">
                <Trash2 className="h-5 w-5" />
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export default Sidebar

