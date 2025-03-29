"use client"

import { useState } from "react"
import Image from "next/image"
import { Menu } from "@headlessui/react"
import { ChevronDown } from "lucide-react"

interface User {
  name: string
  image: string
}

const Header = () => {
  const [user, setUser] = useState<User | null>({
    name: "John Doe",
    image: "https://lh3.googleusercontent.com/a/default-user=s96-c",
  })

  const handleLogout = () => {
    // Implement logout logic here
    console.log("Logging out...")
  }

  return (
    <header className="bg-card shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <h1 className="text-2xl font-bold text-foreground">AI Chat Interface</h1>
        {user && (
          <Menu as="div" className="relative inline-block text-left">
            <Menu.Button className="inline-flex items-center justify-center w-full rounded-md border border-input shadow-sm px-4 py-2 bg-card text-sm font-medium text-foreground hover:bg-muted focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-background focus:ring-primary">
              <Image
                src={user.image || "/placeholder.svg"}
                alt={user.name}
                width={24}
                height={24}
                className="rounded-full mr-2"
              />
              {user.name}
              <ChevronDown className="ml-2 h-5 w-5" aria-hidden="true" />
            </Menu.Button>
            <Menu.Items className="origin-top-right absolute right-0 mt-2 w-56 rounded-md shadow-lg bg-card ring-1 ring-black ring-opacity-5 focus:outline-none">
              <div className="py-1">
                <Menu.Item>
                  {({ active }) => (
                    <button
                      onClick={handleLogout}
                      className={`${
                        active ? "bg-muted text-foreground" : "text-foreground"
                      } block w-full text-left px-4 py-2 text-sm`}
                    >
                      Log out
                    </button>
                  )}
                </Menu.Item>
              </div>
            </Menu.Items>
          </Menu>
        )}
      </div>
    </header>
  )
}

export default Header

