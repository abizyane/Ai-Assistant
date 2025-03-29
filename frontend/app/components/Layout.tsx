import type { PropsWithChildren } from "react"
import Sidebar from "./Sidebar"
import Header from "./Header"

const Layout = ({ children }: PropsWithChildren) => {
  return (
    <div className="flex h-screen bg-background text-foreground">
      <Sidebar />
      <div className="flex flex-col flex-1 overflow-hidden">
        <Header />
        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-background">{children}</main>
      </div>
    </div>
  )
}

export default Layout

