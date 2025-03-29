"use client"

import { useState, useEffect } from "react"
import Layout from "../components/Layout"
import ChatArea from "../components/ChatArea"
import SplashScreen from "../components/SplashScreen"

export default function Home() {
  const [showSplash, setShowSplash] = useState(true)

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowSplash(false)
    }, 2000)

    return () => clearTimeout(timer)
  }, [])

  if (showSplash) {
    return <SplashScreen />
  }

  return (
    <Layout>
      <ChatArea />
    </Layout>
  )
}

