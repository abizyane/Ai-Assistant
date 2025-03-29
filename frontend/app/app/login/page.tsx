"use client"

import { useRouter } from "next/navigation"
import { useEffect } from "react"
import { authenticateWithGoogle } from "../../services/authService"

declare global {
  interface Window {
    google: any
  }
}

const Login = () => {
  const router = useRouter()

  useEffect(() => {
    const script = document.createElement("script")
    script.src = "https://accounts.google.com/gsi/client"
    script.async = true
    script.defer = true
    document.body.appendChild(script)

    return () => {
      document.body.removeChild(script)
    }
  }, [])

  const handleGoogleLogin = async (response: any) => {
    try {
      const result = await authenticateWithGoogle(response.credential)
      console.log("Authentication successful:", result)
      localStorage.setItem("authToken", result.token)
      router.push("/")
    } catch (error) {
      console.error("Authentication failed:", error)
    }
  }

  useEffect(() => {
    if (typeof window !== "undefined" && window.google) {
      window.google.accounts.id.initialize({
        client_id: process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID,
        callback: handleGoogleLogin,
      })

      window.google.accounts.id.renderButton(document.getElementById("googleSignInButton"), {
        theme: "outline",
        size: "large",
      })
    }
  }, [handleGoogleLogin]) // Added handleGoogleLogin to dependencies

  return (
    <div className="min-h-screen flex items-center justify-center bg-background py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-foreground">Welcome to AI Chat Interface</h2>
        </div>
        <div id="googleSignInButton" className="flex justify-center"></div>
      </div>
    </div>
  )
}

export default Login

