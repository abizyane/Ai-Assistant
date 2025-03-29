"use client"

import { motion } from "framer-motion"

const SplashScreen = () => {
  return (
    <div className="flex items-center justify-center h-screen bg-background">
      <motion.h1
        className="text-4xl font-bold text-primary"
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, ease: "easeOut" }}
      >
        AI Chat Interface
      </motion.h1>
    </div>
  )
}

export default SplashScreen

