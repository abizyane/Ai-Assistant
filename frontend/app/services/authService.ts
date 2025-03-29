import axios from "axios"

const API_URL = "http://localhost:8080"

export const authenticateWithGoogle = async (token: string) => {
  const response = await axios.post(`${API_URL}/authentication/google/`, { token })
  return response.data
}

