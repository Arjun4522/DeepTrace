import React, { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from 'react-query'
import Header from './components/Header'
import Dashboard from './pages/Dashboard'
import { SocketProvider } from './contexts/SocketContext'
import './App.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

function App() {
  const [isConnected, setIsConnected] = useState(false)

  return (
    <QueryClientProvider client={queryClient}>
      <SocketProvider>
        <Router>
          <div className="min-h-screen bg-gray-50">
            <Header />
            
            <main className="container mx-auto px-4 py-6">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/flows" element={<Dashboard />} />
                <Route path="/search" element={<Dashboard />} />
                <Route path="/analytics" element={<Dashboard />} />
                <Route path="/settings" element={<Dashboard />} />
              </Routes>
            </main>
          </div>
        </Router>
      </SocketProvider>
    </QueryClientProvider>
  )
}

export default App