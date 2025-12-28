import React, { createContext, useContext, useEffect, useState } from 'react'
import { io } from 'socket.io-client'

const SocketContext = createContext()

export const useSocket = () => {
  const context = useContext(SocketContext)
  if (!context) {
    throw new Error('useSocket must be used within a SocketProvider')
  }
  return context
}

export const SocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [captureStatus, setCaptureStatus] = useState(null)

  useEffect(() => {
    const newSocket = io('http://localhost:5000', {
      transports: ['websocket', 'polling']
    })

    newSocket.on('connect', () => {
      console.log('Connected to server')
      setIsConnected(true)
    })

    newSocket.on('disconnect', () => {
      console.log('Disconnected from server')
      setIsConnected(false)
    })

    newSocket.on('connected', (data) => {
      console.log('Server connection established:', data)
    })

    newSocket.on('capture_started', (data) => {
      console.log('Capture started:', data)
      setCaptureStatus({ ...data, status: 'running' })
    })

    newSocket.on('capture_stopped', (data) => {
      console.log('Capture stopped:', data)
      setCaptureStatus({ ...data, status: 'stopped' })
    })

    newSocket.on('capture_update', (data) => {
      console.log('Capture update:', data)
      setCaptureStatus(prev => ({ ...prev, ...data, status: 'running' }))
    })

    newSocket.on('capture_error', (data) => {
      console.error('Capture error:', data)
      setCaptureStatus({ ...data, status: 'error' })
    })

    setSocket(newSocket)

    return () => {
      newSocket.disconnect()
    }
  }, [])

  const startCapture = (interfaceName = 'wlo1') => {
    if (socket) {
      socket.emit('start_capture', { interface: interfaceName })
    }
  }

  const stopCapture = () => {
    if (socket) {
      socket.emit('stop_capture')
    }
  }

  const value = {
    socket,
    isConnected,
    captureStatus,
    startCapture,
    stopCapture
  }

  return (
    <SocketContext.Provider value={value}>
      {children}
    </SocketContext.Provider>
  )
}