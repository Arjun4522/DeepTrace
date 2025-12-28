import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Network, BarChart3, Search, Settings, Activity } from 'lucide-react'

const Header = () => {
  const location = useLocation()
  
  const navigation = [
    { name: 'Dashboard', href: '/', icon: Activity },
    { name: 'Flows', href: '/flows', icon: Network },
    { name: 'Search', href: '/search', icon: Search },
    { name: 'Analytics', href: '/analytics', icon: BarChart3 },
    { name: 'Settings', href: '/settings', icon: Settings },
  ]

  return (
    <header className="bg-white shadow-sm border-b">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center mr-3">
              <Network className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold text-gray-900">DeepTrace</h1>
          </div>

          {/* Navigation */}
          <nav className="flex space-x-1">
            {navigation.map((item) => {
              const Icon = item.icon
              const isActive = location.pathname === item.href
              
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <Icon className="w-4 h-4 mr-2" />
                  {item.name}
                </Link>
              )
            })}
          </nav>

          {/* Status */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2" />
              <span className="text-sm text-gray-600">Connected</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header