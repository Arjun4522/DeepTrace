import React, { useState, useEffect } from 'react'
import { useQuery } from 'react-query'
import { 
  Activity, 
  Network, 
  BarChart3, 
  TrendingUp, 
  AlertTriangle,
  Clock,
  Download,
  Upload
} from 'lucide-react'
import axios from 'axios'

const Dashboard = () => {
  const [stats, setStats] = useState({})
  const [recentFlows, setRecentFlows] = useState([])

  const { data: healthData } = useQuery('health', async () => {
    const response = await axios.get('/api/health')
    return response.data
  }, {
    refetchInterval: 5000
  })

  const { data: statistics } = useQuery('stats', async () => {
    const response = await axios.get('/api/stats')
    return response.data
  }, {
    refetchInterval: 10000
  })

  const { data: flowsData } = useQuery('recent-flows', async () => {
    const response = await axios.get('/api/flows?limit=5')
    return response.data
  })

  const statsCards = [
    {
      title: 'Total Flows',
      value: statistics?.total_flows || 0,
      icon: Network,
      color: 'blue',
      trend: '+12%'
    },
    {
      title: 'Total Packets',
      value: statistics?.total_packets?.toLocaleString() || '0',
      icon: Activity,
      color: 'green',
      trend: '+8%'
    },
    {
      title: 'Total Bytes',
      value: statistics?.total_bytes ? `${(statistics.total_bytes / 1024 / 1024).toFixed(2)} MB` : '0 MB',
      icon: BarChart3,
      color: 'purple',
      trend: '+15%'
    },
    {
      title: 'Avg Flow Size',
      value: statistics?.avg_bytes_per_flow ? `${Math.round(statistics.avg_bytes_per_flow)} bytes` : '0 bytes',
      icon: TrendingUp,
      color: 'orange',
      trend: '+5%'
    }
  ]

  const protocolDistribution = statistics?.protocol_distribution || {}
  const appProtocolDistribution = statistics?.app_protocol_distribution || {}

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-1">Real-time network traffic monitoring and analysis</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statsCards.map((card, index) => {
          const Icon = card.icon
          const colorClasses = {
            blue: 'bg-blue-100 text-blue-600',
            green: 'bg-green-100 text-green-600',
            purple: 'bg-purple-100 text-purple-600',
            orange: 'bg-orange-100 text-orange-600'
          }

          return (
            <div key={index} className="card">
              <div className="stat-card">
                <div>
                  <p className="stat-label">{card.title}</p>
                  <p className="stat-value">{card.value}</p>
                  <p className="stat-trend trend-up">{card.trend}</p>
                </div>
                <div className={`p-3 rounded-full ${colorClasses[card.color]}`}>
                  <Icon className="w-6 h-6" />
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Protocol Distribution */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-gray-900">Protocol Distribution</h2>
          </div>
          <div className="card-body">
            <div className="space-y-3">
              {Object.entries(protocolDistribution).map(([protocol, count]) => (
                <div key={protocol} className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-600">{protocol}</span>
                  <div className="flex items-center">
                    <div className="progress-bar">
                      <div 
                        className="progress-fill" 
                        style={{ width: `${(count / statistics?.total_flows) * 100 || 0}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium text-gray-900 ml-3">{count}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Application Protocol Distribution */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-gray-900">Application Protocols</h2>
          </div>
          <div className="card-body">
            <div className="space-y-3">
              {Object.entries(appProtocolDistribution)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 5)
                .map(([appProto, count]) => (
                  <div key={appProto} className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-600">{appProto}</span>
                    <div className="flex items-center">
                      <div className="progress-bar">
                        <div 
                          className="progress-fill" 
                          style={{ width: `${(count / statistics?.total_flows) * 100 || 0}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-900 ml-3">{count}</span>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        </div>
      </div>

      {/* Recent Flows */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-lg font-semibold text-gray-900">Recent Network Flows</h2>
          <span className="live-indicator">
            <div className="live-dot"></div>
            LIVE
          </span>
        </div>
        <div className="card-body">
          {flowsData?.flows?.length > 0 ? (
            <table className="flow-table">
              <thead>
                <tr>
                  <th>Source</th>
                  <th>Destination</th>
                  <th>Protocol</th>
                  <th>Packets</th>
                  <th>Size</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {flowsData.flows.map((flow) => (
                  <tr key={flow.id}>
                    <td>
                      <div className="font-medium">
                        {flow.data.flow_id?.src_ip}
                      </div>
                      <div className="text-sm text-gray-500">
                        :{flow.data.flow_id?.src_port}
                      </div>
                    </td>
                    <td>
                      <div className="font-medium">
                        {flow.data.flow_id?.dst_ip}
                      </div>
                      <div className="text-sm text-gray-500">
                        :{flow.data.flow_id?.dst_port}
                      </div>
                    </td>
                    <td>
                      <span className={`protocol-badge protocol-${flow.data.protocol?.proto?.toLowerCase()}`}>
                        {flow.data.protocol?.proto}
                      </span>
                    </td>
                    <td className="font-medium">
                      {flow.data.statistical?.packet_count}
                    </td>
                    <td className="font-medium">
                      {(flow.data.statistical?.total_bytes / 1024).toFixed(1)} KB
                    </td>
                    <td className="text-sm text-gray-500">
                      {new Date(flow.data.timestamp).toLocaleTimeString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-gray-500 text-center py-8">No recent flows found</p>
          )}
        </div>
      </div>
    </div>
  )
}



export default Dashboard