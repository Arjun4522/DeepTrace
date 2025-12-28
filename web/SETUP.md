# DeepTrace Web Dashboard Setup

This guide will help you set up the DeepTrace web dashboard for real-time network traffic monitoring and analysis.

## Prerequisites

- Node.js 16+ and npm
- Python 3.8+
- pip package manager

## Installation

### 1. Install Python Dependencies

```bash
cd /home/arjun/Desktop/deeptrace
pip install -r requirements.txt
```

The requirements now include:
- Flask web framework
- Flask-CORS for cross-origin requests
- Flask-SocketIO for real-time updates
- Other DeepTrace dependencies

### 2. Install Node.js Dependencies

```bash
cd web
npm install
```

This will install:
- React 18
- Vite build tool
- React Router for navigation
- Axios for HTTP requests
- Socket.IO client for real-time updates
- Recharts for data visualization
- Tailwind CSS for styling
- Lucide React for icons

### 3. Start the Development Servers

**Terminal 1 - Backend API:**
```bash
cd /home/arjun/Desktop/deeptrace
python web/app.py
```

**Terminal 2 - Frontend Development Server:**
```bash
cd /home/arjun/Desktop/deeptrace/web
npm run dev
```

### 4. Access the Dashboard

Open your browser and navigate to:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

## Features

The web dashboard includes:

### Dashboard
- Real-time statistics overview
- Protocol distribution charts
- Recent flows display
- System health monitoring

### Flows Browser
- Paginated list of all captured flows
- Detailed flow information
- Flow metadata visualization

### Search Interface
- IP address search
- Protocol filtering
- Port-based search
- Similarity search (requires model)

### Analytics
- Traffic patterns visualization
- Anomaly detection
- Protocol analysis
- Traffic volume charts

### Real-time Updates
- WebSocket connections for live data
- Capture status monitoring
- Live traffic updates

## API Endpoints

The backend provides these REST endpoints:

- `GET /api/health` - System health check
- `GET /api/flows` - List flows with pagination
- `GET /api/flows/<id>` - Get specific flow details
- `GET /api/search/ip/<address>` - Search by IP
- `GET /api/search/protocol/<name>` - Search by protocol
- `GET /api/search/port/<number>` - Search by port
- `GET /api/similar/<id>` - Find similar flows
- `GET /api/anomalies` - Detect anomalies
- `GET /api/stats` - Get statistics

## WebSocket Events

Real-time events:
- `connect` - Client connection
- `disconnect` - Client disconnection
- `start_capture` - Start packet capture
- `stop_capture` - Stop packet capture
- `capture_update` - Live capture statistics

## Development

### Backend Development

The main API server is in `web/app.py`. Key components:
- Flask application with CORS support
- Socket.IO for real-time communication
- Integration with DeepTrace vector store
- Model inference for similarity search

### Frontend Development

The React app is in `web/src/`:
- `App.jsx` - Main application component
- `components/` - Reusable UI components
- `pages/` - Main view components
- `contexts/` - React contexts for state management
- `index.css` - Tailwind CSS styles

### Building for Production

```bash
cd web
npm run build
```

This creates optimized production files in `web/dist/`.

## Troubleshooting

### Common Issues

1. **Dependency errors**: Run `npm install` and `pip install -r requirements.txt`
2. **Port conflicts**: Change ports in `vite.config.js` and `app.py`
3. **CORS errors**: Ensure Flask-CORS is properly configured
4. **Socket.IO errors**: Check WebSocket transport configuration

### Debug Mode

Enable debug mode for detailed logs:
```bash
DEBUG=true python web/app.py
```

## Production Deployment

For production deployment:

1. Build the frontend: `npm run build`
2. Serve static files from Flask
3. Use production WSGI server (Gunicorn)
4. Configure reverse proxy (Nginx)
5. Set up environment variables

## Next Steps

- [ ] Add authentication and user management
- [ ] Implement advanced visualizations
- [ ] Add export functionality
- [ ] Integrate with external SIEM systems
- [ ] Add alerting and notifications