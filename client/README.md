# Speaker Gate Client

Browser-based test client for the Speaker Gate WebSocket server.

## Files

- `index.html` - Main verification interface
- `enroll.html` - Speaker enrollment interface  
- `app.js` - WebSocket client for verification
- `enroll.js` - Enrollment logic
- `audio-processor.js` - AudioWorklet for real-time audio capture
- `styles.css` - Shared styles

## Quick Start

### 1. Start the Server

```bash
cd /path/to/poc
python -m server serve --port 8765 --debug
```

### 2. Serve the Client

The client is static HTML/JS, so you need a simple HTTP server:

```bash
# Option 1: Python's built-in server
cd poc/client
python -m http.server 8080

# Option 2: Node.js (if available)
npx serve .

# Option 3: VS Code Live Server extension
# Right-click index.html → "Open with Live Server"
```

### 3. Open in Browser

- **Verification**: http://localhost:8080/index.html
- **Enrollment**: http://localhost:8080/enroll.html

## Usage

### Enrollment Flow

1. Open `enroll.html`
2. Enter a profile ID (e.g., `user123`)
3. Click "Start Enrollment" 
4. Speak clearly for a few seconds
5. Click "Stop" when done
6. Profile is saved on the server

### Verification Flow

1. Open `index.html`
2. Enter the enrolled profile ID
3. Click "Connect"
4. Speak - the UI shows real-time similarity scores
5. Green = verified speaker, Red = different speaker

## Configuration

Edit the WebSocket URL in `app.js` and `enroll.js` if needed:

```javascript
const WS_URL = 'ws://localhost:8765';
```

## Requirements

- Modern browser with WebRTC support (Chrome, Firefox, Edge)
- Microphone access
- Server running on configured WebSocket port
