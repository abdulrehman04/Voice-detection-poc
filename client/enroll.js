// ============== CONFIGURATION ==============
// Set to true when testing from phone via cloudflare tunnel
const USE_TUNNEL = false;
const WS_TUNNEL_URL = "wss://YOUR-WEBSOCKET-TUNNEL.trycloudflare.com"; // Update this when you run cloudflared
const MIN_RECORDING_SECONDS = 15;
// ============================================

const recordBtn = document.getElementById('recordBtn');
const stopRecordBtn = document.getElementById('stopRecordBtn');
const enrollProfileIdInput = document.getElementById('enrollProfileId');
const statusMsg = document.getElementById('statusMsg');

let audioContext;
let workletNode;
let source;
let ws;
let isRecording = false;
let recordingStartTime = null;

function updateRecordingStatus() {
    if (!isRecording || !recordingStartTime) return;
    
    const elapsed = (Date.now() - recordingStartTime) / 1000;
    const remaining = Math.max(0, MIN_RECORDING_SECONDS - elapsed);
    
    if (remaining > 0) {
        statusMsg.innerHTML = `🎙️ Recording... <b>${Math.ceil(remaining)}s</b> minimum remaining<br><small>Keep speaking clearly</small>`;
        stopRecordBtn.disabled = true;
        stopRecordBtn.textContent = `Wait ${Math.ceil(remaining)}s...`;
    } else {
        statusMsg.innerHTML = `🎙️ Recording... <b>${Math.floor(elapsed)}s</b><br><small>You can stop now or continue for better quality</small>`;
        stopRecordBtn.disabled = false;
        stopRecordBtn.textContent = "Stop Recording";
    }
}

async function startRecording() {
    const profileId = enrollProfileIdInput.value;
    if (!profileId) {
        alert("Please enter a Profile ID");
        return;
    }

    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });

        // Use tunnel URL or localhost based on config
        const wsUrl = USE_TUNNEL 
            ? WS_TUNNEL_URL 
            : `ws://${window.location.hostname || 'localhost'}:8765`;
        ws = new WebSocket(wsUrl);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            recordingStartTime = Date.now();
            updateRecordingStatus();
            // Update status every second
            window.recordingInterval = setInterval(updateRecordingStatus, 1000);
            
            ws.send(JSON.stringify({
                type: "enroll",
                profile_id: profileId,
                audio_format: "f32le",
                sr: 16000
            }));
        };

        ws.onmessage = (event) => {
            if (typeof event.data === 'string') {
                const data = JSON.parse(event.data);
                if (data.type === 'enroll_complete') {
                    clearInterval(window.recordingInterval);
                    statusMsg.innerHTML = `✅ Enrollment Complete!<br>Consistency Score: <b>${data.score.toFixed(3)}</b><br><small>Redirecting...</small>`;
                    statusMsg.className = 'status-success';
                    setTimeout(() => {
                        window.location.href = `index.html?profile_id=${data.profile_id}`;
                    }, 3000);
                } else if (data.type === 'error') {
                    clearInterval(window.recordingInterval);
                    statusMsg.innerHTML = `❌ Error: ${data.message || 'Enrollment failed'}<br><small>Please try again with more audio</small>`;
                    statusMsg.className = 'status-error';
                    resetUI();
                }
            }
        };
        
        ws.onerror = (err) => {
            clearInterval(window.recordingInterval);
            console.error("WebSocket error:", err);
            statusMsg.innerHTML = `❌ Connection error<br><small>Check if the server is running</small>`;
            statusMsg.className = 'status-error';
            resetUI();
        };
        
        ws.onclose = (event) => {
            clearInterval(window.recordingInterval);
            if (isRecording) {
                // Unexpected close
                statusMsg.innerHTML = `❌ Connection lost<br><small>Please try again</small>`;
                statusMsg.className = 'status-error';
                resetUI();
            }
        };

        source = audioContext.createMediaStreamSource(stream);
        
        // Use AudioWorklet (modern API, replaces deprecated ScriptProcessor)
        await audioContext.audioWorklet.addModule('audio-processor.js');
        workletNode = new AudioWorkletNode(audioContext, 'audio-capture-processor');
        
        // Receive audio chunks from the worklet and send to server
        workletNode.port.onmessage = (event) => {
            if (ws && ws.readyState === WebSocket.OPEN && isRecording) {
                ws.send(event.data);
            }
        };

        source.connect(workletNode);

        isRecording = true;
        recordBtn.disabled = true;
        stopRecordBtn.disabled = false;
        enrollProfileIdInput.disabled = true;

    } catch (err) {
        console.error("Error starting recording:", err);
        alert("Could not start recording: " + err.message);
    }
}

function stopRecording() {
    isRecording = false;
    statusMsg.textContent = "Processing enrollment...";
    
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: "end_enroll",
            profile_id: enrollProfileIdInput.value
        }));
    }

    if (workletNode) {
        workletNode.disconnect();
        workletNode = null;
    }
    if (source) {
        source.disconnect();
        source = null;
    }
    // Don't close audioContext immediately, wait for WS response
}

function resetUI() {
    isRecording = false;
    recordingStartTime = null;
    recordBtn.disabled = false;
    stopRecordBtn.disabled = true;
    stopRecordBtn.textContent = "Stop Recording";
    enrollProfileIdInput.disabled = false;
    
    if (workletNode) {
        workletNode.disconnect();
        workletNode = null;
    }
    if (source) {
        source.disconnect();
        source = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

recordBtn.addEventListener('click', startRecording);
stopRecordBtn.addEventListener('click', stopRecording);
