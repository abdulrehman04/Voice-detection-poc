// Set to true when testing from phone via cloudflare tunnel
const USE_TUNNEL = false;
const WS_TUNNEL_URL = "wss://YOUR-WEBSOCKET-TUNNEL.trycloudflare.com";

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const profileIdInput = document.getElementById('profileId');
const vadOnlyCheckbox = document.getElementById('vadOnlyMode');
const stateDisplay = document.getElementById('stateDisplay');
const simValue = document.getElementById('simValue');
const vadValue = document.getElementById('vadValue');
const timeoutBar = document.getElementById('timeoutBar');
const timeoutValue = document.getElementById('timeoutValue');
const connectionStatus = document.getElementById('connectionStatus');
const segmentsList = document.getElementById('segmentsList');
const segmentCount = document.getElementById('segmentCount');
const profilePanel = document.getElementById('profilePanel');
const profileName = document.getElementById('profileName');
const profileSessions = document.getElementById('profileSessions');
const adaptiveProgress = document.getElementById('adaptiveProgress');
const sessionThresholds = document.getElementById('sessionThresholds');
const ctx = document.getElementById('simChart').getContext('2d');

let audioContext;
let workletNode;
let source;
let ws;
let isRunning = false;
let isVadOnlyMode = false;
let currentProfileInfo = null; // Track profile sessions
const clearProfileBtn = document.getElementById('clearProfileBtn');

const CHART_POINTS = 80;
const chartCanvas = document.getElementById('simChart');
const chartContainer = chartCanvas.parentElement;

const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: Array(CHART_POINTS).fill(''),
        datasets: [{
            label: 'Similarity',
            data: Array(CHART_POINTS).fill(0),
            borderColor: 'rgb(75, 192, 192)',
            borderWidth: 2,
            tension: 0.1,
            yAxisID: 'y',
            pointRadius: 0
        }, {
            label: 'VAD Prob',
            data: Array(CHART_POINTS).fill(0),
            borderColor: 'rgb(255, 99, 132)',
            borderWidth: 1,
            tension: 0.1,
            yAxisID: 'y',
            pointRadius: 0
        }, {
            label: 'State (USER)',
            data: Array(CHART_POINTS).fill(0),
            backgroundColor: 'rgba(0, 255, 0, 0.2)',
            borderColor: 'rgba(0, 255, 0, 0.5)',
            borderWidth: 0,
            fill: true,
            tension: 0,
            yAxisID: 'y',
            pointRadius: 0
        }]
    },
    options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                min: 0,
                max: 1
            }
        },
        plugins: {
            legend: {
                labels: {
                    filter: (item) => item.text !== 'State (USER)'
                }
            }
        }
    }
});

const stateHistory = Array(CHART_POINTS).fill(0);

function updateTimeoutBar(timeout, min, max) {
    if (!timeoutBar || !timeoutValue) return;
    
    const range = max - min;
    const position = range > 0 ? ((timeout - min) / range) * 100 : 50;
    const clampedPosition = Math.max(0, Math.min(100, position));
    
    timeoutBar.style.setProperty('--timeout-position', `${clampedPosition}%`);
    timeoutValue.textContent = `${Math.round(timeout * 1000)}ms`;
    timeoutValue.title = `Range: ${Math.round(min * 1000)}ms - ${Math.round(max * 1000)}ms`;
}

function updateChart(sim, vad, state) {
    // Update state history (1 = USER, 0 = other)
    const stateVal = (state === 'USER' || state === 'TRAILING') ? 1 : 0;
    stateHistory.push(stateVal);
    stateHistory.shift();
    
    const simData = chart.data.datasets[0].data;
    const vadData = chart.data.datasets[1].data;
    const stateData = chart.data.datasets[2].data;
    const labels = chart.data.labels;
    
    simData.push(sim);
    vadData.push(vad);
    stateData.push(stateVal);
    labels.push('');
    
    while (simData.length > CHART_POINTS) simData.shift();
    while (vadData.length > CHART_POINTS) vadData.shift();
    while (stateData.length > CHART_POINTS) stateData.shift();
    while (labels.length > CHART_POINTS) labels.shift();
    
    chart.update('none'); // 'none' mode for faster updates
}

function playAudioBuffer(float32Array) {
    if (!audioContext) return;
    
    const buffer = audioContext.createBuffer(1, float32Array.length, 16000);
    buffer.copyToChannel(float32Array, 0);
    
    const bufferSource = audioContext.createBufferSource();
    bufferSource.buffer = buffer;
    bufferSource.connect(audioContext.destination);
    bufferSource.start(0);
}

function updateProfilePanel(info) {
    currentProfileInfo = info;
    if (!profilePanel) return;
    profilePanel.style.display = '';
    profileName.textContent = info.profile_id;

    if (info.sessions === 0) {
        profileSessions.textContent = 'No sessions';
        profileSessions.style.background = '#ef5350';
        sessionThresholds.innerHTML = '<span style="font-size:0.8em;color:#999;">Profile cleared. Go to Enrollment to re-enroll.</span>';
        if (clearProfileBtn) clearProfileBtn.style.display = 'none';
        return;
    }

    profileSessions.textContent = `${info.sessions} session${info.sessions !== 1 ? 's' : ''}`;
    profileSessions.style.background = '#5c6bc0';
    if (clearProfileBtn) clearProfileBtn.style.display = '';

    // Show per-session thresholds
    sessionThresholds.innerHTML = info.session_thresholds
        .map((t, i) => `<span class="session-tag" id="session-tag-${i+1}">S${i+1}: ${t.toFixed(3)}</span>`)
        .join('');
}

function clearProfile() {
    const profileId = profileIdInput.value;
    if (!profileId) return;
    if (!confirm(`Clear all enrollment sessions for "${profileId}"?`)) return;
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "clear_profile", profile_id: profileId }));
    }
}

function highlightSession(sessionIdx) {
    // Highlight the matched session tag briefly
    document.querySelectorAll('.session-tag').forEach(el => el.classList.remove('active'));
    const tag = document.getElementById(`session-tag-${sessionIdx}`);
    if (tag) tag.classList.add('active');
}

let pendingSegmentInfo = null;
let segmentCounter = 0;

function updateSegmentCount() {
    if (segmentCount) {
        segmentCount.textContent = `${segmentCounter} segment${segmentCounter !== 1 ? 's' : ''}`;
    }
}

function formatTime(ms) {
    if (!ms) return '--:--:--';
    const date = new Date(ms);
    const h = String(date.getHours()).padStart(2, '0');
    const m = String(date.getMinutes()).padStart(2, '0');
    const s = String(date.getSeconds()).padStart(2, '0');
    const ms3 = String(date.getMilliseconds()).padStart(3, '0');
    return `${h}:${m}:${s}.${ms3}`;
}

function formatDelta(ms) {
    if (ms === null || ms === undefined) return '';
    if (ms === 0) return '(+0ms)';
    return `(+${ms}ms)`;
}

function addSegmentToList(segmentInfo, audioData, autoplay = true) {
    segmentCounter++;
    updateSegmentCount();
    
    const emptyState = segmentsList.querySelector('.empty-state');
    if (emptyState) emptyState.remove();
    
    const wavBlob = encodeWAV(audioData, 16000);
    const url = URL.createObjectURL(wavBlob);
    
    const duration = segmentInfo.duration;
    const chunkSize = segmentInfo.chunk_size_kb || (audioData.length * 4 / 1024).toFixed(1);
    
    const vadStart = segmentInfo.vad_started_at || 0;
    const userConfirmed = segmentInfo.user_confirmed_at || 0;
    const speechEnded = segmentInfo.speech_ended_at || 0;
    const ready = segmentInfo.timestamp || 0;
    
    const idTime = (vadStart && userConfirmed) ? userConfirmed - vadStart : null;
    const speechDelta = (vadStart && speechEnded) ? speechEnded - vadStart : null;
    const delay = (speechEnded && ready) ? ready - speechEnded : null;
    
    const userIdLabel = isVadOnlyMode ? 'Confirmed' : 'User ID\'d';
    const modeBadge = isVadOnlyMode ? '<span class="mode-badge vad-only">VAD</span>' : '';

    const bestSession = segmentInfo.best_session || 0;
    const totalSessions = segmentInfo.total_sessions || 0;
    const similarity = segmentInfo.similarity || 0;
    const thresholdUsed = segmentInfo.threshold_used || 0;
    const simIsGood = similarity > thresholdUsed;

    if (bestSession > 0) highlightSession(bestSession);

    const simSection = (!isVadOnlyMode && bestSession > 0) ? `
        <div class="segment-similarity">
            <div class="sim-item">
                <span class="sim-label">Match:</span>
                <span class="sim-value">S${bestSession}/${totalSessions}</span>
            </div>
            <div class="sim-item">
                <span class="sim-label">Sim:</span>
                <span class="sim-value ${simIsGood ? 'good' : ''}">${similarity.toFixed(3)}</span>
            </div>
            <div class="sim-item">
                <span class="sim-label">Thr:</span>
                <span class="sim-value">${thresholdUsed.toFixed(3)}</span>
            </div>
        </div>
    ` : '';

    const item = document.createElement('div');
    item.className = 'segment-item' + (isVadOnlyMode ? ' vad-only-segment' : '');
    item.innerHTML = `
        <div class="segment-header">
            <span class="segment-number">#${segmentCounter} ${modeBadge}</span>
            <span class="segment-badge">${duration.toFixed(2)}s · ${chunkSize} KB</span>
        </div>
        ${simSection}
        <div class="segment-timeline">
            <div class="timeline-row">
                <span class="event-name">VAD Start</span>
                <span class="event-time">${formatTime(vadStart)}</span>
                <span class="event-delta"></span>
            </div>
            <div class="timeline-row ${idTime !== null && idTime < 1000 ? 'fast' : ''}">
                <span class="event-name">${userIdLabel}</span>
                <span class="event-time">${formatTime(userConfirmed)}</span>
                <span class="event-delta">${idTime !== null ? formatDelta(idTime) : '--'}</span>
            </div>
            <div class="timeline-row">
                <span class="event-name">Speech End</span>
                <span class="event-time">${formatTime(speechEnded)}</span>
                <span class="event-delta">${speechDelta !== null ? formatDelta(speechDelta) : '--'}</span>
            </div>
            <div class="timeline-row ${delay !== null && delay < 100 ? 'fast' : 'slow'}">
                <span class="event-name">Ready</span>
                <span class="event-time">${formatTime(ready)}</span>
                <span class="event-delta delay">${delay !== null ? `+${delay}ms` : '--'}</span>
            </div>
        </div>
        <div class="segment-audio">
            <audio controls src="${url}"></audio>
        </div>
    `;
    
    segmentsList.insertBefore(item, segmentsList.firstChild);
    
    if (autoplay) {
        playAudioBuffer(audioData);
    }
}

function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (view, offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    const floatTo16BitPCM = (output, offset, input) => {
        for (let i = 0; i < input.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, input[i]));
            output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
    };

    floatTo16BitPCM(view, 44, samples);

    return new Blob([view], { type: 'audio/wav' });
}

async function startSession() {
    const profileId = profileIdInput.value;
    isVadOnlyMode = vadOnlyCheckbox?.checked || false;
    
    if (!profileId && !isVadOnlyMode) {
        alert("Please enter a Profile ID or enable VAD Only mode");
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

        const wsUrl = USE_TUNNEL
            ? WS_TUNNEL_URL 
            : `ws://${window.location.hostname || 'localhost'}:8765`;
        ws = new WebSocket(wsUrl);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            connectionStatus.textContent = "Connected";
            connectionStatus.style.color = "green";
            ws.send(JSON.stringify({
                type: "hello",
                client_id: "web_client",
                profile_id: profileId,
                vad_only: isVadOnlyMode
            }));
            
            pendingSegmentInfo = null;
            segmentCounter = 0;
            updateSegmentCount();
            segmentsList.innerHTML = `
                <div class="empty-state">
                    <span>No segments yet</span>
                    <small>Start speaking to see detected segments</small>
                </div>
            `;
            stateHistory.fill(0);
        };

        ws.onmessage = async (event) => {
            if (typeof event.data === 'string') {
                const data = JSON.parse(event.data);
                
                if (data.type === 'stats') {
                    if (data.vad_only !== undefined) {
                        isVadOnlyMode = data.vad_only;
                    }
                    
                    if (isVadOnlyMode) {
                        simValue.textContent = 'N/A';
                        simValue.title = 'VAD-only mode - no speaker verification';
                    } else {
                        simValue.textContent = data.similarity.toFixed(2);
                        simValue.title = '';
                    }
                    vadValue.textContent = data.vad_prob.toFixed(2);
                    
                    if (data.timeout !== undefined) {
                        updateTimeoutBar(data.timeout, data.timeout_min, data.timeout_max);
                    }
                    
                    const displayState = isVadOnlyMode && data.state === 'USER' ? 'SPEECH' : data.state;
                    stateDisplay.textContent = displayState;
                    switch (data.state) {
                        case 'USER':
                            stateDisplay.className = isVadOnlyMode 
                                ? 'state-indicator state-vad-speech' 
                                : 'state-indicator state-user';
                            break;
                        case 'PENDING':
                            stateDisplay.className = 'state-indicator state-pending';
                            break;
                        case 'TRAILING':
                            stateDisplay.className = 'state-indicator state-trailing';
                            break;
                        default:
                            stateDisplay.className = 'state-indicator state-unknown';
                    }
                    
                    const simForChart = isVadOnlyMode ? 0 : data.similarity;
                    updateChart(simForChart, data.vad_prob, data.state);
                }
                else if (data.type === 'profile_info') {
                    updateProfilePanel(data);
                }
                else if (data.type === 'profile_update') {
                    updateProfilePanel(data);
                    const newTag = document.getElementById(`session-tag-${data.sessions}`);
                    if (newTag) newTag.classList.add('new');
                    console.log(`Adaptive enrollment: now ${data.sessions} sessions`);
                }
                else if (data.type === 'segment_complete') {
                    pendingSegmentInfo = {
                        timestamp: data.timestamp,
                        duration: data.duration,
                        samples: data.samples,
                        vad_started_at: data.vad_started_at,
                        user_confirmed_at: data.user_confirmed_at,
                        speech_ended_at: data.speech_ended_at,
                        processing_delay_ms: data.processing_delay_ms,
                        chunk_size_kb: data.chunk_size_kb,
                        best_session: data.best_session,
                        total_sessions: data.total_sessions,
                        similarity: data.similarity,
                        threshold_used: data.threshold_used,
                    };
                    console.log(`Segment complete: ${data.duration.toFixed(2)}s, session=${data.best_session}/${data.total_sessions}, sim=${data.similarity?.toFixed(3)}, thr=${data.threshold_used?.toFixed(3)}`);
                }
            } else {
                const float32 = new Float32Array(event.data);
                if (pendingSegmentInfo && float32.length === pendingSegmentInfo.samples) {
                    addSegmentToList(pendingSegmentInfo, float32, true);
                    pendingSegmentInfo = null;
                }
            }
        };


        ws.onclose = () => {
            connectionStatus.textContent = "Disconnected";
            connectionStatus.style.color = "red";
            stopSession();
        };

        source = audioContext.createMediaStreamSource(stream);

        await audioContext.audioWorklet.addModule('audio-processor.js');
        workletNode = new AudioWorkletNode(audioContext, 'audio-capture-processor');
        
        workletNode.port.onmessage = (event) => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(event.data);
            }
        };

        source.connect(workletNode);
        isRunning = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        profileIdInput.disabled = true;

    } catch (err) {
        console.error("Error starting session:", err);
        alert("Could not start audio session: " + err.message);
    }
}

function stopSession() {
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
    if (ws) {
        ws.close();
        ws = null;
    }

    isRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    profileIdInput.disabled = false;
    connectionStatus.textContent = "Disconnected";
}

startBtn.addEventListener('click', startSession);
stopBtn.addEventListener('click', stopSession);
if (clearProfileBtn) clearProfileBtn.addEventListener('click', clearProfile);

// Check for profile_id in URL
const urlParams = new URLSearchParams(window.location.search);
const pid = urlParams.get('profile_id');
if (pid) {
    profileIdInput.value = pid;
}
