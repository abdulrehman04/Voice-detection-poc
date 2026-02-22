/**
 * AudioWorklet processor for capturing microphone audio.
 * Runs in a separate audio thread for better performance.
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 512; // Match server's expected chunk size
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const channelData = input[0];
        
        // Accumulate samples until we have a full buffer
        for (let i = 0; i < channelData.length; i++) {
            this.buffer[this.bufferIndex++] = channelData[i];
            
            if (this.bufferIndex >= this.bufferSize) {
                // Send the full buffer to main thread
                this.port.postMessage(this.buffer.slice());
                this.bufferIndex = 0;
            }
        }

        return true; // Keep processor alive
    }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);
