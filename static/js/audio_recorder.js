class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.stream = null;
        this.audioContext = null;
        this.analyser = null;
        this.visualizer = null;
    }

    async startRecording() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 16000
                } 
            });

            this.setupVisualization();

            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.start();
            this.isRecording = true;
            this.startVisualization();

            console.log('Recording started');
        } catch (error) {
            console.error('Error starting recording:', error);
            throw new Error('Failed to access microphone. Please check permissions.');
        }
    }

    async stopRecording() {
        if (!this.isRecording || !this.mediaRecorder) {
            console.log('Cannot stop recording: not recording or no media recorder');
            return null;
        }

        if (this.mediaRecorder.state === 'inactive') {
            console.log('MediaRecorder is already inactive');
            return null;
        }

        return new Promise((resolve, reject) => {
            this.isRecording = false;

            const timeout = setTimeout(() => {
                console.error('Recording stop timeout');
                this.cleanup();
                reject(new Error('Recording stop timeout'));
            }, 5000);

            this.mediaRecorder.onstop = () => {
                clearTimeout(timeout);
                this.stopVisualization();
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });

                console.log('Recording stopped, audio blob size:', audioBlob.size);
                this.cleanup();
                resolve(audioBlob);
            };

            try {
                this.mediaRecorder.stop();
            } catch (err) {
                clearTimeout(timeout);
                this.cleanup();
                reject(err);
            }
        });
    }

    setupVisualization() {
        const canvas = document.getElementById('audio-visualizer');
        if (!canvas) return;

        canvas.style.display = 'block';
        this.visualizer = canvas.getContext('2d');

        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.analyser = this.audioContext.createAnalyser();

        const source = this.audioContext.createMediaStreamSource(this.stream);
        source.connect(this.analyser);

        this.analyser.fftSize = 256;
        this.bufferLength = this.analyser.frequencyBinCount;
        this.dataArray = new Uint8Array(this.bufferLength);
    }

    startVisualization() {
        if (!this.visualizer || !this.analyser) return;

        const canvas = document.getElementById('audio-visualizer');
        const canvasCtx = this.visualizer;
        const WIDTH = canvas.width;
        const HEIGHT = canvas.height;

        const draw = () => {
            if (!this.isRecording) return;

            requestAnimationFrame(draw);

            this.analyser.getByteFrequencyData(this.dataArray);

            canvasCtx.fillStyle = 'rgba(40, 44, 52, 0.8)';
            canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);

            const barWidth = (WIDTH / this.bufferLength) * 2.5;
            let barHeight;
            let x = 0;

            for (let i = 0; i < this.bufferLength; i++) {
                barHeight = (this.dataArray[i] / 255) * HEIGHT;

                const r = barHeight + 25 * (i / this.bufferLength);
                const g = 250 * (i / this.bufferLength);
                const b = 50;

                canvasCtx.fillStyle = `rgb(${r},${g},${b})`;
                canvasCtx.fillRect(x, HEIGHT - barHeight, barWidth, barHeight);

                x += barWidth + 1;
            }
        };

        draw();
    }

    stopVisualization() {
        const canvas = document.getElementById('audio-visualizer');
        if (canvas && this.visualizer) {
            this.visualizer.clearRect(0, 0, canvas.width, canvas.height);
            canvas.style.display = 'none';
        }

        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }

    cleanup() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        this.stopVisualization();
        this.mediaRecorder = null;
        this.isRecording = false;
        console.log('Audio recorder cleanup completed');
    }
}