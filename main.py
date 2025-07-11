
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إذاعة صوتية احترافية - Professional Radio Streaming App
تطبيق بث صوتي مع واجهة ويب وتحكم كامل في الصوت
"""

import threading
import socket
import pyaudio
import wave
import time
import json
import base64
import os
import sys
import queue  # إضافة هذا الاستيراد
from datetime import datetime
from collections import deque  # إضافة هذا الاستيراد
from flask import Flask, render_template_string, request, jsonify, Response
from flask_socketio import SocketIO, emit
import numpy as np
from scipy.signal import butter, filtfilt
import noisereduce as nr
import requests

# إعدادات الصوت
# تحديث الإعدادات لتحسين الأداء
CHUNK = 4096  # زيادة حجم البفر
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050  # تقليل معدل العينة لتحسين الأداء
RECORD_SECONDS = 0.1

class AudioProcessor:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.volume = 1.0
        self.noise_reduction = False
        self.muted = False
        self.low_pass_filter = False
        self.high_pass_filter = False
        self.audio_queue = queue.Queue(maxsize=20)  # بفر للصوت
        self.buffer_thread = None
        
    def start_recording(self):
        try:
            self.stream = self.audio.open(format=FORMAT,
                                        channels=CHANNELS,
                                        rate=RATE,
                                        input=True,
                                        frames_per_buffer=CHUNK,
                                        stream_callback=self._audio_callback)
            self.is_recording = True
            self.stream.start_stream()
            return True
        except Exception as e:
            print(f"خطأ في بدء التسجيل: {e}")
            return False

    def process_audio_fast(self, data):
        """معالجة سريعة للصوت"""
        if self.muted:
            return np.zeros_like(data)
        
        # تطبيق مستوى الصوت فقط لتجنب التأخير
        data = data * self.volume
        
        # تطبيع الصوت
        data = np.clip(data, -32767, 32767)
        return data.astype(np.int16) 
    
    def stop_recording(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.is_recording = False
        
        # تنظيف الطابور
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """callback للصوت لتجنب التقطيع"""
        try:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            processed_data = self.process_audio_fast(audio_data)
            
            # إضافة البيانات للطابور
            if not self.audio_queue.full():
                self.audio_queue.put(processed_data.tobytes())
            
        except Exception as e:
            print(f"خطأ في callback: {e}")
        
        return (None, pyaudio.paContinue)

    def process_audio(self, data):
        if self.muted:
            return np.zeros_like(data)
        
        # تطبيق مستوى الصوت
        data = data * self.volume
        
        # تقليل الضوضاء
        if self.noise_reduction:
            try:
                data = nr.reduce_noise(y=data, sr=RATE)
            except:
                pass
        
        # مرشح تمرير منخفض
        if self.low_pass_filter:
            try:
                b, a = butter(4, 3000, btype='low', fs=RATE)
                data = filtfilt(b, a, data)
            except:
                pass
        
        # مرشح تمرير عالي
        if self.high_pass_filter:
            try:
                b, a = butter(4, 300, btype='high', fs=RATE)
                data = filtfilt(b, a, data)
            except:
                pass
        
        # تطبيع الصوت
        data = np.clip(data, -32767, 32767)
        return data.astype(np.int16)
    
    def get_audio_chunk(self):
        try:
            # استخراج البيانات من الطابور
            return self.audio_queue.get(timeout=0.1)
        except queue.Empty:
            return None
        except Exception as e:
            print(f"خطأ في قراءة الصوت: {e}")
            return None

class NetworkManager:
    @staticmethod
    def get_public_ip():
        try:
            response = requests.get('https://api.ipify.org', timeout=5)
            return response.text.strip()
        except:
            return "غير متاح"
    
    @staticmethod
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

# تطبيق Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'radio_streaming_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# متغيرات عامة
audio_processor = AudioProcessor()
network_manager = NetworkManager()
listeners = {}
streaming_active = False
server_stats = {"listeners": 0, "start_time": None, "data_sent": 0}

# صفحة الويب الرئيسية
HTML_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>إذاعة صوتية مباشرة</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Arial', sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            justify-content: center;
            color: white;
        }
        .container { 
            background: rgba(255,255,255,0.1); 
            padding: 30px; 
            border-radius: 20px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            max-width: 800px;
            width: 90%;
        }
        h1 { 
            text-align: center; 
            margin-bottom: 30px; 
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .controls { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; 
            margin-bottom: 30px; 
        }
        .control-group { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .control-group h3 { 
            margin-bottom: 15px; 
            font-size: 1.2em;
            color: #fff;
        }
        button { 
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white; 
            border: none; 
            padding: 15px 25px; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        button.active { 
            background: linear-gradient(45deg, #00b894, #00a085);
        }
        .slider-container { 
            margin: 15px 0; 
        }
        .slider { 
            width: 100%; 
            height: 10px; 
            border-radius: 5px;
            background: rgba(255,255,255,0.3);
            outline: none;
            -webkit-appearance: none;
        }
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 25px;
            height: 25px;
            border-radius: 50%;
            background: #fff;
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .status { 
            text-align: center; 
            margin: 20px 0; 
            font-size: 1.1em;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
        }
        .live-indicator { 
            display: inline-block; 
            width: 12px; 
            height: 12px; 
            background: #ff0000; 
            border-radius: 50%; 
            margin-left: 10px;
            animation: pulse 1s infinite;
        }
        @keyframes pulse { 
            0% { opacity: 1; } 
            50% { opacity: 0.5; } 
            100% { opacity: 1; } 
        }
        .stats { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px; 
            margin-top: 20px;
        }
        .stat-item { 
            background: rgba(255,255,255,0.1); 
            padding: 15px; 
            border-radius: 10px;
            text-align: center;
        }
        .volume-display { 
            font-size: 1.2em; 
            font-weight: bold; 
            color: #ffd700;
        }
        .audio-visualizer { 
            width: 100%; 
            height: 50px; 
            background: rgba(0,0,0,0.2); 
            border-radius: 10px;
            margin: 20px 0;
            position: relative;
            overflow: hidden;
        }
        .wave-bar { 
            position: absolute; 
            bottom: 0; 
            width: 4px; 
            background: linear-gradient(to top, #ff6b6b, #ffd700);
            margin: 0 1px;
            transition: height 0.1s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ إذاعة صوتية مباشرة</h1>
        
        <div class="controls">
            <div class="control-group">
                <h3>🎚️ التحكم الأساسي</h3>
                <button id="startBtn" onclick="toggleStreaming()">بدء البث</button>
                <button id="muteBtn" onclick="toggleMute()">كتم الصوت</button>
            </div>
            
            <div class="control-group">
                <h3>🔊 مستوى الصوت</h3>
                <div class="slider-container">
                    <input type="range" min="0" max="200" value="100" class="slider" id="volumeSlider" oninput="changeVolume(this.value)">
                    <div class="volume-display" id="volumeDisplay">100%</div>
                </div>
            </div>
            
            <div class="control-group">
                <h3>🎛️ فلاتر الصوت</h3>
                <button id="noiseBtn" onclick="toggleNoise()">تقليل الضوضاء</button>
                <button id="lowPassBtn" onclick="toggleLowPass()">مرشح منخفض</button>
                <button id="highPassBtn" onclick="toggleHighPass()">مرشح عالي</button>
            </div>
        </div>
        
        <div class="audio-visualizer" id="visualizer"></div>
        
        <div class="status" id="status">
            <span>حالة البث: متوقف</span>
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <strong>المستمعين</strong><br>
                <span id="listeners">0</span>
            </div>
            <div class="stat-item">
                <strong>وقت البث</strong><br>
                <span id="uptime">00:00:00</span>
            </div>
            <div class="stat-item">
                <strong>البيانات المرسلة</strong><br>
                <span id="dataSent">0 KB</span>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let isStreaming = false;
        let startTime = null;
        let visualizerBars = [];
        
        // إنشاء مصور الصوت
        function createVisualizer() {
            const visualizer = document.getElementById('visualizer');
            for (let i = 0; i < 50; i++) {
                const bar = document.createElement('div');
                bar.className = 'wave-bar';
                bar.style.left = (i * 6) + 'px';
                bar.style.height = '2px';
                visualizer.appendChild(bar);
                visualizerBars.push(bar);
            }
        }
        
        // تحديث مصور الصوت
        function updateVisualizer() {
            visualizerBars.forEach((bar, index) => {
                const height = Math.random() * 40 + 2;
                bar.style.height = height + 'px';
            });
        }
        
        // تبديل البث
        function toggleStreaming() {
            const btn = document.getElementById('startBtn');
            if (!isStreaming) {
                socket.emit('start_stream');
                btn.textContent = 'إيقاف البث';
                btn.classList.add('active');
                isStreaming = true;
                startTime = Date.now();
                updateStatus('البث مُفعل', true);
                setInterval(updateVisualizer, 100);
            } else {
                socket.emit('stop_stream');
                btn.textContent = 'بدء البث';
                btn.classList.remove('active');
                isStreaming = false;
                updateStatus('البث متوقف', false);
            }
        }
        
        // تبديل الكتم
        function toggleMute() {
            const btn = document.getElementById('muteBtn');
            socket.emit('toggle_mute');
            btn.classList.toggle('active');
        }
        
        // تغيير مستوى الصوت
        function changeVolume(value) {
            const display = document.getElementById('volumeDisplay');
            display.textContent = value + '%';
            socket.emit('change_volume', {volume: value / 100});
        }
        
        // تبديل تقليل الضوضاء
        function toggleNoise() {
            const btn = document.getElementById('noiseBtn');
            socket.emit('toggle_noise');
            btn.classList.toggle('active');
        }
        
        // تبديل المرشح المنخفض
        function toggleLowPass() {
            const btn = document.getElementById('lowPassBtn');
            socket.emit('toggle_low_pass');
            btn.classList.toggle('active');
        }
        
        // تبديل المرشح العالي
        function toggleHighPass() {
            const btn = document.getElementById('highPassBtn');
            socket.emit('toggle_high_pass');
            btn.classList.toggle('active');
        }
        
        // تحديث الحالة
        function updateStatus(message, isLive) {
            const status = document.getElementById('status');
            if (isLive) {
                status.innerHTML = `<span>حالة البث: ${message}</span><span class="live-indicator"></span>`;
            } else {
                status.innerHTML = `<span>حالة البث: ${message}</span>`;
            }
        }
        
        // تحديث الوقت
        function updateUptime() {
            if (startTime) {
                const elapsed = Date.now() - startTime;
                const hours = Math.floor(elapsed / 3600000);
                const minutes = Math.floor((elapsed % 3600000) / 60000);
                const seconds = Math.floor((elapsed % 60000) / 1000);
                document.getElementById('uptime').textContent = 
                    `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
        }
        
        // استقبال الأحداث
        socket.on('stats_update', function(data) {
            document.getElementById('listeners').textContent = data.listeners;
            document.getElementById('dataSent').textContent = Math.round(data.data_sent / 1024) + ' KB';
        });
        
        socket.on('stream_started', function() {
            updateStatus('البث مُفعل', true);
        });
        
        socket.on('stream_stopped', function() {
            updateStatus('البث متوقف', false);
        });
        
        // تهيئة الصفحة
        document.addEventListener('DOMContentLoaded', function() {
            createVisualizer();
            setInterval(updateUptime, 1000);
        });
    </script>
</body>
</html>
"""

# صفحة الاستماع
LISTEN_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>الاستماع للإذاعة</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Arial', sans-serif; 
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            min-height: 100vh; 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            justify-content: center;
            color: white;
        }
        .player { 
            background: rgba(255,255,255,0.1); 
            padding: 40px; 
            border-radius: 20px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            text-align: center;
            max-width: 600px;
            width: 90%;
        }
        h1 { 
            margin-bottom: 30px; 
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .play-button { 
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            color: white; 
            border: none; 
            padding: 20px 40px; 
            border-radius: 50px; 
            cursor: pointer; 
            font-size: 20px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            margin: 20px 0;
        }
        .play-button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .play-button.playing { 
            background: linear-gradient(45deg, #27ae60, #2ecc71);
        }
        .volume-control { 
            margin: 30px 0; 
        }
        .volume-slider { 
            width: 200px; 
            margin: 0 10px;
        }
        .status { 
            margin: 20px 0; 
            font-size: 1.2em;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
        }
        .equalizer { 
            display: flex; 
            justify-content: center; 
            align-items: end; 
            height: 60px; 
            margin: 30px 0;
        }
        .eq-bar { 
            width: 8px; 
            margin: 0 2px; 
            background: linear-gradient(to top, #3498db, #2ecc71);
            border-radius: 4px;
            animation: bounce 0.5s infinite alternate;
        }
        @keyframes bounce { 
            from { height: 10px; } 
            to { height: 50px; } 
        }
    </style>
</head>
<body>
    <div class="player">
        <h1>🎵 الاستماع للإذاعة</h1>
        
        <button class="play-button" id="playBtn" onclick="togglePlay()">
            ▶️ تشغيل
        </button>
        
        <div class="volume-control">
            <span>🔊 مستوى الصوت:</span>
            <input type="range" min="0" max="100" value="50" class="volume-slider" id="volumeSlider" oninput="changeVolume(this.value)">
            <span id="volumeValue">50%</span>
        </div>
        
        <div class="equalizer" id="equalizer" style="display: none;">
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
        </div>
        
        <div class="status" id="status">
            حالة الاتصال: متصل
        </div>
    </div>

    <script>
        const socket = io();
        let audioContext;
        let audioBuffer = [];
        let isPlaying = false;
        let currentSource;
        let gainNode;
        let nextTime = 0;
        
        function initAudio() {
            try {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                gainNode = audioContext.createGain();
                gainNode.connect(audioContext.destination);
                gainNode.gain.value = 0.5;
                nextTime = audioContext.currentTime;
                return true;
            } catch (e) {
                console.error('خطأ في تهيئة الصوت:', e);
                return false;
            }
        }
        
        function togglePlay() {
            const btn = document.getElementById('playBtn');
            const equalizer = document.getElementById('equalizer');
            
            if (!isPlaying) {
                if (initAudio()) {
                    btn.textContent = '⏸️ إيقاف';
                    btn.classList.add('playing');
                    isPlaying = true;
                    equalizer.style.display = 'flex';
                    socket.emit('join_listeners');
                    updateStatus('جاري الاستماع...');
                }
            } else {
                btn.textContent = '▶️ تشغيل';
                btn.classList.remove('playing');
                isPlaying = false;
                equalizer.style.display = 'none';
                if (currentSource) {
                    currentSource.stop();
                }
                socket.emit('leave_listeners');
                updateStatus('متوقف');
            }
        }
        
        function changeVolume(value) {
            document.getElementById('volumeValue').textContent = value + '%';
            if (gainNode) {
                gainNode.gain.value = value / 100;
            }
        }
        
        function updateStatus(message) {
            document.getElementById('status').textContent = `حالة الاتصال: ${message}`;
        }
        
        socket.on('audio_data', function(data) {
            if (isPlaying && audioContext) {
                try {
                    // تحويل base64 إلى ArrayBuffer
                    const binaryString = atob(data);
                    const arrayBuffer = new ArrayBuffer(binaryString.length);
                    const uint8Array = new Uint8Array(arrayBuffer);
                    
                    for (let i = 0; i < binaryString.length; i++) {
                        uint8Array[i] = binaryString.charCodeAt(i);
                    }
                    
                    // تحويل إلى Float32Array
                    const int16Array = new Int16Array(arrayBuffer);
                    const float32Array = new Float32Array(int16Array.length);
                    
                    for (let i = 0; i < int16Array.length; i++) {
                        float32Array[i] = int16Array[i] / 32768.0;
                    }
                    
                    // إنشاء AudioBuffer
                    const audioBuffer = audioContext.createBuffer(1, float32Array.length, 22050);
                    audioBuffer.getChannelData(0).set(float32Array);
                    
                    // تشغيل الصوت بتوقيت محدد
                    const source = audioContext.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(gainNode);
                    
                    const playTime = Math.max(nextTime, audioContext.currentTime);
                    source.start(playTime);
                    nextTime = playTime + audioBuffer.duration;
                    
                } catch (e) {
                    console.error('خطأ في تشغيل الصوت:', e);
                }
            }
        });
        
        socket.on('stream_status', function(data) {
            if (data.active) {
                updateStatus('البث مُفعل');
            } else {
                updateStatus('البث متوقف');
            }
        });
        
        socket.on('disconnect', function() {
            updateStatus('انقطع الاتصال');
        });
        
        socket.on('connect', function() {
            updateStatus('متصل');
        });
    </script>
</body>
</html>
"""

# المسارات
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/listen')
def listen():
    return render_template_string(LISTEN_TEMPLATE)

@app.route('/status')
def status():
    public_ip = network_manager.get_public_ip()
    local_ip = network_manager.get_local_ip()
    current_time = time.time()
    uptime = int(current_time - server_stats.get('start_time', current_time)) if server_stats.get('start_time') else 0
    
    return jsonify({
        'public_ip': public_ip,
        'local_ip': local_ip,
        'streaming_active': streaming_active,
        'listeners': len(listeners),
        'uptime_seconds': uptime,
        'data_sent_mb': round(server_stats['data_sent'] / (1024*1024), 2)
    })

# أحداث WebSocket
@socketio.on('connect')
def handle_connect():
    print(f"عميل جديد متصل: {request.sid}")
    emit('stream_status', {'active': streaming_active})

@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in listeners:
        del listeners[request.sid]
    print(f"عميل منقطع: {request.sid}")

@socketio.on('start_stream')
def handle_start_stream():
    global streaming_active
    if audio_processor.start_recording():
        streaming_active = True
        server_stats['start_time'] = time.time()
        emit('stream_started', broadcast=True)
        print("تم بدء البث")
    else:
        emit('error', {'message': 'فشل في بدء البث'})

@socketio.on('stop_stream')
def handle_stop_stream():
    global streaming_active
    audio_processor.stop_recording()
    streaming_active = False
    emit('stream_stopped', broadcast=True)
    print("تم إيقاف البث")

@socketio.on('toggle_mute')
def handle_toggle_mute():
    audio_processor.muted = not audio_processor.muted
    print(f"كتم الصوت: {'مُفعل' if audio_processor.muted else 'معطل'}")

@socketio.on('change_volume')
def handle_change_volume(data):
    audio_processor.volume = data['volume']
    print(f"تغيير مستوى الصوت إلى: {data['volume']}")

@socketio.on('toggle_noise')
def handle_toggle_noise():
    audio_processor.noise_reduction = not audio_processor.noise_reduction
    print(f"تقليل الضوضاء: {'مُفعل' if audio_processor.noise_reduction else 'معطل'}")

@socketio.on('toggle_low_pass')
def handle_toggle_low_pass():
    audio_processor.low_pass_filter = not audio_processor.low_pass_filter
    print(f"المرشح المنخفض: {'مُفعل' if audio_processor.low_pass_filter else 'معطل'}")

@socketio.on('toggle_high_pass')
def handle_toggle_high_pass():
    audio_processor.high_pass_filter = not audio_processor.high_pass_filter
    print(f"المرشح العالي: {'مُفعل' if audio_processor.high_pass_filter else 'معطل'}")

@socketio.on('join_listeners')
def handle_join_listeners():
    listeners[request.sid] = {
        'joined_at': time.time(),
        'buffer': deque(maxlen=10)
    }
    print(f"مستمع جديد: {request.sid}")
    
    # إرسال إشارة بدء التشغيل
    emit('stream_ready', {'sample_rate': RATE, 'channels': CHANNELS})

@socketio.on('leave_listeners')
def handle_leave_listeners():
    if request.sid in listeners:
        del listeners[request.sid]
    print(f"مستمع غادر: {request.sid}")

# تحديث دالة البث لتكون أكثر كفاءة
def audio_streaming_thread():
    """خيط بث الصوت محسن"""
    last_stats_update = 0
    audio_buffer = deque(maxlen=5)  # بفر إضافي
    
    while True:
        if streaming_active and audio_processor.is_recording:
            try:
                audio_data = audio_processor.get_audio_chunk()
                if audio_data:
                    # إضافة البيانات للبفر
                    audio_buffer.append(audio_data)
                    
                    # إرسال البيانات إذا كان هناك مستمعين
                    if len(listeners) > 0 and audio_buffer:
                        # أخذ البيانات من البفر
                        data_to_send = audio_buffer.popleft()
                        
                        # تحويل البيانات إلى base64
                        audio_b64 = base64.b64encode(data_to_send).decode('utf-8')
                        
                        # إرسال البيانات لجميع المستمعين دفعة واحدة
                        if listeners:
                            socketio.emit('audio_data', audio_b64, room=None)
                        
                        # تحديث الإحصائيات
                        server_stats['data_sent'] += len(data_to_send)
                        server_stats['listeners'] = len(listeners)
                        
                        # إرسال الإحصائيات كل 5 ثوان
                        current_time = time.time()
                        if current_time - last_stats_update >= 5:
                            stats_to_send = {
                                'listeners': server_stats['listeners'],
                                'data_sent': server_stats['data_sent'],
                                'uptime': int(current_time - server_stats.get('start_time', current_time)) if server_stats.get('start_time') else 0
                            }
                            socketio.emit('stats_update', stats_to_send)
                            last_stats_update = current_time
                            
            except Exception as e:
                print(f"خطأ في خيط البث: {e}")
        
        time.sleep(0.02)  # تقليل التأخير

def print_server_info():
    """طباعة معلومات الخادم"""
    print("\n" + "="*60)
    print("🎙️  إذاعة صوتية احترافية - Professional Radio Streaming")
    print("="*60)
    
    local_ip = network_manager.get_local_ip()
    public_ip = network_manager.get_public_ip()
    port = 5000
    
    print(f"📡 الخادم المحلي: http://{local_ip}:{port}")
    print(f"🌐 الخادم العام: http://{public_ip}:{port}")
    print(f"🎧 رابط الاستماع المحلي: http://{local_ip}:{port}/listen")
    print(f"🎧 رابط الاستماع العام: http://{public_ip}:{port}/listen")
    print(f"📊 معلومات الخادم: http://{local_ip}:{port}/status")
    print("\n💡 نصائح:")
    print("   - استخدم الرابط العام للوصول من أي مكان")
    print("   - تأكد من فتح المنفذ 5000 في جدار الحماية")
    print("   - للأفضل أداء، استخدم سماعات أذن لتجنب الصدى")
    print("="*60)

def setup_audio_requirements():
    """التحقق من متطلبات الصوت"""
    try:
        import pyaudio
        import numpy as np
        import scipy.signal
        import noisereduce as nr
        print("✅ جميع مكتبات الصوت متوفرة")
        return True
    except ImportError as e:
        print(f"❌ مكتبة مفقودة: {e}")
        print("💡 قم بتثبيت المكتبات المطلوبة:")
        print("   pip install pyaudio numpy scipy noisereduce")
        return False

def main():
    """الدالة الرئيسية"""
    print("🚀 بدء تشغيل الخادم...")
    
    # التحقق من المتطلبات
    if not setup_audio_requirements():
        print("❌ فشل في التحقق من المتطلبات")
        return
    
    # طباعة معلومات الخادم
    print_server_info()
    
    # بدء خيط البث الصوتي
    audio_thread = threading.Thread(target=audio_streaming_thread, daemon=True)
    audio_thread.start()
    print("🎵 تم بدء خيط البث الصوتي")
    
    try:
        # تشغيل الخادم
        print("🌐 تشغيل خادم الويب...")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n🛑 إيقاف الخادم...")
        audio_processor.stop_recording()
        print("✅ تم إيقاف الخادم بنجاح")
    except Exception as e:
        print(f"❌ خطأ في تشغيل الخادم: {e}")
        print("💡 تأكد من أن المنفذ 5000 غير مستخدم")

if __name__ == "__main__":
    main()