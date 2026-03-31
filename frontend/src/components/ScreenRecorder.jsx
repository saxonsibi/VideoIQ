import React, { useEffect, useState, useRef } from 'react';
import { EyeIcon, EyeSlashIcon } from '@heroicons/react/24/outline'
import { videoAPI } from '../services/api'

const getPreferredRecordingMimeType = () => {
  if (typeof MediaRecorder === 'undefined' || typeof MediaRecorder.isTypeSupported !== 'function') {
    return ''
  }

  const candidates = [
    'video/webm;codecs=vp8,opus',
    'video/webm;codecs=vp9,opus',
    'video/webm',
  ]
  return candidates.find((mimeType) => MediaRecorder.isTypeSupported(mimeType)) || ''
}

const getRecordingExtension = (mimeType) => {
  const normalized = String(mimeType || '').toLowerCase()
  if (normalized.includes('mp4')) return 'mp4'
  return 'webm'
}

const ScreenRecorder = ({ onVideoRecorded }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [showPreview, setShowPreview] = useState(true)
  const [hasActivePreview, setHasActivePreview] = useState(false)
  const [recordedBlob, setRecordedBlob] = useState(null)
  const [recordedPreviewUrl, setRecordedPreviewUrl] = useState(null)
  const [recordedMimeType, setRecordedMimeType] = useState('video/webm')
  
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const videoPreviewRef = useRef(null);
  const activeStreamRef = useRef(null)

  useEffect(() => () => {
    if (recordedPreviewUrl) {
      URL.revokeObjectURL(recordedPreviewUrl)
    }
  }, [recordedPreviewUrl])

  const getScreenCaptureStream = async () => {
    const baseVideoConstraints = {
      displaySurface: 'monitor',
    }

    try {
      return await navigator.mediaDevices.getDisplayMedia({
        video: baseVideoConstraints,
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
          sampleRate: 48000,
        },
      })
    } catch (primaryError) {
      console.warn('Screen capture with audio failed, retrying without audio:', primaryError)
      return navigator.mediaDevices.getDisplayMedia({
        video: baseVideoConstraints,
        audio: false,
      })
    }
  }

  const startRecording = async () => {
    try {
      setError(null);
      chunksRef.current = [];
      if (recordedPreviewUrl) {
        URL.revokeObjectURL(recordedPreviewUrl)
      }
      setRecordedBlob(null)
      setRecordedPreviewUrl(null)
      setRecordedMimeType('video/webm')
      
      const stream = await getScreenCaptureStream()

      activeStreamRef.current = stream
      setHasActivePreview(true)

      if (videoPreviewRef.current) {
        videoPreviewRef.current.srcObject = stream
        videoPreviewRef.current.play().catch(() => {})
      }

      // Create MediaRecorder
      const preferredMimeType = getPreferredRecordingMimeType()
      const mediaRecorder = preferredMimeType
        ? new MediaRecorder(stream, { mimeType: preferredMimeType })
        : new MediaRecorder(stream);

      mediaRecorderRef.current = mediaRecorder;

      // Handle data available
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      // Handle recording stop
      mediaRecorder.onstop = async () => {
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
        activeStreamRef.current = null
        setHasActivePreview(false)
        
        // Hide preview
        if (videoPreviewRef.current) {
          videoPreviewRef.current.srcObject = null;
        }
        
        // Create video blob
        const finalMimeType = mediaRecorder.mimeType || preferredMimeType || 'video/webm'
        const blob = new Blob(chunksRef.current, { type: finalMimeType });
        setRecordedBlob(blob)
        setRecordedMimeType(finalMimeType)
        setRecordedPreviewUrl(URL.createObjectURL(blob))
      };

      // Start recording
      mediaRecorder.start(1000);
      setIsRecording(true);
      setRecordingTime(0);

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

      // Handle stream ended
      stream.getVideoTracks()[0].onended = () => {
        stopRecording();
      };

    } catch (err) {
      console.error('Error starting recording:', err);
      setError('Failed to start screen recording. Please grant screen capture access and try again.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const uploadRecording = async (blob) => {
    setUploading(true);
    try {
      const mimeType = blob?.type || recordedMimeType || 'video/webm'
      const extension = getRecordingExtension(mimeType)
      const file = new File([blob], `screen_recording_${Date.now()}.${extension}`, {
        type: mimeType
      });

      const formData = new FormData();
      formData.append('file', file);
      formData.append('title', `Screen Recording ${new Date().toLocaleString()}`);
      formData.append('description', 'Recorded from screen capture');

      const response = await videoAPI.upload(formData)
      const data = response.data
      
      if (onVideoRecorded) {
        onVideoRecorded(data);
      }

      setRecordingTime(0);
      if (recordedPreviewUrl) {
        URL.revokeObjectURL(recordedPreviewUrl)
      }
      setRecordedBlob(null)
      setRecordedPreviewUrl(null)
      
    } catch (err) {
      console.error('Error uploading recording:', err);
      setError(`Failed to upload recording: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
      <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
        <svg 
          className="w-5 h-5" 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth={2} 
            d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" 
          />
        </svg>
        Screen Recorder
      </h3>

      {error && (
        <div className="bg-red-500/20 border border-red-500 rounded-lg p-3 mb-4 text-red-200">
          {error}
        </div>
      )}

      <div className="space-y-4">
        <div className="flex justify-end">
          <button
            type="button"
            onClick={() => setShowPreview((prev) => !prev)}
            className="px-3 py-1.5 rounded-lg bg-gray-700 text-gray-200 hover:bg-gray-600 transition-colors flex items-center gap-2"
          >
            {showPreview ? <EyeSlashIcon className="w-4 h-4" /> : <EyeIcon className="w-4 h-4" />}
            {showPreview ? 'Hide Preview' : 'Show Preview'}
          </button>
        </div>

        {showPreview && (
          <div className="relative aspect-video bg-gray-900 rounded-lg overflow-hidden mb-4 border border-gray-700">
            {recordedPreviewUrl ? (
              <video
                src={recordedPreviewUrl}
                className="w-full h-full object-contain bg-black"
                controls
                playsInline
                preload="metadata"
              />
            ) : (
              <video 
                ref={videoPreviewRef}
                className="w-full h-full object-contain bg-black"
                muted
                playsInline
                autoPlay
              />
            )}
            {!hasActivePreview && !recordedPreviewUrl && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-900/70">
                <span className="text-gray-400 text-sm">
                  Screen preview starts when recording begins
                </span>
              </div>
            )}
          </div>
        )}

        {/* Recording Status */}
        <div className="flex items-center justify-center py-4">
          {isRecording ? (
            <div className="flex items-center gap-3">
              <span className="relative flex h-4 w-4">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-4 w-4 bg-red-500"></span>
              </span>
              <span className="text-2xl font-mono font-bold text-red-400">
                {formatTime(recordingTime)}
              </span>
            </div>
          ) : (
            <span className="text-gray-400">Ready to record</span>
          )}
        </div>

        {/* Control Buttons */}
        <div className="flex gap-3 justify-center">
          {!isRecording && !recordedBlob ? (
            <button 
              onClick={startRecording}
              disabled={uploading}
              className="px-6 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z"/>
              </svg>
              {uploading ? 'Uploading...' : 'Start Recording'}
            </button>
          ) : (
            <button 
              onClick={stopRecording}
              className="px-6 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 6h12v12H6z"/>
              </svg>
              Stop Recording
            </button>
          )}
          {!isRecording && recordedBlob && (
            <>
              <button
                onClick={() => uploadRecording(recordedBlob)}
                disabled={uploading}
                className="px-6 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M5 12l5 5L20 7" />
                </svg>
                {uploading ? 'Uploading...' : 'Use This Recording'}
              </button>
              <button
                type="button"
                onClick={() => {
                  if (recordedPreviewUrl) {
                    URL.revokeObjectURL(recordedPreviewUrl)
                  }
                  setRecordedBlob(null)
                  setRecordedPreviewUrl(null)
                  setHasActivePreview(false)
                }}
                className="px-6 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors"
              >
                Record Again
              </button>
            </>
          )}
        </div>

        {/* Help Text */}
        {!isRecording && (
          <div className="space-y-2 text-sm text-gray-400">
            <p className="text-center">
              Select a screen or window to capture.
              <span className="text-yellow-400 font-medium"> Enable "Share audio"</span> in the browser dialog if you want transcript audio from the recording.
            </p>
            <p className="text-center text-gray-500">
              If your browser does not provide an audio track, recording will still continue as video-only.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ScreenRecorder;
