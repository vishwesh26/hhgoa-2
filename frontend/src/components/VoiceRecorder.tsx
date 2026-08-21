import React, { useState, useRef, useEffect } from 'react';

interface VoiceRecorderProps {
  onAudioSubmit: (audioBlob: Blob, filename: string) => void;
  disabled?: boolean;
}

export const VoiceRecorder: React.FC<VoiceRecorderProps> = ({
  onAudioSubmit,
  disabled = false
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const mimeTypeRef = useRef<string>('audio/webm');

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      let selectedMime = 'audio/webm';
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
        selectedMime = 'audio/webm;codecs=opus';
      } else if (MediaRecorder.isTypeSupported('audio/webm')) {
        selectedMime = 'audio/webm';
      } else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
        selectedMime = 'audio/ogg;codecs=opus';
      } else if (MediaRecorder.isTypeSupported('audio/wav')) {
        selectedMime = 'audio/wav';
      }
      mimeTypeRef.current = selectedMime;

      const mediaRecorder = new MediaRecorder(stream, { mimeType: selectedMime });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const rawMime = mimeTypeRef.current;
        const cleanMime = rawMime.split(';')[0];
        let extension = 'webm';
        if (cleanMime.includes('ogg')) extension = 'ogg';
        else if (cleanMime.includes('mp4')) extension = 'mp4';
        else if (cleanMime.includes('wav')) extension = 'wav';

        const audioBlob = new Blob(audioChunksRef.current, { type: cleanMime });
        const filename = `recording_${Date.now()}.${extension}`;
        
        if (audioBlob.size > 1000) {
          onAudioSubmit(audioBlob, filename);
        } else {
          console.warn('Audio recording too small or empty');
        }
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start(100);
      setIsRecording(true);
      setRecordingTime(0);
      timerRef.current = window.setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error('Microphone error:', err);
      alert('Microphone access denied or unavailable. Please enable microphone permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      if (mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.requestData();
        mediaRecorderRef.current.stop();
      }
      setIsRecording(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const remainingSecs = secs % 60;
    return `${mins}:${remainingSecs < 10 ? '0' : ''}${remainingSecs}`;
  };

  return (
    <div className="flex flex-col items-center gap-4">
      {/* Brutalist Circular Record Button */}
      <button
        type="button"
        onClick={isRecording ? stopRecording : startRecording}
        disabled={disabled}
        className={`w-32 h-32 rounded-full border-4 border-black flex flex-col items-center justify-center transition-transform hard-shadow active:translate-x-1 active:translate-y-1 ${
          isRecording
            ? 'bg-neon-yellow text-black animate-pulse'
            : 'bg-primary text-white hover:bg-primary-container'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
      >
        <span className="material-symbols-outlined text-5xl mb-0.5">
          {isRecording ? 'stop_circle' : 'mic'}
        </span>
        <span className="font-mono text-[11px] font-black uppercase tracking-wider">
          {isRecording ? 'STOP' : 'RECORD'}
        </span>
      </button>

      {/* Recording Status / Timer */}
      {isRecording ? (
        <div className="flex items-center gap-2 font-mono text-xs font-bold text-red-600 bg-red-50 px-3 py-1 border-2 border-red-500 rounded">
          <div className="w-2.5 h-2.5 bg-red-600 rounded-full animate-ping"></div>
          RECORDING: {formatTime(recordingTime)} (CLICK TO FINISH)
        </div>
      ) : (
        <span className="font-mono text-xs text-slate-500 font-bold">
          Click the button and speak into your mic
        </span>
      )}
    </div>
  );
};
