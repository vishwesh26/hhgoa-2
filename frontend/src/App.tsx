import React, { useState, useEffect } from 'react';
import { KineticDashboard } from './components/KineticDashboard';
import { PerformanceAnalytics } from './components/PerformanceAnalytics';
import { DocumentationView } from './components/DocumentationView';
import { ProfileSettings } from './components/ProfileSettings';

export const App: React.FC = () => {
  const [activeScreen, setActiveScreen] = useState<'dashboard' | 'analytics' | 'docs' | 'settings'>('dashboard');
  const [apiUrl, setApiUrl] = useState<string>(() => {
    return localStorage.getItem('kinetic_api_url') || import.meta.env.VITE_API_BASE_URL || '';
  });
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [response, setResponse] = useState<any>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [benchmarkData, setBenchmarkData] = useState<any>(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState<boolean>(false);

  const handleApiUrlChange = (newUrl: string) => {
    const formatted = newUrl.trim().replace(/\/+$/, '');
    setApiUrl(formatted);
    localStorage.setItem('kinetic_api_url', formatted);
  };

  // Fetch benchmark data on load or when apiUrl changes
  useEffect(() => {
    fetch(`${apiUrl}/api/benchmark/results`)
      .then((res) => res.json())
      .then((data) => setBenchmarkData(data))
      .catch(() => {});
  }, [apiUrl]);

  const handleQuerySubmit = async (query: string) => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const res = await fetch(`${apiUrl}/api/rag/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      if (!res.ok) {
        if (res.status === 404) {
          throw new Error('Backend endpoint returned 404. If deployed on Vercel, please set your live backend URL (e.g. ngrok tunnel) in the Configuration/Settings tab.');
        }
        const errorData = await res.json().catch(() => ({ detail: `Server error (${res.status})` }));
        throw new Error(errorData.detail || `Query execution failed (${res.status}).`);
      }
      const data = await res.json();
      setResponse(data);
    } catch (err: any) {
      setErrorMessage(err.message || 'Network error connecting to backend.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAudioSubmit = async (audioBlob: Blob, filename: string = 'recording.webm') => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const formData = new FormData();
      formData.append('file', audioBlob, filename);

      const res = await fetch(`${apiUrl}/api/voice/query`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        if (res.status === 404) {
          throw new Error('Backend endpoint returned 404. If deployed on Vercel, please set your live backend URL (e.g. ngrok tunnel) in the Configuration/Settings tab.');
        }
        const errorData = await res.json().catch(() => ({ detail: `Voice server error (${res.status})` }));
        throw new Error(errorData.detail || `Voice processing failed (${res.status}).`);
      }
      const data = await res.json();
      setResponse(data);
    } catch (err: any) {
      setErrorMessage(
        err.message || 'Unable to reach voice backend. Make sure FastAPI is running on port 8000.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunBenchmark = async () => {
    setBenchmarkLoading(true);
    try {
      const res = await fetch(`${apiUrl}/api/benchmark/run`, { method: 'POST' });
      const data = await res.json();
      setBenchmarkData(data);
    } catch (err) {
      console.error(err);
    } finally {
      setBenchmarkLoading(false);
    }
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-surface text-on-surface font-body">
      {/* Top Navigation Bar */}
      <nav className="bg-white w-full h-16 md:h-20 border-b-4 border-black flex justify-between items-center px-4 md:px-8 shrink-0 z-20">
        <div className="flex items-center gap-4">
          <div 
            onClick={() => setActiveScreen('dashboard')}
            className="cursor-pointer font-headline font-black text-xl md:text-2xl tracking-tighter text-black uppercase flex items-center gap-2"
          >
            <span className="bg-primary text-white p-1 border-2 border-black hard-shadow-sm text-sm">
              <span className="material-symbols-outlined text-base">mic</span>
            </span>
            KINETIC.AI
            <span className="hidden sm:inline-block font-mono text-[10px] bg-black text-white px-2 py-0.5 ml-2 font-bold">
              VOICE RAG V2.0
            </span>
          </div>
        </div>

        {/* Quick Backend URL Connection Bar */}
        <div className="hidden md:flex items-center gap-2 font-mono text-xs">
          <span className="font-bold flex items-center gap-1 text-slate-700">
            <span className="material-symbols-outlined text-sm text-primary">cloud_sync</span>
            Backend URL:
          </span>
          <input
            type="text"
            value={apiUrl}
            onChange={(e) => handleApiUrlChange(e.target.value)}
            placeholder="e.g. https://xxxx.onrender.com or https://xxxx.ngrok-free.app"
            className="w-64 lg:w-80 p-1.5 border-2 border-black font-mono text-xs bg-surface focus:outline-none focus:ring-2 focus:ring-primary font-bold"
          />
          <span className={`px-2 py-1 border-2 border-black font-bold text-[10px] ${
            apiUrl ? 'bg-neon-green text-black' : 'bg-neon-yellow text-black'
          }`}>
            {apiUrl ? 'CONNECTED' : 'DEFAULT'}
          </span>
        </div>

        {/* Top Action Icons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveScreen('settings')}
            title="Settings & Profile"
            className={`p-2 border-2 border-black hover:bg-neon-yellow hover:text-black transition-transform ${
              activeScreen === 'settings' ? 'bg-black text-white' : 'bg-surface text-black'
            }`}
          >
            <span className="material-symbols-outlined text-xl">settings</span>
          </button>
          <button
            onClick={() => setActiveScreen('docs')}
            title="Documentation"
            className={`p-2 border-2 border-black hover:bg-neon-yellow hover:text-black transition-transform ${
              activeScreen === 'docs' ? 'bg-black text-white' : 'bg-surface text-black'
            }`}
          >
            <span className="material-symbols-outlined text-xl">help</span>
          </button>
          <button
            onClick={() => setActiveScreen('settings')}
            title="Account"
            className="p-2 border-2 border-black bg-surface text-black hover:bg-neon-yellow hover:text-black transition-transform"
          >
            <span className="material-symbols-outlined text-xl">account_circle</span>
          </button>
        </div>
      </nav>

      {/* Body Area with Sidebar + Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Side Navigation Bar */}
        <aside className="hidden md:flex flex-col h-full bg-surface border-r-4 border-black w-60 shrink-0 select-none">
          <div className="p-5 border-b-2 border-black">
            <h2 className="font-headline font-black text-lg text-black uppercase tracking-tight">KINETIC</h2>
            <p className="font-mono text-xs text-primary font-bold mt-0.5">V2.0.4-STABLE</p>
            <button 
              onClick={() => setActiveScreen('dashboard')}
              className="w-full mt-4 py-2 border-2 border-black bg-primary text-white font-mono text-xs font-bold hard-shadow hover:bg-primary-container uppercase"
            >
              + NEW VOICE QUERY
            </button>
          </div>

          <nav className="flex-1 overflow-y-auto font-mono text-xs">
            <button
              onClick={() => setActiveScreen('dashboard')}
              className={`w-full flex items-center gap-3 px-5 py-4 border-b-2 border-black font-bold uppercase transition-colors text-left ${
                activeScreen === 'dashboard' ? 'bg-neon-green text-black' : 'hover:bg-neon-yellow text-slate-900'
              }`}
            >
              <span className="material-symbols-outlined text-lg">sensors</span>
              Live Stream
            </button>

            <button
              onClick={() => setActiveScreen('analytics')}
              className={`w-full flex items-center gap-3 px-5 py-4 border-b-2 border-black font-bold uppercase transition-colors text-left ${
                activeScreen === 'analytics' ? 'bg-neon-green text-black' : 'hover:bg-neon-yellow text-slate-900'
              }`}
            >
              <span className="material-symbols-outlined text-lg">monitoring</span>
              Analytics
            </button>

            <button
              onClick={() => setActiveScreen('docs')}
              className={`w-full flex items-center gap-3 px-5 py-4 border-b-2 border-black font-bold uppercase transition-colors text-left ${
                activeScreen === 'docs' ? 'bg-neon-green text-black' : 'hover:bg-neon-yellow text-slate-900'
              }`}
            >
              <span className="material-symbols-outlined text-lg">database</span>
              Sources & Models
            </button>

            <button
              onClick={() => setActiveScreen('settings')}
              className={`w-full flex items-center gap-3 px-5 py-4 border-b-2 border-black font-bold uppercase transition-colors text-left ${
                activeScreen === 'settings' ? 'bg-neon-green text-black' : 'hover:bg-neon-yellow text-slate-900'
              }`}
            >
              <span className="material-symbols-outlined text-lg">tune</span>
              Configuration
            </button>
          </nav>

          <div className="mt-auto border-t-4 border-black font-mono text-xs">
            <button
              onClick={() => setActiveScreen('docs')}
              className="w-full flex items-center gap-3 px-5 py-3.5 border-b-2 border-black hover:bg-neon-yellow text-slate-900 font-bold uppercase text-left"
            >
              <span className="material-symbols-outlined text-lg">menu_book</span>
              Docs
            </button>
            <button
              onClick={() => setActiveScreen('settings')}
              className="w-full flex items-center gap-3 px-5 py-3.5 hover:bg-neon-yellow text-slate-900 font-bold uppercase text-left"
            >
              <span className="material-symbols-outlined text-lg">code</span>
              API Settings
            </button>
          </div>
        </aside>

        {/* Main Content View Switcher */}
        <main className="flex-1 flex flex-col min-w-0 overflow-y-auto bg-surface">
          {activeScreen === 'dashboard' && (
            <KineticDashboard
              onQuerySubmit={handleQuerySubmit}
              onAudioSubmit={handleAudioSubmit}
              loading={isLoading}
              result={response}
              error={errorMessage}
            />
          )}

          {activeScreen === 'analytics' && (
            <PerformanceAnalytics
              currentLatency={response?.latency}
              benchmarkData={benchmarkData}
              onRunBenchmark={handleRunBenchmark}
              benchmarkLoading={benchmarkLoading}
            />
          )}

          {activeScreen === 'docs' && <DocumentationView />}

          {activeScreen === 'settings' && (
            <ProfileSettings apiUrl={apiUrl} onApiUrlChange={handleApiUrlChange} />
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="bg-black text-white w-full py-2.5 border-t-4 border-black flex flex-wrap justify-between items-center px-4 md:px-8 shrink-0 font-mono text-xs z-20">
        <span className="font-bold">© 2026 KINETIC.AI | SYSTEM STATUS: OPTIMAL</span>
        <div className="flex gap-4">
          <span>LATENCY TARGET: &lt;200MS</span>
          <span>LANGUAGE: ENGLISH (EN)</span>
          <span className="text-neon-green font-bold">ONLINE</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
