import React, { useState, useEffect } from 'react';
import { Play, FileText, CheckCircle, AlertTriangle, XCircle, Search, Settings, Activity } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [templates, setTemplates] = useState([]);
  const [scenarios, setScenarios] = useState([]);
  
  const [nlText, setNlText] = useState("An auto-rickshaw suddenly stops 12 metres ahead while a motorcycle approaches from the opposite direction at 30 km/h on a village road.");
  const [generatedScenario, setGeneratedScenario] = useState(null);
  const [assumptions, setAssumptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => {
    fetch('/api/v1/templates')
      .then(r => r.json())
      .then(data => setTemplates(data));
      
    fetch('/api/v1/scenarios')
      .then(r => r.json())
      .then(data => setScenarios(data));
  }, []);

  const handleGenerateText = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/v1/scenarios/from-text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: nlText })
      });
      const data = await res.json();
      if (res.ok) {
        setGeneratedScenario(data.scenario);
        setAssumptions(data.assumptions || []);
        setActiveTab('review');
      } else {
        alert("Error: " + JSON.stringify(data));
      }
    } catch (e) {
      alert("Error calling NLP api");
    }
    setLoading(false);
  };

  const handleRun = async (scenario) => {
    setLoading(true);
    try {
      const res = await fetch('/api/v1/scenarios/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(scenario)
      });
      const data = await res.json();
      if (res.ok) {
        pollResult(data.result_id);
      }
    } catch(e) {
      alert("Error starting run");
      setLoading(false);
    }
  };

  const pollResult = async (resultId) => {
    const timer = setInterval(async () => {
      const res = await fetch(`/api/v1/results/${resultId}`);
      const data = await res.json();
      if (data.status !== 'running') {
        clearInterval(timer);
        setResult(data);
        setActiveTab('results');
        setLoading(false);
      }
    }, 1000);
  };

  return (
    <div className="min-h-screen p-6 bg-slate-900 text-slate-100 font-sans">
      <header className="flex items-center justify-between pb-6 border-b border-slate-700 mb-6">
        <h1 className="text-2xl font-bold text-emerald-400 flex items-center gap-2">
          <Activity size={24} />
          ADAS Scenario Lab
        </h1>
        <div className="flex gap-4">
          <button onClick={() => setActiveTab('dashboard')} className={`px-4 py-2 rounded ${activeTab === 'dashboard' ? 'bg-slate-700' : 'hover:bg-slate-800'}`}>Dashboard</button>
          <button onClick={() => setActiveTab('create')} className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded font-medium flex items-center gap-2">
            + New Scenario
          </button>
        </div>
      </header>

      {loading && (
        <div className="fixed inset-0 bg-slate-900/80 flex items-center justify-center z-50">
          <div className="text-xl animate-pulse flex flex-col items-center gap-4">
            <Activity size={48} className="text-emerald-500 animate-spin" />
            Processing...
          </div>
        </div>
      )}

      {activeTab === 'dashboard' && (
        <div className="grid grid-cols-2 gap-8">
          <div>
            <h2 className="text-xl mb-4 font-semibold text-slate-300">Quick Templates</h2>
            <div className="grid grid-cols-1 gap-4">
              {templates.map(t => (
                <div key={t.id} className="p-4 bg-slate-800 border border-slate-700 rounded shadow hover:border-emerald-500 cursor-pointer transition"
                     onClick={() => { setGeneratedScenario(t.scenario); setActiveTab('review'); }}>
                  <h3 className="font-bold text-emerald-300 text-lg mb-1">{t.name}</h3>
                  <p className="text-sm text-slate-400">{t.description}</p>
                  <div className="mt-3 flex gap-2">
                    {t.tags?.map(tag => <span key={tag} className="px-2 py-1 bg-slate-700 text-xs rounded-full">{tag}</span>)}
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div>
            <h2 className="text-xl mb-4 font-semibold text-slate-300">Recent Scenarios</h2>
            <div className="space-y-3">
              {scenarios.length === 0 ? <p className="text-slate-500">No scenarios generated yet.</p> :
               scenarios.map(s => (
                <div key={s.id} className="p-3 bg-slate-800/50 border border-slate-700 rounded flex justify-between items-center">
                  <div>
                    <div className="font-medium">{s.name}</div>
                    <div className="text-xs text-slate-500">{new Date(s.created_at).toLocaleString()}</div>
                  </div>
                  <div className="text-xs px-2 py-1 bg-slate-700 rounded text-slate-300">{s.source}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'create' && (
        <div className="max-w-3xl mx-auto">
          <div className="bg-slate-800 p-6 rounded-lg shadow-lg border border-slate-700">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <FileText className="text-emerald-400" /> Describe your scenario
            </h2>
            <textarea 
              className="w-full h-32 bg-slate-900 border border-slate-600 rounded p-4 text-slate-200 font-mono text-sm focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500 transition"
              value={nlText}
              onChange={(e) => setNlText(e.target.value)}
            />
            <div className="mt-4 flex justify-end">
              <button 
                onClick={handleGenerateText}
                className="px-6 py-3 bg-emerald-600 hover:bg-emerald-500 rounded text-white font-medium transition flex items-center gap-2"
              >
                Interpret Scenario <Play size={16}/>
              </button>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'review' && generatedScenario && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
            <h2 className="text-xl font-bold mb-4 text-amber-400 flex items-center gap-2">
              <AlertTriangle size={20} /> Assumptions Made
            </h2>
            <div className="bg-slate-900 p-4 rounded text-sm text-slate-300 space-y-2 font-mono">
              {assumptions.length === 0 ? <p>No implicit assumptions detected.</p> : 
                assumptions.map((a, i) => <div key={i} className="flex gap-2"><CheckCircle size={16} className="text-emerald-500 shrink-0 mt-0.5"/> {a}</div>)
              }
              <div className="flex gap-2 mt-4 pt-4 border-t border-slate-800">
                <CheckCircle size={16} className="text-emerald-500 shrink-0 mt-0.5"/> 
                Goal X: {generatedScenario.ego_vehicle.goal_x_m}m
              </div>
            </div>
          </div>

          <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Settings size={20} /> JSON Representation
            </h2>
            <pre className="bg-slate-900 p-4 rounded text-xs text-slate-300 overflow-auto h-64 border border-slate-700">
              {JSON.stringify(generatedScenario, null, 2)}
            </pre>
            <div className="mt-6 flex justify-between">
              <button onClick={() => setActiveTab('create')} className="px-4 py-2 border border-slate-600 hover:bg-slate-700 rounded">
                ← Back
              </button>
              <button onClick={() => handleRun(generatedScenario)} className="px-6 py-2 bg-emerald-600 hover:bg-emerald-500 rounded flex items-center gap-2 font-medium">
                Run Simulation <Play size={16} fill="currentColor" />
              </button>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'results' && result && (
        <div className="max-w-4xl mx-auto space-y-6">
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-slate-800 p-6 rounded-lg border border-slate-700 flex flex-col items-center justify-center text-center">
              {result.arrived ? (
                <CheckCircle size={64} className="text-emerald-500 mb-4" />
              ) : (
                <XCircle size={64} className="text-rose-500 mb-4" />
              )}
              <h2 className="text-2xl font-bold mb-2">{result.explanation?.outcome_summary}</h2>
              <div className="text-slate-400 text-sm">
                Duration: {result.t_total_s}s • Collisions: {result.collisions}
              </div>
            </div>
            
            <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
              <h3 className="font-bold text-lg mb-3">Key Metrics</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="bg-slate-900 p-3 rounded">
                  <div className="text-slate-500">Min Clearance</div>
                  <div className="text-xl font-mono">{result.min_clearance_m?.toFixed(2)}m</div>
                </div>
                <div className="bg-slate-900 p-3 rounded">
                  <div className="text-slate-500">Replans</div>
                  <div className="text-xl font-mono">{result.replans}</div>
                </div>
                <div className="bg-slate-900 p-3 rounded">
                  <div className="text-slate-500">Avg Replan Latency</div>
                  <div className="text-xl font-mono">{result.mean_replan_ms?.toFixed(2)}ms</div>
                </div>
                <div className="bg-slate-900 p-3 rounded">
                  <div className="text-slate-500">Max Speed</div>
                  <div className="text-xl font-mono">{result.max_speed_kmh?.toFixed(1)} km/h</div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
            <h3 className="font-bold text-lg mb-4 border-b border-slate-700 pb-2">Analysis & Explanation</h3>
            <ul className="space-y-2 mb-6 text-slate-300">
              {result.explanation?.critical_events.map((e, i) => (
                <li key={i} className="flex gap-2 items-start">
                  <span className="text-rose-400 mt-1">•</span> {e}
                </li>
              ))}
              {result.explanation?.failure_reason && (
                <li className="flex gap-2 items-start text-rose-300 mt-4 p-3 bg-rose-950/30 rounded border border-rose-900/50">
                  <AlertTriangle size={18} className="shrink-0 mt-0.5" />
                  Failure Reason: {result.explanation.failure_reason}
                </li>
              )}
            </ul>
            
            <h3 className="font-bold text-sm text-slate-400 mb-2 uppercase tracking-wider">Decision Distribution</h3>
            <div className="h-4 w-full bg-slate-900 rounded-full flex overflow-hidden">
              <div style={{width: `${result.decision_proceed * 100}%`}} className="bg-emerald-500" title={`Proceed: ${(result.decision_proceed*100).toFixed(0)}%`}></div>
              <div style={{width: `${result.decision_caution * 100}%`}} className="bg-amber-400" title={`Caution: ${(result.decision_caution*100).toFixed(0)}%`}></div>
              <div style={{width: `${result.decision_slow * 100}%`}} className="bg-orange-500" title={`Slow: ${(result.decision_slow*100).toFixed(0)}%`}></div>
              <div style={{width: `${result.decision_brake * 100}%`}} className="bg-rose-500" title={`Brake: ${(result.decision_brake*100).toFixed(0)}%`}></div>
              <div style={{width: `${result.decision_emergency_stop * 100}%`}} className="bg-rose-700" title={`EmStop: ${(result.decision_emergency_stop*100).toFixed(0)}%`}></div>
            </div>
            <div className="flex justify-between text-xs text-slate-500 mt-2">
              <span className="text-emerald-500">Proceed</span>
              <span className="text-amber-400">Caution</span>
              <span className="text-rose-500">Brake</span>
              <span className="text-rose-700">Em. Stop</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
