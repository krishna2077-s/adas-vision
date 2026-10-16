import React, { useState, useEffect } from 'react';
import {
  Play, FileText, CheckCircle, AlertTriangle, XCircle,
  Activity, LayoutDashboard, PlusCircle, Car, Copy,
  ArrowLeft, Clock, Shield, ChevronRight, Layers,
  Cpu, Gauge, BookOpen, X, RotateCcw, Download,
  Code, FileCode, Globe, Box, Compass, MapPin,
  Check, CloudSun
} from 'lucide-react';

// ─── Types ────────────────────────────────────────────────────────────────────

interface Actor {
  id: string;
  class?: string;
  actor_class?: string;
  initial_x_m: number;
  initial_y_m: number;
  initial_vx_mps?: number;
  initial_vy_mps?: number;
  width_m?: number;
  length_m?: number;
  behaviour: string;
}

interface Scenario {
  scenario_id: string;
  metadata: {
    name: string;
    description: string;
    source: string;
    created_at: string;
    tags: string[];
  };
  road: {
    type: string;
    lane_half_width_m: number;
    length_m: number;
    surface?: string;
    markings?: string;
    has_shoulder?: boolean;
  };
  environment?: {
    time_of_day?: string;
    weather?: string;
    visibility_m?: number;
  };
  ego_vehicle: {
    initial_speed_kmh: number;
    goal_x_m: number;
    mode: string;
    initial_x_m?: number;
    initial_y_m?: number;
    initial_heading_deg?: number;
  };
  actors: Actor[];
  simulation_parameters: {
    dt_s?: number;
    max_time_s: number;
  };
}

interface SimResult {
  arrived: boolean;
  t_total_s: number;
  collisions: number;
  min_clearance_m: number;
  replans: number;
  mean_replan_ms: number;
  max_replan_ms: number;
  mean_speed_kmh: number;
  max_speed_kmh: number;
  decision_proceed: number;
  decision_caution: number;
  decision_slow: number;
  decision_brake: number;
  decision_emergency_stop: number;
  traj_x: number[];
  traj_y: number[];
  explanation: {
    outcome_summary: string;
    critical_events: string[];
    avoidance_successful: boolean;
    failure_reason?: string;
  };
  status: string;
  mode: string;
}

interface IndianAsset {
  id: string;
  class: string;
  category: string;
  name: string;
  dimensions: {
    length_m: number;
    width_m: number;
    height_m: number;
  };
  collision_classification: string;
  radar_cross_section?: string;
  mass_kg?: number;
  typical_speed_kmh?: number;
  description: string;
}

interface ExportModalState {
  isOpen: boolean;
  type: 'dsd' | 'opendrive';
  title: string;
  filename: string;
  content: string;
  loading: boolean;
  error?: string | null;
}

type Tab = 'dashboard' | 'create' | 'review' | 'results' | 'assets';
type CreateMode = 'nlp' | 'osm';

// ─── Helpers ──────────────────────────────────────────────────────────────────

const ACTOR_COLORS: Record<string, string> = {
  cow: '#f59e0b',
  buffalo: '#d97706',
  dog: '#a78bfa',
  goat: '#fb7185',
  person: '#60a5fa',
  child: '#f472b6',
  motorcycle: '#34d399',
  scooter: '#4ade80',
  bicycle: '#86efac',
  auto_rickshaw: '#fb923c',
  e_rickshaw: '#fbbf24',
  pushcart: '#94a3b8',
  thela: '#cbd5e1',
  car: '#64748b',
  suv: '#475569',
  bus: '#1d4ed8',
  truck: '#1e40af',
  tractor: '#15803d',
  barricade: '#e11d48',
  police_barricade: '#ef4444',
  construction_barrier: '#f97316',
  pothole: '#94a3b8',
  obstacle: '#dc2626',
};

const getActorClass = (a: Actor): string => a.actor_class || a.class || 'car';

const EXAMPLE_PROMPTS = [
  'A cow suddenly crosses a narrow village road at night with heavy fog. Ego vehicle is going 40 km/h.',
  'Two children run across an urban street. A motorcycle is approaching from the opposite side at 30 km/h.',
  'A tractor pulling a load is moving slowly on the highway. An auto-rickshaw cuts in from the left.',
  'A stray dog runs into a market street. A pushcart is blocking half the road ahead.',
];

const OSM_PRESETS = [
  { name: 'Mumbai (BKC)', lat: 19.0657, lon: 72.8687, desc: 'Bandra Kurla Complex - Urban Arterial Highway' },
  { name: 'Delhi (CP)', lat: 28.6304, lon: 77.2177, desc: 'Connaught Place - Dense Radial Market Street' },
  { name: 'Bengaluru (Silk Board)', lat: 12.9176, lon: 77.6238, desc: 'Heavy Mixed Traffic & Highway Interchange' },
  { name: 'Nashik (K.K. Wagh - SIH Winners)', lat: 20.0110, lon: 73.8236, desc: 'Campus & Ring Road - Team TwinX Origin' },
];

const WEATHER_ICONS: Record<string, string> = {
  clear: '☀️',
  overcast: '☁️',
  light_rain: '🌦️',
  heavy_rain: '🌧️',
  fog: '🌫️',
};

const TOD_ICONS: Record<string, string> = {
  day: '🌅',
  dusk: '🌆',
  night: '🌙',
};

// ─── Sub-components ───────────────────────────────────────────────────────────

function ErrorBanner({ message, onClose }: { message: string; onClose: () => void }) {
  return (
    <div className="flex items-start gap-3 p-4 mb-6 bg-rose-950/60 border border-rose-700 rounded-lg text-rose-200 text-sm">
      <AlertTriangle size={18} className="shrink-0 mt-0.5 text-rose-400" />
      <span className="flex-1">{message}</span>
      <button onClick={onClose} className="text-rose-400 hover:text-rose-200 transition"><X size={16} /></button>
    </div>
  );
}

function LoadingOverlay({ step }: { step: string }) {
  return (
    <div className="fixed inset-0 bg-slate-900/85 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-slate-800 border border-slate-700 rounded-2xl p-8 flex flex-col items-center gap-5 shadow-2xl min-w-64">
        <div className="relative">
          <div className="w-14 h-14 rounded-full border-4 border-slate-700 border-t-emerald-500 animate-spin" />
          <Activity size={20} className="absolute inset-0 m-auto text-emerald-400" />
        </div>
        <div className="text-center">
          <p className="text-white font-semibold text-lg">Processing</p>
          <p className="text-slate-400 text-sm mt-1">{step}</p>
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon, label, value, color }: { icon: React.ReactNode; label: string; value: string | number; color: string }) {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl p-5 flex items-center gap-4">
      <div className={`p-3 rounded-lg ${color}`}>{icon}</div>
      <div>
        <p className="text-slate-400 text-xs uppercase tracking-wider">{label}</p>
        <p className="text-2xl font-bold text-white mt-0.5">{value}</p>
      </div>
    </div>
  );
}

function MetricCard({ label, value, unit, good }: { label: string; value: string | number; unit?: string; good?: boolean }) {
  return (
    <div className="bg-slate-900 border border-slate-700/50 rounded-xl p-4">
      <p className="text-slate-500 text-xs uppercase tracking-wider mb-1">{label}</p>
      <p className={`text-2xl font-mono font-bold ${good === true ? 'text-emerald-400' : good === false ? 'text-rose-400' : 'text-white'}`}>
        {value}<span className="text-sm text-slate-500 ml-1 font-sans font-normal">{unit}</span>
      </p>
    </div>
  );
}

// Bird's-eye scenario canvas
function ScenarioCanvas({ scenario }: { scenario: Scenario }) {
  const W = 560, H = 200;
  const roadLength = scenario.road.length_m;
  const halfWidth = scenario.road.lane_half_width_m;
  const scale = (W - 60) / roadLength;
  const cx = (x: number) => 30 + x * scale;
  const cy = (y: number) => H / 2 - y * scale * 2.5;

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="rounded-lg bg-slate-950 border border-slate-700">
      {/* Road surface */}
      <rect x={30} y={cy(halfWidth)} width={(roadLength) * scale} height={halfWidth * 2 * scale * 2.5}
        fill="#1e293b" rx="2" />
      {/* Center line */}
      <line x1={30} y1={H / 2} x2={30 + roadLength * scale} y2={H / 2}
        stroke="#475569" strokeWidth="1" strokeDasharray="8,6" />
      {/* Road edges */}
      <line x1={30} y1={cy(halfWidth)} x2={30 + roadLength * scale} y2={cy(halfWidth)} stroke="#334155" strokeWidth="1.5" />
      <line x1={30} y1={cy(-halfWidth)} x2={30 + roadLength * scale} y2={cy(-halfWidth)} stroke="#334155" strokeWidth="1.5" />

      {/* Actors */}
      {scenario.actors.map(actor => {
        const ax = cx(actor.initial_x_m);
        const ay = cy(actor.initial_y_m);
        const aClass = getActorClass(actor);
        const color = ACTOR_COLORS[aClass] ?? '#94a3b8';
        const isHazard = ['barricade', 'police_barricade', 'construction_barrier', 'pothole', 'obstacle'].includes(aClass);
        return (
          <g key={actor.id}>
            {isHazard ? (
              <rect x={ax - 6} y={ay - 6} width="12" height="12" fill={color} opacity={0.9} rx="2" />
            ) : (
              <circle cx={ax} cy={ay} r={7} fill={color} opacity={0.9} />
            )}
            <text x={ax} y={ay - 11} textAnchor="middle" fontSize="9" fill={color} fontFamily="monospace">
              {aClass}
            </text>
          </g>
        );
      })}

      {/* Ego vehicle */}
      <g transform={`translate(${cx(scenario.ego_vehicle.initial_x_m ?? 0)},${H / 2})`}>
        <rect x="-10" y="-6" width="20" height="12" rx="3" fill="#10b981" />
        <polygon points="10,0 14,-4 14,4" fill="#10b981" />
        <text y="20" textAnchor="middle" fontSize="9" fill="#10b981" fontFamily="monospace">ego</text>
      </g>

      {/* Goal flag */}
      <line x1={cx(scenario.ego_vehicle.goal_x_m)} y1={H / 2 - 30} x2={cx(scenario.ego_vehicle.goal_x_m)} y2={H / 2 + 10} stroke="#6366f1" strokeWidth="1.5" />
      <rect x={cx(scenario.ego_vehicle.goal_x_m)} y={H / 2 - 30} width="16" height="10" fill="#6366f1" rx="1" />
      <text x={cx(scenario.ego_vehicle.goal_x_m) + 2} y={H / 2 - 23} fontSize="8" fill="white">G</text>

      {/* Legend */}
      <circle cx={W - 80} cy={H - 14} r={5} fill="#10b981" />
      <text x={W - 72} y={H - 10} fontSize="9" fill="#94a3b8">Ego</text>
      <circle cx={W - 40} cy={H - 14} r={5} fill="#6366f1" />
      <text x={W - 32} y={H - 10} fontSize="9" fill="#94a3b8">Goal</text>
    </svg>
  );
}

// Trajectory replay canvas (results)
function TrajectoryCanvas({ traj_x, traj_y, scenario }: { traj_x: number[]; traj_y: number[]; scenario?: Scenario | null }) {
  if (!traj_x?.length) return (
    <div className="rounded-lg bg-slate-950 border border-slate-700 flex items-center justify-center h-36 text-slate-600 text-sm">
      No trajectory data
    </div>
  );
  const W = 560, H = 160;
  const minX = Math.min(...traj_x), maxX = Math.max(...traj_x);
  const minY = Math.min(...traj_y), maxY = Math.max(...traj_y);
  const pad = 20;
  const sx = (x: number) => pad + ((x - minX) / (maxX - minX || 1)) * (W - 2 * pad);
  const sy = (y: number) => H - pad - ((y - minY) / (maxY - minY || 1)) * (H - 2 * pad);
  const pts = traj_x.map((x, i) => `${sx(x)},${sy(traj_y[i])}`).join(' ');

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="rounded-lg bg-slate-950 border border-slate-700">
      {scenario?.road && (
        <line x1={pad} y1={H / 2} x2={W - pad} y2={H / 2} stroke="#334155" strokeDasharray="4,4" />
      )}
      <polyline points={pts} fill="none" stroke="#10b981" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" opacity="0.8" />
      {/* Start */}
      <circle cx={sx(traj_x[0])} cy={sy(traj_y[0])} r={5} fill="#6366f1" />
      {/* End */}
      <circle cx={sx(traj_x[traj_x.length - 1])} cy={sy(traj_y[traj_y.length - 1])} r={5} fill="#10b981" />
      <text x={sx(traj_x[0]) + 8} y={sy(traj_y[0]) + 4} fontSize="9" fill="#818cf8">start</text>
      <text x={sx(traj_x[traj_x.length - 1]) + 8} y={sy(traj_y[traj_y.length - 1]) + 4} fontSize="9" fill="#34d399">end</text>
    </svg>
  );
}

// Decision distribution bar
function DecisionBar({ result }: { result: SimResult }) {
  const segments = [
    { key: 'proceed', val: result.decision_proceed, color: 'bg-emerald-500', label: 'Proceed' },
    { key: 'caution', val: result.decision_caution, color: 'bg-amber-400', label: 'Caution' },
    { key: 'slow', val: result.decision_slow, color: 'bg-orange-500', label: 'Slow' },
    { key: 'brake', val: result.decision_brake, color: 'bg-rose-500', label: 'Brake' },
    { key: 'estop', val: result.decision_emergency_stop, color: 'bg-rose-800', label: 'E-Stop' },
  ];
  return (
    <div>
      <div className="h-5 w-full bg-slate-900 rounded-full flex overflow-hidden border border-slate-700">
        {segments.map(s => (
          <div key={s.key} style={{ width: `${s.val * 100}%` }} className={`${s.color} transition-all`}
            title={`${s.label}: ${(s.val * 100).toFixed(1)}%`} />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
        {segments.map(s => (
          <div key={s.key} className="flex items-center gap-1.5 text-xs text-slate-400">
            <div className={`w-2.5 h-2.5 rounded-sm ${s.color}`} />
            {s.label} <span className="text-slate-500">{(s.val * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Code Preview & Download Modal for DSD and OpenDRIVE
function ExportPreviewModal({
  modal,
  onClose,
}: {
  modal: ExportModalState;
  onClose: () => void;
}) {
  const [copied, setCopied] = useState(false);

  if (!modal.isOpen) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(modal.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([modal.content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = modal.filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 bg-slate-950/85 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/90">
          <div className="flex items-center gap-3">
            {modal.type === 'dsd' ? (
              <div className="p-2 bg-amber-500/10 text-amber-400 rounded-lg">
                <Code size={20} />
              </div>
            ) : (
              <div className="p-2 bg-sky-500/10 text-sky-400 rounded-lg">
                <FileCode size={20} />
              </div>
            )}
            <div>
              <h2 className="text-white font-bold text-base">{modal.title}</h2>
              <p className="text-slate-400 text-xs font-mono">{modal.filename}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              disabled={modal.loading || !!modal.error}
              className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-slate-200 text-xs font-medium flex items-center gap-1.5 transition disabled:opacity-40"
            >
              {copied ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
              {copied ? 'Copied!' : 'Copy Code'}
            </button>
            <button
              onClick={handleDownload}
              disabled={modal.loading || !!modal.error}
              className="px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-white text-xs font-medium flex items-center gap-1.5 transition disabled:opacity-40 shadow-sm"
            >
              <Download size={14} /> Download File
            </button>
            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition ml-2"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-auto p-6 bg-slate-950 font-mono text-xs text-slate-300">
          {modal.loading && (
            <div className="flex flex-col items-center justify-center py-20 text-slate-400 gap-3">
              <div className="w-8 h-8 rounded-full border-2 border-slate-700 border-t-emerald-500 animate-spin" />
              <p>Generating simulator export file...</p>
            </div>
          )}
          {modal.error && (
            <div className="p-4 bg-rose-950/50 border border-rose-800 rounded-lg text-rose-300 flex items-start gap-2">
              <AlertTriangle size={16} className="shrink-0 mt-0.5" />
              <span>{modal.error}</span>
            </div>
          )}
          {!modal.loading && !modal.error && (
            <pre className="whitespace-pre overflow-x-auto leading-relaxed select-text">{modal.content}</pre>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('dashboard');
  const [createMode, setCreateMode] = useState<CreateMode>('nlp');
  const [templates, setTemplates] = useState<any[]>([]);
  const [scenarios, setScenarios] = useState<any[]>([]);
  const [nlText, setNlText] = useState('');
  const [generatedScenario, setGeneratedScenario] = useState<Scenario | null>(null);
  const [assumptions, setAssumptions] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState('');
  const [result, setResult] = useState<SimResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [jsonVisible, setJsonVisible] = useState(false);
  const [copied, setCopied] = useState(false);

  // OSM Import state
  const [osmLat, setOsmLat] = useState('19.0657');
  const [osmLon, setOsmLon] = useState('72.8687');
  const [osmRadius, setOsmRadius] = useState('300');
  const [osmAttachWeather, setOsmAttachWeather] = useState(true);
  const [osmSummary, setOsmSummary] = useState<any | null>(null);

  // Assets Library state
  const [assetCatalog, setAssetCatalog] = useState<{ assets: IndianAsset[]; categories: Record<string, IndianAsset[]> } | null>(null);

  // Export Modal state
  const [exportModal, setExportModal] = useState<ExportModalState>({
    isOpen: false,
    type: 'dsd',
    title: '',
    filename: '',
    content: '',
    loading: false,
    error: null,
  });

  const loadScenarios = () => {
    fetch('/api/v1/scenarios').then(r => r.json()).then(setScenarios).catch(() => {});
  };

  useEffect(() => {
    fetch('/api/v1/templates').then(r => r.json()).then(setTemplates).catch(() => {});
    loadScenarios();
    fetch('/api/v1/assets/library')
      .then(r => r.json())
      .then(setAssetCatalog)
      .catch(() => {});
  }, []);

  const handleGenerateText = async () => {
    if (!nlText.trim()) return;
    setLoading(true);
    setLoadingStep('Calling Gemini to parse your scenario...');
    setError(null);
    try {
      const res = await fetch('/api/v1/scenarios/from-text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: nlText }),
      });
      const data = await res.json();
      if (res.ok) {
        setGeneratedScenario(data.scenario);
        setAssumptions(data.assumptions || []);
        setOsmSummary(null);
        setResult(null);
        setActiveTab('review');
        loadScenarios();
      } else {
        setError(data.detail ?? JSON.stringify(data));
      }
    } catch {
      setError('Could not reach backend. Is the server running on port 8000?');
    } finally {
      setLoading(false);
    }
  };

  const handleImportOSM = async () => {
    const lat = parseFloat(osmLat);
    const lon = parseFloat(osmLon);
    const radius = parseFloat(osmRadius);

    if (isNaN(lat) || isNaN(lon) || isNaN(radius)) {
      setError('Please provide valid numerical coordinates and radius.');
      return;
    }

    setLoading(true);
    setLoadingStep('Querying OpenStreetMap Overpass API & Live Climate model...');
    setError(null);

    try {
      const res = await fetch('/api/v1/scenarios/from-osm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lat,
          lon,
          radius,
          attach_weather: osmAttachWeather,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setGeneratedScenario(data.scenario);
        setOsmSummary(data.summary);
        setAssumptions([
          `Imported from OpenStreetMap around GPS (${lat}, ${lon}) with ${radius}m radius.`,
          `Detected ${data.summary?.segment_count || 1} road segments (${data.summary?.total_length_m || 200}m total road length).`,
          data.summary?.weather
            ? `Live weather applied: ${data.summary.weather.condition} (${data.summary.weather.temperature_c}°C, wetness: ${data.summary.weather.road_wetness}).`
            : 'Default clear daytime weather applied.',
        ]);
        setResult(null);
        setActiveTab('review');
        loadScenarios();
      } else {
        setError(data.detail ?? 'Failed to import OpenStreetMap road data.');
      }
    } catch {
      setError('Network error while connecting to OpenStreetMap importer endpoint.');
    } finally {
      setLoading(false);
    }
  };

  const handleRun = async (scenario: Scenario) => {
    setLoading(true);
    setLoadingStep('Queuing MATLAB simulation...');
    setError(null);
    try {
      const res = await fetch('/api/v1/scenarios/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(scenario),
      });
      const data = await res.json();
      if (res.ok) {
        setLoadingStep('Running simulation — polling for results...');
        pollResult(data.result_id);
      } else {
        setError(data.detail ?? 'Failed to start simulation.');
        setLoading(false);
      }
    } catch {
      setError('Could not reach the backend.');
      setLoading(false);
    }
  };

  const pollResult = (resultId: string) => {
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`/api/v1/results/${resultId}`);
        const data = await res.json();
        if (data.status !== 'running') {
          clearInterval(timer);
          if (data.status?.startsWith('error')) {
            setError(data.message ?? 'Simulation failed.');
          } else {
            setResult(data);
            setActiveTab('results');
          }
          setLoading(false);
        }
      } catch {
        clearInterval(timer);
        setError('Lost connection while polling for results.');
        setLoading(false);
      }
    }, 1200);
  };

  const handleOpenExport = async (scenarioId: string, type: 'dsd' | 'opendrive') => {
    setExportModal({
      isOpen: true,
      type,
      title: type === 'dsd'
        ? 'MATLAB Driving Scenario Designer (.m) Script'
        : 'MathWorks RoadRunner / ASAM OpenDRIVE (.xodr) XML',
      filename: `${scenarioId}.${type === 'dsd' ? 'm' : 'xodr'}`,
      content: '',
      loading: true,
      error: null,
    });

    try {
      const body = generatedScenario && generatedScenario.scenario_id === scenarioId
        ? JSON.stringify(generatedScenario)
        : undefined;
      const res = await fetch(`/api/v1/scenarios/${scenarioId}/export/${type}`, {
        method: 'POST',
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
        body,
      });
      const data = await res.json();
      if (res.ok) {
        setExportModal(prev => ({
          ...prev,
          loading: false,
          filename: data.filename,
          content: type === 'dsd' ? data.script : data.xodr,
        }));
      } else {
        setExportModal(prev => ({
          ...prev,
          loading: false,
          error: data.detail || `Failed to export scenario to ${type}.`,
        }));
      }
    } catch {
      setExportModal(prev => ({
        ...prev,
        loading: false,
        error: 'Network error calling exporter endpoint.',
      }));
    }
  };

  const handleCopyJSON = () => {
    if (!generatedScenario) return;
    navigator.clipboard.writeText(JSON.stringify(generatedScenario, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleReset = () => {
    setGeneratedScenario(null);
    setResult(null);
    setAssumptions([]);
    setNlText('');
    setOsmSummary(null);
    setError(null);
    setActiveTab('create');
  };

  const navItems: { id: Tab; icon: React.ReactNode; label: string; badge?: string }[] = [
    { id: 'dashboard', icon: <LayoutDashboard size={18} />, label: 'Dashboard' },
    { id: 'create', icon: <PlusCircle size={18} />, label: 'Create', badge: 'OSM + NLP' },
    { id: 'review', icon: <FileText size={18} />, label: 'Review', badge: generatedScenario ? '✓' : undefined },
    { id: 'results', icon: <Activity size={18} />, label: 'Results', badge: result ? '✓' : undefined },
    { id: 'assets', icon: <Box size={18} />, label: '3D Assets', badge: 'Indian' },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex font-sans">
      {loading && <LoadingOverlay step={loadingStep} />}
      <ExportPreviewModal modal={exportModal} onClose={() => setExportModal(prev => ({ ...prev, isOpen: false }))} />

      {/* Sidebar */}
      <aside className="w-60 shrink-0 bg-slate-900 border-r border-slate-800 flex flex-col min-h-screen">
        <div className="px-5 py-5 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-emerald-500/20 rounded-lg flex items-center justify-center">
              <Activity size={18} className="text-emerald-400" />
            </div>
            <div>
              <p className="text-white font-bold text-sm leading-tight">ADAS</p>
              <p className="text-emerald-400 text-xs leading-tight">Scenario Lab</p>
            </div>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1">
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all
                ${activeTab === item.id
                  ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'}`}
            >
              {item.icon}
              <span className="flex-1 text-left">{item.label}</span>
              {item.badge && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700/80 text-slate-300 font-mono">
                  {item.badge}
                </span>
              )}
            </button>
          ))}
        </nav>

        <div className="px-4 py-4 border-t border-slate-800">
          <div className="text-xs text-slate-500 space-y-1">
            <p className="flex items-center justify-between">
              <span>Backend API</span>
              <span className="text-emerald-400 font-mono">:8000</span>
            </p>
            <p className="flex items-center justify-between">
              <span>Frontend App</span>
              <span className="text-emerald-400 font-mono">:5173</span>
            </p>
            <p className="text-[11px] text-slate-600 pt-1">SIH Winner ADAS Engine</p>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <div className="max-w-5xl mx-auto px-8 py-8">

          {error && <ErrorBanner message={error} onClose={() => setError(null)} />}

          {/* ── DASHBOARD ── */}
          {activeTab === 'dashboard' && (
            <div className="space-y-8">
              <div>
                <h1 className="text-2xl font-bold text-white">Dashboard</h1>
                <p className="text-slate-400 mt-1 text-sm">
                  Indian road ADAS scenario synthesis: OpenStreetMap Live GPS, Weather API, MATLAB DSD & RoadRunner OpenDRIVE.
                </p>
              </div>

              <div className="grid grid-cols-4 gap-4">
                <StatCard icon={<Layers size={20} className="text-emerald-400" />} label="Scenarios" value={scenarios.length} color="bg-emerald-500/10" />
                <StatCard icon={<BookOpen size={20} className="text-violet-400" />} label="Templates" value={templates.length} color="bg-violet-500/10" />
                <StatCard icon={<Box size={20} className="text-amber-400" />} label="Indian 3D Assets" value={assetCatalog?.assets?.length || 14} color="bg-amber-500/10" />
                <StatCard icon={<Globe size={20} className="text-sky-400" />} label="OSM Roads" value="Live GPS" color="bg-sky-500/10" />
              </div>

              {/* Templates */}
              <div>
                <h2 className="text-lg font-semibold text-slate-200 mb-3 flex items-center gap-2">
                  <BookOpen size={18} className="text-violet-400" /> Curated Indian Road Templates
                </h2>
                {templates.length === 0 ? (
                  <div className="border border-dashed border-slate-700 rounded-xl p-8 text-center text-slate-500 text-sm">
                    No templates loaded.
                  </div>
                ) : (
                  <div className="grid grid-cols-2 gap-4">
                    {templates.map((t: any) => (
                      <div key={t.id}
                        className="p-4 bg-slate-800 border border-slate-700 rounded-xl hover:border-emerald-500/50 cursor-pointer transition-all group flex flex-col justify-between"
                        onClick={() => { setGeneratedScenario(t.scenario); setAssumptions([]); setActiveTab('review'); }}>
                        <div>
                          <div className="flex items-start justify-between">
                            <h3 className="font-semibold text-emerald-300 text-sm">{t.name}</h3>
                            <ChevronRight size={16} className="text-slate-600 group-hover:text-emerald-400 transition mt-0.5" />
                          </div>
                          <p className="text-xs text-slate-400 mt-1">{t.description}</p>
                        </div>
                        <div className="mt-3 flex items-center justify-between">
                          <div className="flex flex-wrap gap-1.5">
                            {t.tags?.slice(0, 3).map((tag: string) => (
                              <span key={tag} className="px-2 py-0.5 bg-slate-700 text-[11px] rounded-full text-slate-300">{tag}</span>
                            ))}
                          </div>
                          <div className="flex items-center gap-1.5" onClick={e => e.stopPropagation()}>
                            <button
                              onClick={() => handleOpenExport(t.id, 'dsd')}
                              title="Export MATLAB DSD (.m)"
                              className="px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded text-[11px] font-mono text-amber-300 flex items-center gap-1 transition"
                            >
                              <Code size={11} /> DSD
                            </button>
                            <button
                              onClick={() => handleOpenExport(t.id, 'opendrive')}
                              title="Export OpenDRIVE (.xodr)"
                              className="px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded text-[11px] font-mono text-sky-300 flex items-center gap-1 transition"
                            >
                              <FileCode size={11} /> .xodr
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Recent scenarios with instant export buttons */}
              <div>
                <h2 className="text-lg font-semibold text-slate-200 mb-3 flex items-center gap-2">
                  <Clock size={18} className="text-sky-400" /> Recent Scenarios & Simulation Workspace
                </h2>
                {scenarios.length === 0 ? (
                  <div className="border border-dashed border-slate-700 rounded-xl p-10 text-center text-slate-500">
                    <FileText size={32} className="mx-auto mb-3 opacity-40" />
                    <p className="text-sm">No scenarios yet in database.</p>
                    <button onClick={() => setActiveTab('create')}
                      className="mt-3 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-white text-xs font-medium transition">
                      Create or Import scenario →
                    </button>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {[...scenarios].reverse().slice(0, 8).map((s: any) => (
                      <div key={s.id}
                        className="flex items-center justify-between px-4 py-3 bg-slate-800/60 border border-slate-700/60 rounded-xl hover:border-slate-600 transition">
                        <div
                          className="flex-1 cursor-pointer"
                          onClick={() => {
                            fetch(`/api/v1/scenarios/${s.id}`)
                              .then(r => r.json())
                              .then(sc => { setGeneratedScenario(sc); setAssumptions([]); setActiveTab('review'); })
                              .catch(() => {});
                          }}
                        >
                          <p className="text-sm font-medium text-slate-200 hover:text-emerald-400 transition">{s.name ?? s.id}</p>
                          <p className="text-xs text-slate-500 mt-0.5">
                            {s.created_at ? new Date(s.created_at).toLocaleString() : ''} &nbsp;·&nbsp;
                            <span className="capitalize">{s.source}</span>
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => handleOpenExport(s.id, 'dsd')}
                            className="px-2.5 py-1 bg-slate-700 hover:bg-slate-600 rounded-lg text-xs font-mono text-amber-300 flex items-center gap-1.5 transition"
                            title="Export MATLAB DSD (.m)"
                          >
                            <Code size={13} /> DSD (.m)
                          </button>
                          <button
                            onClick={() => handleOpenExport(s.id, 'opendrive')}
                            className="px-2.5 py-1 bg-slate-700 hover:bg-slate-600 rounded-lg text-xs font-mono text-sky-300 flex items-center gap-1.5 transition"
                            title="Export RoadRunner OpenDRIVE (.xodr)"
                          >
                            <FileCode size={13} /> OpenDRIVE (.xodr)
                          </button>
                          <button
                            onClick={() => {
                              fetch(`/api/v1/scenarios/${s.id}`)
                                .then(r => r.json())
                                .then(sc => { setGeneratedScenario(sc); setAssumptions([]); setActiveTab('review'); })
                                .catch(() => {});
                            }}
                            className="p-1.5 text-slate-400 hover:text-white transition"
                            title="Open Scenario in Reviewer"
                          >
                            <ChevronRight size={16} />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── CREATE (NLP + OpenStreetMap) ── */}
          {activeTab === 'create' && (
            <div className="max-w-3xl mx-auto space-y-6">
              <div>
                <h1 className="text-2xl font-bold text-white">Create New Scenario</h1>
                <p className="text-slate-400 mt-1 text-sm">
                  Import real-world Indian road coordinates via OpenStreetMap or synthesize challenging edge-case situations using Gemini.
                </p>
              </div>

              {/* Mode Selector Tabs */}
              <div className="flex p-1 bg-slate-900 border border-slate-800 rounded-xl">
                <button
                  onClick={() => setCreateMode('osm')}
                  className={`flex-1 py-2.5 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition ${
                    createMode === 'osm'
                      ? 'bg-emerald-600 text-white shadow-lg'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <Globe size={16} /> Import from OpenStreetMap (Live GPS)
                </button>
                <button
                  onClick={() => setCreateMode('nlp')}
                  className={`flex-1 py-2.5 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition ${
                    createMode === 'nlp'
                      ? 'bg-emerald-600 text-white shadow-lg'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <FileText size={16} /> Natural Language Prompt (Gemini AI)
                </button>
              </div>

              {/* OSM Import Mode */}
              {createMode === 'osm' && (
                <div className="space-y-6">
                  <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6 space-y-5">
                    <div>
                      <h2 className="text-base font-semibold text-white flex items-center gap-2">
                        <Compass size={18} className="text-emerald-400" /> Geographic GPS Coordinates & Search Radius
                      </h2>
                      <p className="text-xs text-slate-400 mt-1">
                        Queries the Overpass API for real road geometries, maps Indian roadway types, and attaches synchronized live weather.
                      </p>
                    </div>

                    {/* Presets */}
                    <div>
                      <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Famous Indian Presets</p>
                      <div className="grid grid-cols-2 gap-2.5">
                        {OSM_PRESETS.map((p) => (
                          <button
                            key={p.name}
                            type="button"
                            onClick={() => {
                              setOsmLat(p.lat.toString());
                              setOsmLon(p.lon.toString());
                            }}
                            className={`text-left p-3 rounded-xl border transition group ${
                              osmLat === p.lat.toString() && osmLon === p.lon.toString()
                                ? 'bg-emerald-500/10 border-emerald-500 text-emerald-300'
                                : 'bg-slate-900/60 border-slate-700/80 text-slate-300 hover:border-slate-500'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <span className="font-semibold text-xs flex items-center gap-1.5">
                                <MapPin size={13} className="text-emerald-400" /> {p.name}
                              </span>
                              <span className="text-[10px] font-mono text-slate-500">{p.lat.toFixed(2)}, {p.lon.toFixed(2)}</span>
                            </div>
                            <p className="text-[11px] text-slate-400 mt-1 line-clamp-1">{p.desc}</p>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Coordinates Inputs */}
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <label className="block text-xs font-medium text-slate-400 mb-1.5">Latitude (°N)</label>
                        <input
                          type="number"
                          step="0.0001"
                          value={osmLat}
                          onChange={e => setOsmLat(e.target.value)}
                          className="w-full bg-slate-900 border border-slate-600 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 font-mono focus:border-emerald-500 focus:outline-none"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-slate-400 mb-1.5">Longitude (°E)</label>
                        <input
                          type="number"
                          step="0.0001"
                          value={osmLon}
                          onChange={e => setOsmLon(e.target.value)}
                          className="w-full bg-slate-900 border border-slate-600 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 font-mono focus:border-emerald-500 focus:outline-none"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-slate-400 mb-1.5">Radius (meters)</label>
                        <input
                          type="number"
                          step="50"
                          min="100"
                          max="1500"
                          value={osmRadius}
                          onChange={e => setOsmRadius(e.target.value)}
                          className="w-full bg-slate-900 border border-slate-600 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 font-mono focus:border-emerald-500 focus:outline-none"
                        />
                      </div>
                    </div>

                    {/* Weather Checkbox */}
                    <div className="flex items-center gap-2 pt-1">
                      <input
                        type="checkbox"
                        id="attachWeather"
                        checked={osmAttachWeather}
                        onChange={e => setOsmAttachWeather(e.target.checked)}
                        className="rounded border-slate-600 text-emerald-600 focus:ring-emerald-500 bg-slate-900 h-4 w-4"
                      />
                      <label htmlFor="attachWeather" className="text-xs text-slate-300 flex items-center gap-1.5 cursor-pointer">
                        <CloudSun size={14} className="text-amber-400" />
                        Synchronize with Live Climate / Fallback Physical Model (Wetness & Visibility)
                      </label>
                    </div>

                    {/* Action Button */}
                    <div className="pt-2">
                      <button
                        onClick={handleImportOSM}
                        className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 rounded-xl text-white font-medium text-sm transition flex items-center justify-center gap-2 shadow-lg"
                      >
                        <Globe size={16} /> Fetch OSM Road & Weather
                      </button>
                    </div>
                  </div>

                  {/* Indian Road Assets Reference Palette */}
                  <div className="bg-slate-800/60 border border-slate-700/80 rounded-2xl p-5 space-y-3">
                    <div className="flex items-center justify-between">
                      <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                        <Box size={14} className="text-amber-400" /> Supported Indian Road Assets
                      </p>
                      <button
                        type="button"
                        onClick={() => setActiveTab('assets')}
                        className="text-xs text-emerald-400 hover:text-emerald-300 transition flex items-center gap-1"
                      >
                        Browse Full 3D Catalogue <ChevronRight size={13} />
                      </button>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {[
                        { label: 'Pothole (Khadha)', color: 'bg-slate-700 text-slate-300 border-slate-600' },
                        { label: 'Police Barricade', color: 'bg-rose-950/60 text-rose-300 border-rose-700/50' },
                        { label: 'Construction Barrier', color: 'bg-orange-950/60 text-orange-300 border-orange-700/50' },
                        { label: 'Desi Cow', color: 'bg-amber-950/60 text-amber-300 border-amber-700/50' },
                        { label: 'Water Buffalo', color: 'bg-amber-900/40 text-amber-200 border-amber-600/50' },
                        { label: 'Auto-Rickshaw', color: 'bg-emerald-950/60 text-emerald-300 border-emerald-700/50' },
                        { label: 'E-Rickshaw', color: 'bg-yellow-950/60 text-yellow-300 border-yellow-700/50' },
                        { label: 'Pushcart (Thela)', color: 'bg-blue-950/60 text-blue-300 border-blue-700/50' },
                        { label: 'Two-Wheeler', color: 'bg-teal-950/60 text-teal-300 border-teal-700/50' },
                      ].map((badge) => (
                        <span
                          key={badge.label}
                          className={`px-2.5 py-1 text-xs rounded-lg border font-medium ${badge.color}`}
                        >
                          {badge.label}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* NLP Prompt Mode */}
              {createMode === 'nlp' && (
                <div className="space-y-6">
                  <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6 space-y-4">
                    <label className="flex items-center gap-2 text-sm font-medium text-slate-300">
                      <FileText size={16} className="text-emerald-400" /> Describe the scenario
                    </label>
                    <textarea
                      className="w-full h-36 bg-slate-900 border border-slate-600 rounded-xl p-4 text-slate-100 text-sm resize-none focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500 transition placeholder:text-slate-600"
                      placeholder="e.g. A cow suddenly crosses a narrow village road at night with fog…"
                      value={nlText}
                      onChange={e => setNlText(e.target.value)}
                      onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleGenerateText(); }}
                    />

                    {/* Indian Road Assets Quick-Insert Chips */}
                    <div>
                      <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                        <Box size={13} className="text-amber-400" /> Insert Indian Road Hazard / Entity
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {[
                          { label: 'Pothole (Khadha)', insert: 'with a deep monsoon pothole in the centre lane' },
                          { label: 'Police Barricade', insert: 'with a police metal barricade checkpoint blocking the lane' },
                          { label: 'Desi Cow', insert: 'a cow suddenly crosses the road from the right' },
                          { label: 'Water Buffalo', insert: 'a slow water buffalo stands stationary in the ego vehicle path' },
                          { label: 'Auto-Rickshaw', insert: 'an auto-rickshaw abruptly cuts in from the left shoulder' },
                          { label: 'Pushcart / Thela', insert: 'a street vendor pushcart occupies half the road' },
                          { label: 'Two-Wheeler Cut-In', insert: 'a commuter motorcycle filters lanes and cuts in ahead' },
                          { label: 'Construction Barrier', insert: 'a heavy construction barrier narrows the carriageway' },
                        ].map((chip) => (
                          <button
                            key={chip.label}
                            type="button"
                            onClick={() => {
                              setNlText(prev => prev.trim() ? `${prev.trim()}. ${chip.insert}.` : chip.insert);
                            }}
                            className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 hover:border-amber-400/50 rounded-lg text-xs text-slate-300 flex items-center gap-1.5 transition"
                          >
                            <PlusCircle size={12} className="text-amber-400" />
                            {chip.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-1">
                      <span className="text-xs text-slate-600">Ctrl+Enter to submit</span>
                      <button
                        onClick={handleGenerateText}
                        disabled={!nlText.trim()}
                        className="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-xl text-white font-medium text-sm transition flex items-center gap-2"
                      >
                        <Play size={15} /> Interpret Scenario
                      </button>
                    </div>
                  </div>

                  {/* Example prompts */}
                  <div>
                    <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Try an example</p>
                    <div className="space-y-2">
                      {EXAMPLE_PROMPTS.map((p, i) => (
                        <button key={i} onClick={() => setNlText(p)}
                          className="w-full text-left px-4 py-3 bg-slate-800/50 border border-slate-700 hover:border-emerald-500/50 rounded-xl text-slate-300 text-sm transition group">
                          <span className="text-emerald-500 mr-2 text-xs font-mono">#{i + 1}</span>
                          {p}
                          <ChevronRight size={14} className="inline ml-2 text-slate-600 group-hover:text-emerald-400 transition" />
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── REVIEW ── */}
          {activeTab === 'review' && !generatedScenario && (
            <div className="flex flex-col items-center justify-center min-h-96 text-slate-500 gap-4">
              <FileText size={48} className="opacity-20" />
              <p>No scenario loaded yet.</p>
              <button onClick={() => setActiveTab('create')}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-white text-sm font-medium transition">
                Create or Import scenario →
              </button>
            </div>
          )}

          {activeTab === 'review' && generatedScenario && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-2xl font-bold text-white">{generatedScenario.metadata.name}</h1>
                  <p className="text-slate-400 mt-1 text-sm">{generatedScenario.metadata.description}</p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleOpenExport(generatedScenario.scenario_id, 'dsd')}
                    className="px-3.5 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-amber-300 text-sm font-medium flex items-center gap-2 transition"
                    title="Export MATLAB DSD script"
                  >
                    <Code size={15} /> Export DSD (.m)
                  </button>
                  <button
                    onClick={() => handleOpenExport(generatedScenario.scenario_id, 'opendrive')}
                    className="px-3.5 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-sky-300 text-sm font-medium flex items-center gap-2 transition"
                    title="Export OpenDRIVE XML"
                  >
                    <FileCode size={15} /> Export RoadRunner (.xodr)
                  </button>
                  <button onClick={() => setActiveTab('create')}
                    className="px-3 py-2 border border-slate-700 hover:bg-slate-800 rounded-lg text-sm flex items-center gap-1.5 transition text-slate-300">
                    <ArrowLeft size={14} /> Back
                  </button>
                  <button onClick={() => handleRun(generatedScenario)}
                    className="px-5 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-white font-medium text-sm flex items-center gap-2 transition shadow-lg">
                    <Play size={14} fill="currentColor" /> Run Simulation
                  </button>
                </div>
              </div>

              {/* Tags & OSM info badges */}
              <div className="flex flex-wrap items-center gap-2">
                {generatedScenario.metadata.source === 'osm' && (
                  <span className="px-2.5 py-1 bg-emerald-950/60 border border-emerald-700/60 text-xs rounded-full text-emerald-300 flex items-center gap-1 font-mono">
                    <Globe size={12} /> OpenStreetMap GPS Sourced
                  </span>
                )}
                {osmSummary?.weather && (
                  <span className="px-2.5 py-1 bg-sky-950/60 border border-sky-700/60 text-xs rounded-full text-sky-300 flex items-center gap-1">
                    <CloudSun size={12} /> Live Climate: {osmSummary.weather.condition} ({osmSummary.weather.temperature_c}°C)
                  </span>
                )}
                {generatedScenario.metadata.tags?.map(tag => (
                  <span key={tag} className="px-2.5 py-1 bg-slate-800 border border-slate-700 text-xs rounded-full text-slate-300">{tag}</span>
                ))}
              </div>

              {/* Bird's-eye canvas */}
              <div>
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Scenario Preview</p>
                <ScenarioCanvas scenario={generatedScenario} />
              </div>

              {/* Info cards */}
              <div className="grid grid-cols-3 gap-4">
                {/* Road */}
                <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
                  <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                    🛣️ Road
                  </p>
                  <dl className="space-y-2 text-sm">
                    <div className="flex justify-between"><dt className="text-slate-400">Type</dt><dd className="text-slate-200 capitalize">{generatedScenario.road.type.replace(/_/g, ' ')}</dd></div>
                    <div className="flex justify-between"><dt className="text-slate-400">Length</dt><dd className="text-slate-200">{generatedScenario.road.length_m}m</dd></div>
                    <div className="flex justify-between"><dt className="text-slate-400">Lane width</dt><dd className="text-slate-200">±{generatedScenario.road.lane_half_width_m}m</dd></div>
                    {generatedScenario.road.surface && (
                      <div className="flex justify-between"><dt className="text-slate-400">Surface</dt><dd className="text-slate-200 capitalize">{generatedScenario.road.surface}</dd></div>
                    )}
                  </dl>
                </div>

                {/* Environment */}
                <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
                  <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
                    {TOD_ICONS[generatedScenario.environment?.time_of_day ?? 'day']} Environment
                  </p>
                  <dl className="space-y-2 text-sm">
                    <div className="flex justify-between"><dt className="text-slate-400">Time</dt><dd className="text-slate-200 capitalize">{generatedScenario.environment?.time_of_day ?? 'day'}</dd></div>
                    <div className="flex justify-between">
                      <dt className="text-slate-400">Weather</dt>
                      <dd className="text-slate-200">{WEATHER_ICONS[generatedScenario.environment?.weather ?? 'clear']} {generatedScenario.environment?.weather ?? 'clear'}</dd>
                    </div>
                    {generatedScenario.environment?.visibility_m && (
                      <div className="flex justify-between"><dt className="text-slate-400">Visibility</dt><dd className="text-slate-200">{generatedScenario.environment.visibility_m}m</dd></div>
                    )}
                  </dl>
                </div>

                {/* Ego vehicle */}
                <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
                  <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                    <Car size={13} /> Ego Vehicle
                  </p>
                  <dl className="space-y-2 text-sm">
                    <div className="flex justify-between"><dt className="text-slate-400">Init speed</dt><dd className="text-slate-200">{generatedScenario.ego_vehicle.initial_speed_kmh} km/h</dd></div>
                    <div className="flex justify-between"><dt className="text-slate-400">Goal</dt><dd className="text-slate-200">{generatedScenario.ego_vehicle.goal_x_m}m</dd></div>
                    <div className="flex justify-between">
                      <dt className="text-slate-400">Mode</dt>
                      <dd className={`capitalize font-medium ${generatedScenario.ego_vehicle.mode === 'adaptive' ? 'text-emerald-400' : 'text-amber-400'}`}>
                        {generatedScenario.ego_vehicle.mode}
                      </dd>
                    </div>
                  </dl>
                </div>
              </div>

              {/* Actors table */}
              <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
                <div className="px-5 py-3 border-b border-slate-700 flex items-center gap-2">
                  <Shield size={15} className="text-amber-400" />
                  <p className="text-sm font-semibold text-slate-200">Actors ({generatedScenario.actors.length})</p>
                </div>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-xs text-slate-500 uppercase tracking-wider border-b border-slate-700/50">
                      <th className="text-left px-5 py-2.5">Class</th>
                      <th className="text-left px-5 py-2.5">Position (x, y)</th>
                      <th className="text-left px-5 py-2.5">Velocity (vx, vy)</th>
                      <th className="text-left px-5 py-2.5">Behaviour</th>
                    </tr>
                  </thead>
                  <tbody>
                    {generatedScenario.actors.map((a, i) => {
                      const aClass = getActorClass(a);
                      return (
                        <tr key={a.id} className={i % 2 === 0 ? 'bg-slate-900/30' : ''}>
                          <td className="px-5 py-2.5">
                            <span className="flex items-center gap-2">
                              <span className="w-2.5 h-2.5 rounded-full" style={{ background: ACTOR_COLORS[aClass] ?? '#94a3b8' }} />
                              <span className="capitalize text-slate-200">{aClass.replace(/_/g, ' ')}</span>
                            </span>
                          </td>
                          <td className="px-5 py-2.5 font-mono text-slate-400 text-xs">{a.initial_x_m}, {a.initial_y_m}</td>
                          <td className="px-5 py-2.5 font-mono text-slate-400 text-xs">{a.initial_vx_mps ?? 0}, {a.initial_vy_mps ?? 0} m/s</td>
                          <td className="px-5 py-2.5">
                            <span className="px-2 py-0.5 bg-slate-700 rounded-full text-xs text-slate-300 capitalize">{a.behaviour.replace(/_/g, ' ')}</span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Indian Road Assets Footprint Badges */}
              <div className="bg-slate-800/80 border border-slate-700 rounded-xl p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                    <Box size={14} className="text-amber-400" /> Indian 3D Asset Library Footprint
                  </p>
                  <button
                    onClick={() => setActiveTab('assets')}
                    className="text-xs text-emerald-400 hover:text-emerald-300 transition flex items-center gap-1"
                  >
                    View All 14 Assets <ChevronRight size={13} />
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  {generatedScenario.actors.map(actor => {
                    const aClass = getActorClass(actor);
                    const matchedAsset = assetCatalog?.assets?.find(x => x.class === aClass || x.id === aClass);
                    const color = ACTOR_COLORS[aClass] ?? '#94a3b8';
                    return (
                      <div key={actor.id} className="p-3 bg-slate-900 border border-slate-700/80 rounded-lg flex items-start gap-3">
                        <div className="w-3 h-3 rounded-full mt-1 shrink-0" style={{ background: color }} />
                        <div className="flex-1">
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-semibold text-white capitalize">{matchedAsset?.name || aClass.replace(/_/g, ' ')}</span>
                            <span className="text-[10px] font-mono px-1.5 py-0.5 bg-slate-800 rounded text-slate-400">{matchedAsset?.category || 'Hazard'}</span>
                          </div>
                          <p className="text-[11px] text-slate-400 font-mono mt-0.5">
                            3D Box: {matchedAsset?.dimensions ? `${matchedAsset.dimensions.length_m}m × ${matchedAsset.dimensions.width_m}m × ${matchedAsset.dimensions.height_m}m` : 'Custom actor geometry'}
                          </p>
                          <p className="text-[10px] text-slate-500 mt-1 line-clamp-1">{matchedAsset?.description || 'Standard simulation traffic participant'}</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Assumptions */}
              {assumptions.length > 0 && (
                <div className="bg-amber-950/30 border border-amber-700/40 rounded-xl p-5">
                  <p className="text-sm font-semibold text-amber-400 mb-3 flex items-center gap-2">
                    <AlertTriangle size={15} /> Generation Assumptions & Data Context
                  </p>
                  <ul className="space-y-1.5">
                    {assumptions.map((a, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-amber-200/80">
                        <CheckCircle size={14} className="text-amber-500 shrink-0 mt-0.5" /> {a}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Raw JSON toggle */}
              <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
                <button onClick={() => setJsonVisible(v => !v)}
                  className="w-full flex items-center justify-between px-5 py-3 text-sm text-slate-300 hover:bg-slate-700/50 transition">
                  <span className="flex items-center gap-2"><Cpu size={15} /> Raw JSON Schema</span>
                  <div className="flex items-center gap-3">
                    <button onClick={e => { e.stopPropagation(); handleCopyJSON(); }}
                      className="flex items-center gap-1.5 text-xs px-2.5 py-1 bg-slate-700 hover:bg-slate-600 rounded-lg transition">
                      <Copy size={12} /> {copied ? 'Copied!' : 'Copy'}
                    </button>
                    <span className="text-slate-500 text-xs">{jsonVisible ? '▲ hide' : '▼ show'}</span>
                  </div>
                </button>
                {jsonVisible && (
                  <pre className="bg-slate-950 px-5 py-4 text-xs text-slate-300 overflow-auto max-h-72 border-t border-slate-700">
                    {JSON.stringify(generatedScenario, null, 2)}
                  </pre>
                )}
              </div>
            </div>
          )}

          {/* ── RESULTS ── */}
          {activeTab === 'results' && !result && (
            <div className="flex flex-col items-center justify-center min-h-96 text-slate-500 gap-4">
              <Activity size={48} className="opacity-20" />
              <p>No simulation results yet.</p>
              {generatedScenario && (
                <button onClick={() => handleRun(generatedScenario)}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-white text-sm font-medium transition flex items-center gap-2">
                  <Play size={14} /> Run Simulation
                </button>
              )}
            </div>
          )}

          {activeTab === 'results' && result && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h1 className="text-2xl font-bold text-white">Simulation Results</h1>
                <button onClick={handleReset}
                  className="px-3 py-2 border border-slate-600 hover:bg-slate-800 rounded-lg text-sm flex items-center gap-2 transition">
                  <RotateCcw size={14} /> New Scenario
                </button>
              </div>

              {/* Outcome banner */}
              <div className={`rounded-2xl p-6 flex items-center gap-6 border ${result.arrived ? 'bg-emerald-950/40 border-emerald-700/50' : 'bg-rose-950/40 border-rose-700/50'}`}>
                {result.arrived
                  ? <CheckCircle size={52} className="text-emerald-400 shrink-0" />
                  : <XCircle size={52} className="text-rose-400 shrink-0" />}
                <div>
                  <p className="text-xl font-bold text-white">{result.explanation?.outcome_summary}</p>
                  <p className="text-sm text-slate-400 mt-1">
                    Mode: <span className="text-white capitalize">{result.mode}</span> &nbsp;·&nbsp;
                    Duration: <span className="text-white">{result.t_total_s}s</span> &nbsp;·&nbsp;
                    Collisions: <span className={result.collisions > 0 ? 'text-rose-400' : 'text-emerald-400'}>{result.collisions}</span>
                  </p>
                  {result.explanation?.failure_reason && (
                    <p className="mt-2 text-sm text-rose-300 flex items-center gap-2">
                      <AlertTriangle size={14} /> {result.explanation.failure_reason}
                    </p>
                  )}
                </div>
              </div>

              {/* Metrics grid */}
              <div className="grid grid-cols-3 gap-4">
                <MetricCard label="Min Clearance" value={result.min_clearance_m?.toFixed(2)} unit="m" good={result.min_clearance_m > 1} />
                <MetricCard label="Replans" value={result.replans} good={result.replans < 10} />
                <MetricCard label="Avg Replan" value={result.mean_replan_ms?.toFixed(1)} unit="ms" good={result.mean_replan_ms < 50} />
                <MetricCard label="Max Replan" value={result.max_replan_ms?.toFixed(1)} unit="ms" />
                <MetricCard label="Mean Speed" value={result.mean_speed_kmh?.toFixed(1)} unit="km/h" />
                <MetricCard label="Max Speed" value={result.max_speed_kmh?.toFixed(1)} unit="km/h" />
              </div>

              {/* Trajectory */}
              <div className="bg-slate-800 border border-slate-700 rounded-xl p-5">
                <p className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                  <Gauge size={15} className="text-sky-400" /> Ego Trajectory
                </p>
                <TrajectoryCanvas traj_x={result.traj_x} traj_y={result.traj_y} scenario={generatedScenario} />
              </div>

              {/* Decision distribution */}
              <div className="bg-slate-800 border border-slate-700 rounded-xl p-5">
                <p className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
                  <Activity size={15} className="text-emerald-400" /> Decision Distribution
                </p>
                <DecisionBar result={result} />
              </div>

              {/* Critical events */}
              {result.explanation?.critical_events?.length > 0 && (
                <div className="bg-slate-800 border border-slate-700 rounded-xl p-5">
                  <p className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
                    <AlertTriangle size={15} className="text-amber-400" /> Critical Events
                  </p>
                  <ul className="space-y-2">
                    {result.explanation.critical_events.map((ev, i) => (
                      <li key={i} className="flex items-start gap-3 text-sm text-slate-300">
                        <span className="w-5 h-5 shrink-0 bg-rose-500/20 text-rose-400 rounded-full flex items-center justify-center text-xs font-bold">{i + 1}</span>
                        {ev}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* ── 3D ASSETS LIBRARY PALETTE ── */}
          {activeTab === 'assets' && (
            <div className="space-y-8">
              <div>
                <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
                  <Box size={24} className="text-amber-400" /> Indian Road 3D Asset Library
                </h1>
                <p className="text-slate-400 mt-1 text-sm">
                  3D bounding boxes, collision classifications, and radar reflectivity profiles for authentic Indian road entities.
                </p>
              </div>

              {assetCatalog?.categories && (
                <div className="space-y-8">
                  {Object.entries(assetCatalog.categories).map(([category, items]) => (
                    <div key={category} className="space-y-3">
                      <h2 className="text-base font-bold text-slate-200 border-b border-slate-800 pb-2 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-emerald-400" /> {category} ({items.length})
                      </h2>
                      <div className="grid grid-cols-2 gap-4">
                        {items.map(asset => {
                          const color = ACTOR_COLORS[asset.class] ?? '#94a3b8';
                          return (
                            <div key={asset.id} className="p-4 bg-slate-800 border border-slate-700/80 rounded-xl flex flex-col justify-between hover:border-slate-600 transition">
                              <div>
                                <div className="flex items-start justify-between">
                                  <div className="flex items-center gap-2">
                                    <span className="w-3 h-3 rounded-full shrink-0" style={{ background: color }} />
                                    <h3 className="font-semibold text-white text-sm">{asset.name}</h3>
                                  </div>
                                  <span className="text-[10px] font-mono px-2 py-0.5 bg-slate-900 border border-slate-700 rounded text-slate-400 capitalize">
                                    {asset.collision_classification.replace(/_/g, ' ')}
                                  </span>
                                </div>
                                <p className="text-xs text-slate-400 mt-2.5 leading-relaxed">{asset.description}</p>
                              </div>
                              <div className="mt-4 pt-3 border-t border-slate-700/60 flex items-center justify-between text-xs text-slate-400 font-mono">
                                <div>
                                  <span className="text-slate-500">Box: </span>
                                  {asset.dimensions.length_m}m × {asset.dimensions.width_m}m × {asset.dimensions.height_m}m
                                </div>
                                <div>
                                  <span className="text-slate-500">RCS: </span>
                                  <span className="text-amber-400">{asset.radar_cross_section || 'medium'}</span>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

        </div>
      </main>
    </div>
  );
}
