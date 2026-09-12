import React, { useEffect, useState } from 'react';
import { ShieldCheck, Building2, Factory, Trash2, Layers, CheckCircle2, XCircle, Cpu, AlertTriangle } from 'lucide-react';
import { fetchApi } from '../api/client';
import { AIRuntimeStatus, Company, PlatformAnalytics } from '../types';
import { ErrorBanner } from '../components/ErrorBanner';

export const AdminPage: React.FC = () => {
  const [companies, setCompanies] = useState<Company[]>([]);
  const [platform, setPlatform] = useState<PlatformAnalytics | null>(null);
  const [aiStatus, setAiStatus] = useState<AIRuntimeStatus | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchApi<{ items: Company[] }>('/companies').then((d) => setCompanies(d.items)).catch((err) => setError(err.message || 'Failed to load companies'));
    fetchApi<PlatformAnalytics>('/analytics/platform').then(setPlatform).catch((err) => setError(err.message || 'Failed to load platform analytics'));
    fetchApi<AIRuntimeStatus>('/analytics/ai-status').then(setAiStatus).catch((err) => setError(err.message || 'Failed to load AI runtime status'));
  }, []);

  const gnnLive = Boolean(aiStatus?.model_loaded && aiStatus?.graph_cached);
  const cacheAge =
    aiStatus?.age_seconds == null
      ? 'never built'
      : aiStatus.age_seconds < 60
      ? `${Math.round(aiStatus.age_seconds)}s ago`
      : `${Math.round(aiStatus.age_seconds / 60)}m ago`;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Platform Admin Panel</h1>
          <p className="text-xs text-industrial-400">Verify company credentials, monitor ecosystem statistics, and AI health</p>
        </div>
      </div>

      {error && <ErrorBanner message={error} />}

      {/* Overview Cards */}
      {platform && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="bg-industrial-900 border border-industrial-800 p-4 rounded-2xl">
            <span className="text-[10px] text-industrial-400 font-semibold uppercase block">Registered Companies</span>
            <span className="text-2xl font-black text-white mt-1 block">{platform.total_companies}</span>
          </div>
          <div className="bg-industrial-900 border border-industrial-800 p-4 rounded-2xl">
            <span className="text-[10px] text-industrial-400 font-semibold uppercase block">Operating Plants</span>
            <span className="text-2xl font-black text-white mt-1 block">{platform.total_plants}</span>
          </div>
          <div className="bg-industrial-900 border border-industrial-800 p-4 rounded-2xl">
            <span className="text-[10px] text-industrial-400 font-semibold uppercase block">Active Listings</span>
            <span className="text-2xl font-black text-white mt-1 block">{platform.total_waste_listings}</span>
          </div>
          <div className="bg-industrial-900 border border-industrial-800 p-4 rounded-2xl">
            <span className="text-[10px] text-industrial-400 font-semibold uppercase block">Completed Exchanges</span>
            <span className="text-2xl font-black text-eco-400 mt-1 block">{platform.total_exchanges}</span>
          </div>
        </div>
      )}

      {/* AI Runtime Status — the real state of inference, not what the UI claims */}
      {aiStatus && (
        <div className="bg-industrial-900 border border-industrial-800 rounded-2xl p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              <Cpu className="w-4 h-4 text-eco-400" />
              <span>AI Runtime Status</span>
            </h2>
            <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wider ${
              gnnLive
                ? 'bg-eco-500/20 text-eco-400 border border-eco-500/30'
                : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
            }`}>
              {gnnLive ? <CheckCircle2 className="w-3 h-3" /> : <AlertTriangle className="w-3 h-3" />}
              <span>{gnnLive ? 'MC-GNN serving' : 'Baseline fallback'}</span>
            </span>
          </div>

          {!gnnLive && (
            <p className="text-xs text-amber-400/90">
              Recommendations are being scored by the rule-based baseline, not the neural model.
            </p>
          )}

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-industrial-950 p-3 rounded-xl border border-industrial-800">
              <span className="text-[10px] text-industrial-400 uppercase font-semibold block">Checkpoint</span>
              <span className={`text-sm font-bold mt-0.5 block ${aiStatus.model_loaded ? 'text-eco-400' : 'text-amber-400'}`}>
                {aiStatus.model_loaded ? 'Loaded' : 'Not loaded'}
              </span>
            </div>
            <div className="bg-industrial-950 p-3 rounded-xl border border-industrial-800">
              <span className="text-[10px] text-industrial-400 uppercase font-semibold block">Graph Cache</span>
              <span className="text-sm font-bold text-white mt-0.5 block">
                {aiStatus.graph_cached ? `${aiStatus.node_count} nodes` : 'Empty'}
              </span>
              <span className="text-[11px] text-industrial-400">{aiStatus.edge_count} edges</span>
            </div>
            <div className="bg-industrial-950 p-3 rounded-xl border border-industrial-800">
              <span className="text-[10px] text-industrial-400 uppercase font-semibold block">Last Rebuilt</span>
              <span className="text-sm font-bold text-white mt-0.5 block">{cacheAge}</span>
              <span className="text-[11px] text-industrial-400">refresh every {Math.round(aiStatus.ttl_seconds / 60)}m</span>
            </div>
            <div className="bg-industrial-950 p-3 rounded-xl border border-industrial-800">
              <span className="text-[10px] text-industrial-400 uppercase font-semibold block">Last Error</span>
              <span className={`text-sm font-bold mt-0.5 block ${aiStatus.last_error ? 'text-red-400' : 'text-eco-400'}`}>
                {aiStatus.last_error ? 'Present' : 'None'}
              </span>
            </div>
          </div>

          {aiStatus.last_error && (
            <p className="text-[11px] text-red-400 font-mono bg-red-500/10 border border-red-500/20 rounded-lg p-2.5 break-all">
              {aiStatus.last_error}
            </p>
          )}
        </div>
      )}

      {/* Company Verification Table */}
      <div className="bg-industrial-900 border border-industrial-800 rounded-2xl p-5 shadow-sm space-y-4">
        <h2 className="text-sm font-bold text-white flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-eco-400" />
          <span>Company Verification & Credentials</span>
        </h2>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-industrial-300">
            <thead className="bg-industrial-950 text-industrial-400 uppercase text-[10px] tracking-wider border-b border-industrial-800">
              <tr>
                <th className="p-3">Company Name</th>
                <th className="p-3">Industry</th>
                <th className="p-3">Registration No</th>
                <th className="p-3">GST No</th>
                <th className="p-3">Trust Score</th>
                <th className="p-3">Verification</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-industrial-800/60">
              {companies.map((c) => (
                <tr key={c.id} className="hover:bg-industrial-950/40 transition-colors">
                  <td className="p-3 font-bold text-white">{c.company_name}</td>
                  <td className="p-3">{c.industry_type}</td>
                  <td className="p-3 font-mono text-[11px]">{c.registration_number}</td>
                  <td className="p-3 font-mono text-[11px]">{c.gst_number}</td>
                  <td className="p-3 font-bold text-amber-400">{c.trust_score.toFixed(1)} / 100</td>
                  <td className="p-3">
                    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                      c.verification_status === 'Verified' ? 'bg-eco-500/20 text-eco-400 border border-eco-500/30' : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                    }`}>
                      {c.verification_status === 'Verified' ? <CheckCircle2 className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
                      <span>{c.verification_status}</span>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
