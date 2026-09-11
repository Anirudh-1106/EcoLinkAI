import React, { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell
} from 'recharts';
import { BarChart3, Leaf, Sparkles, TrendingUp, Layers, Award } from 'lucide-react';
import { fetchApi } from '../api/client';
import { AIModelMetrics, MaterialDistribution, PlatformAnalytics } from '../types';
import { ErrorBanner, EmptyState } from '../components/ErrorBanner';

export const AnalyticsPage: React.FC = () => {
  const [platform, setPlatform] = useState<PlatformAnalytics | null>(null);
  const [distribution, setDistribution] = useState<MaterialDistribution[]>([]);
  const [aiMetrics, setAiMetrics] = useState<AIModelMetrics | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchApi<PlatformAnalytics>('/analytics/company').then(setPlatform).catch((err) => setError(err.message || 'Failed to load analytics'));
    fetchApi<MaterialDistribution[]>('/analytics/materials-distribution').then(setDistribution).catch((err) => setError(err.message || 'Failed to load material distribution'));
    fetchApi<AIModelMetrics>('/analytics/ai-metrics').then(setAiMetrics).catch((err) => setError(err.message || 'Failed to load AI metrics'));
  }, []);

  const COLORS = ['#22c55e', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6'];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Analytics & AI Insights</h1>
          <p className="text-xs text-industrial-400">Your company's performance, carbon impact, and MC-GNN evaluation metrics</p>
        </div>
      </div>

      {error && <ErrorBanner message={error} />}

      {/* Top AI Model Evaluation Card */}
      {aiMetrics && (
        <div className="bg-gradient-to-r from-industrial-900 via-eco-950/40 to-industrial-900 border border-eco-500/30 rounded-3xl p-6 shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-eco-400 text-xs font-bold uppercase tracking-wider">
              <Sparkles className="w-4 h-4" />
              <span>AI Engine Metrics — {aiMetrics.model_version}</span>
            </div>
            <span className="text-xs text-industrial-400">Training Samples: {aiMetrics.training_samples}</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
            <div className="bg-industrial-950 p-3 rounded-2xl border border-industrial-800">
              <span className="text-[10px] text-industrial-400 uppercase font-semibold block">Precision@5</span>
              <span className="text-2xl font-black text-white mt-1 block">{(aiMetrics.precision_at_5 * 100).toFixed(1)}%</span>
            </div>
            <div className="bg-industrial-950 p-3 rounded-2xl border border-industrial-800">
              <span className="text-[10px] text-industrial-400 uppercase font-semibold block">Recall@5</span>
              <span className="text-2xl font-black text-white mt-1 block">{(aiMetrics.recall_at_5 * 100).toFixed(1)}%</span>
            </div>
            <div className="bg-industrial-950 p-3 rounded-2xl border border-industrial-800">
              <span className="text-[10px] text-industrial-400 uppercase font-semibold block">MC-GNN NDCG@5</span>
              <span className="text-2xl font-black text-eco-400 mt-1 block">{(aiMetrics.ndcg_at_5 * 100).toFixed(1)}%</span>
            </div>
            <div className="bg-industrial-950 p-3 rounded-2xl border border-industrial-800">
              <span className="text-[10px] text-industrial-400 uppercase font-semibold block">Baseline NDCG@5</span>
              <span className="text-2xl font-black text-amber-400 mt-1 block">{(aiMetrics.baseline_ndcg_at_5 * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Material Category Distribution Chart */}
        <div className="bg-industrial-900 border border-industrial-800 rounded-2xl p-5 shadow-sm space-y-3">
          <h2 className="text-sm font-bold text-white flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-eco-400" />
            <span>Waste Volume by Material Category</span>
          </h2>

          <div className="h-64 w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={distribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="category" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', fontSize: '12px' }}
                />
                <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                  {distribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Impact Overview */}
        {platform && (
          <div className="bg-industrial-900 border border-industrial-800 rounded-2xl p-5 shadow-sm space-y-4">
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              <Leaf className="w-4 h-4 text-eco-400" />
              <span>Your Carbon & Sustainability Impact</span>
            </h2>

            <div className="space-y-3">
              <div className="bg-industrial-950 p-4 rounded-xl border border-industrial-800 flex items-center justify-between">
                <div>
                  <span className="text-xs text-industrial-400 block font-semibold">Total Waste Exchanged</span>
                  <span className="text-xl font-bold text-white">{Number(platform.total_waste_exchanged_tons).toFixed(1)} Tonnes</span>
                </div>
                <div className="bg-eco-600/10 text-eco-400 border border-eco-500/20 p-2.5 rounded-xl">
                  <BarChart3 className="w-5 h-5" />
                </div>
              </div>

              <div className="bg-industrial-950 p-4 rounded-xl border border-industrial-800 flex items-center justify-between">
                <div>
                  <span className="text-xs text-industrial-400 block font-semibold">Total Carbon Savings</span>
                  <span className="text-xl font-bold text-eco-400">{Number(platform.total_carbon_saved_kg).toFixed(0)} kg CO₂e</span>
                </div>
                <div className="bg-emerald-600/10 text-emerald-400 border border-emerald-500/20 p-2.5 rounded-xl">
                  <Leaf className="w-5 h-5" />
                </div>
              </div>

              <div className="bg-industrial-950 p-4 rounded-xl border border-industrial-800 flex items-center justify-between">
                <div>
                  <span className="text-xs text-industrial-400 block font-semibold">Exchange Success Rate</span>
                  <span className="text-xl font-bold text-blue-400">{platform.exchange_success_rate.toFixed(1)}%</span>
                </div>
                <div className="bg-blue-600/10 text-blue-400 border border-blue-500/20 p-2.5 rounded-xl">
                  <TrendingUp className="w-5 h-5" />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
