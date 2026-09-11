import React, { useEffect, useState } from 'react';
import { Factory, Trash2, ClipboardList, ArrowRightLeft, Truck, Leaf, ShieldCheck, Star, Sparkles } from 'lucide-react';
import { fetchApi } from '../api/client';
import { DashboardStats } from '../types';
import { StatCard } from '../components/StatCard';
import { ErrorBanner } from '../components/ErrorBanner';
import { Link } from 'react-router-dom';

export const DashboardPage: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchApi<DashboardStats>('/analytics/dashboard')
      .then((data) => setStats(data))
      .catch((err) => setError(err.message || 'Failed to load dashboard stats'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      {error && <ErrorBanner message={error} />}
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-eco-950 via-industrial-900 to-industrial-900 border border-eco-500/20 rounded-3xl p-6 sm:p-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-6 shadow-xl">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-eco-600/20 border border-eco-500/30 text-eco-400 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Industrial Symbiosis Network</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-black text-white">Company Operations Overview</h1>
          <p className="text-xs sm:text-sm text-industrial-300 max-w-xl">
            Monitor plants, active waste listings, material requirements, and track real-time environmental carbon savings.
          </p>
        </div>
        <Link
          to="/recommendations"
          className="bg-eco-600 hover:bg-eco-500 text-white text-xs sm:text-sm font-bold px-6 py-3 rounded-xl shadow-lg shadow-eco-600/30 flex items-center gap-2 shrink-0 transition-all"
        >
          <Sparkles className="w-4 h-4" />
          <span>Find Suitable Sellers</span>
        </Link>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Plants"
          value={stats ? stats.total_plants : '...'}
          subtitle="Operating facilities"
          icon={Factory}
          color="blue"
        />
        <StatCard
          title="Active Listings"
          value={stats ? stats.active_waste_listings : '...'}
          subtitle="Supply available"
          icon={Trash2}
          color="amber"
        />
        <StatCard
          title="Open Requirements"
          value={stats ? stats.open_requirements : '...'}
          subtitle="Demand posted"
          icon={ClipboardList}
          color="purple"
        />
        <StatCard
          title="Pending Requests"
          value={stats ? stats.pending_exchange_requests : '...'}
          subtitle="Awaiting action"
          icon={ArrowRightLeft}
          color="eco"
        />
      </div>

      {/* Second Row Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <StatCard
          title="Completed Exchanges"
          value={stats ? stats.completed_exchanges : '...'}
          subtitle="Successful transactions"
          icon={Truck}
          color="emerald"
        />
        <StatCard
          title="Carbon Saved"
          value={stats ? `${Number(stats.total_carbon_saved_kg).toFixed(0)} kg` : '...'}
          subtitle="CO₂e avoided"
          icon={Leaf}
          color="eco"
        />
        <StatCard
          title="Trust & Rating"
          value={stats ? `${stats.trust_score.toFixed(1)} / 100` : '...'}
          subtitle={
            stats && stats.average_rating > 0
              ? `Avg Rating: ${stats.average_rating.toFixed(1)} ★`
              : 'No ratings yet'
          }
          icon={ShieldCheck}
          color="amber"
        />
      </div>
    </div>
  );
};
