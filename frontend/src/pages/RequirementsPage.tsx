import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ClipboardList, MapPin, Calendar, Sparkles } from 'lucide-react';
import { fetchApi } from '../api/client';
import { Requirement } from '../types';
import { ErrorBanner, EmptyState } from '../components/ErrorBanner';
import { useAuth } from '../context/AuthContext';

export const RequirementsPage: React.FC = () => {
  const [requirements, setRequirements] = useState<Requirement[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const { user } = useAuth();

  useEffect(() => {
    if (!user?.company_id) return;

    fetchApi<{ items: Requirement[] }>(`/requirements?company_id=${user.company_id}`)
      .then((data) => setRequirements(data.items))
      .catch((err) => setError(err.message || 'Failed to load requirements'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Material Requirements</h1>
          <p className="text-xs text-industrial-400">Your material demands — use AI to find the best sellers</p>
        </div>
      </div>

      {error && <ErrorBanner message={error} />}
      {!loading && !error && requirements.length === 0 && (
        <EmptyState message="No material requirements posted yet." />
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {requirements.map((item) => (
          <div key={item.id} className="bg-industrial-900 border border-industrial-800 rounded-2xl p-5 shadow-sm hover:border-industrial-700 transition-all flex flex-col justify-between">
            <div>
              <div className="flex items-start justify-between gap-3 mb-3">
                <div>
                  <span className="bg-purple-500/10 border border-purple-500/20 text-purple-400 text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider">
                    {item.status} DEMAND
                  </span>
                  <h3 className="font-bold text-white text-lg mt-1">{item.material_name}</h3>
                  <p className="text-xs text-industrial-400 flex items-center gap-1 mt-0.5">
                    <MapPin className="w-3.5 h-3.5 text-industrial-400" />
                    <span>{item.plant_name} ({item.company_name})</span>
                  </p>
                </div>
                <div className="text-right">
                  <span className="text-xl font-black text-white">{item.quantity}</span>
                  <span className="text-xs text-industrial-400 block uppercase font-medium">{item.unit}</span>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2 my-3 text-xs bg-industrial-950/60 p-2.5 rounded-xl border border-industrial-800/80">
                <div>
                  <span className="text-[10px] text-industrial-400 uppercase font-semibold block">Min Purity</span>
                  <span className="font-semibold text-white">
                    {item.minimum_purity != null ? `${item.minimum_purity}%` : 'Not specified'}
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-industrial-400 uppercase font-semibold block">Max Budget</span>
                  <span className="font-semibold text-white">
                    {item.maximum_budget_per_unit != null ? `₹${item.maximum_budget_per_unit}` : 'Open'}
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-industrial-400 uppercase font-semibold block">Max Distance</span>
                  <span className="font-semibold text-white">
                    {item.preferred_max_distance_km != null ? `${item.preferred_max_distance_km} km` : 'Any'}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between border-t border-industrial-800/80 pt-3.5 mt-2">
              <span className="text-[11px] text-industrial-400 flex items-center gap-1">
                <Calendar className="w-3.5 h-3.5" />
                <span>Required before: {item.required_before}</span>
              </span>

              <button
                onClick={() => navigate(`/recommendations?requirement_id=${item.id}`)}
                className="bg-eco-600 hover:bg-eco-500 text-white font-bold text-xs px-3.5 py-2 rounded-xl transition-all shadow-md shadow-eco-600/20 flex items-center gap-1.5"
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Find Suitable Sellers</span>
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
