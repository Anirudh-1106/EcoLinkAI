import React, { useEffect, useState } from 'react';
import { ArrowRightLeft, Check, X, MapPin, Sparkles, Clock } from 'lucide-react';
import { fetchApi } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { ExchangeRequest } from '../types';
import { ErrorBanner, EmptyState } from '../components/ErrorBanner';

export const ExchangeRequestsPage: React.FC = () => {
  const { user } = useAuth();
  const [requests, setRequests] = useState<ExchangeRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadRequests = () => {
    setLoading(true);
    fetchApi<ExchangeRequest[]>('/exchange-requests')
      .then((data) => {
        setRequests(data);
        setError('');
      })
      .catch((err) => setError(err.message || 'Failed to load exchange requests'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadRequests();
  }, []);

  const handleAction = async (requestId: string, action: 'accept' | 'reject') => {
    try {
      await fetchApi(`/exchange-requests/${requestId}/action`, {
        method: 'POST',
        body: JSON.stringify({ action }),
      });
      loadRequests();
    } catch (err: any) {
      alert(`Action failed: ${err.message}`);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Exchange Requests</h1>
          <p className="text-xs text-industrial-400">Incoming & outgoing AI-recommended waste exchange proposals</p>
        </div>
      </div>

      {error && <ErrorBanner message={error} />}
      {!loading && !error && requests.length === 0 && (
        <EmptyState message="No exchange requests yet. Send one from AI Recommendations to get started." />
      )}

      <div className="space-y-4">
        {requests.map((req) => (
          <div key={req.id} className="bg-industrial-900 border border-industrial-800 rounded-2xl p-5 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div className="space-y-1.5 max-w-xl">
              <div className="flex items-center gap-2">
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider ${
                  req.status === 'Accepted' ? 'bg-eco-500/20 text-eco-400 border border-eco-500/30' :
                  req.status === 'Rejected' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                  'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                }`}>
                  {req.status}
                </span>
                <span className="text-xs font-semibold text-eco-400 flex items-center gap-1">
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>AI Match Score: {req.compatibility_score}%</span>
                </span>
              </div>

              <h3 className="font-bold text-white text-base">
                <strong className="text-eco-400">{req.buyer_company_name}</strong> is requesting to buy this from{' '}
                <strong className="text-white">{req.supplier_company_name}</strong>
              </h3>
              <p className="text-xs text-industrial-300">
                Seller Plant: <strong>{req.supplier_plant_name}</strong> | Buyer Plant: <strong>{req.buyer_plant_name}</strong>
              </p>
              <p className="text-xs text-industrial-400 flex items-center gap-3">
                <span className="flex items-center gap-1"><MapPin className="w-3.5 h-3.5" /> Distance: {req.distance_km} km</span>
                <span>Est Transport: ₹{req.estimated_transport_cost.toLocaleString()}</span>
                <span>Carbon Saving: {req.estimated_carbon_saving} kg CO₂e</span>
              </p>
              {req.recommendation_reason && (
                <p className="text-xs text-industrial-300 italic bg-industrial-950/60 p-2 rounded-lg border border-industrial-800/80 mt-1">
                  "{req.recommendation_reason}"
                </p>
              )}
            </div>

            {req.status === 'Pending' && req.supplier_company_id === user?.company_id && (
              <div className="flex items-center gap-2 shrink-0">
                <button
                  onClick={() => handleAction(req.id, 'accept')}
                  className="bg-eco-600 hover:bg-eco-500 text-white text-xs font-bold px-4 py-2 rounded-xl transition-all shadow-md flex items-center gap-1.5"
                >
                  <Check className="w-4 h-4" />
                  <span>Accept Request</span>
                </button>
                <button
                  onClick={() => handleAction(req.id, 'reject')}
                  className="bg-industrial-800 hover:bg-red-950/50 hover:text-red-400 text-industrial-300 border border-industrial-700 text-xs font-semibold px-4 py-2 rounded-xl transition-all flex items-center gap-1.5"
                >
                  <X className="w-4 h-4" />
                  <span>Reject</span>
                </button>
              </div>
            )}
            {req.status === 'Pending' && req.supplier_company_id !== user?.company_id && (
              <span className="text-[11px] text-industrial-400 italic shrink-0">Awaiting seller's response</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
