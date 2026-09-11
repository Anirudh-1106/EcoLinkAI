import React, { useEffect, useState } from 'react';
import { Trash2, MapPin, Calendar, DollarSign, Tag } from 'lucide-react';
import { fetchApi } from '../api/client';
import { WasteListing } from '../types';
import { ErrorBanner, EmptyState } from '../components/ErrorBanner';
import { useAuth } from '../context/AuthContext';

export const WasteListingsPage: React.FC = () => {
  const [listings, setListings] = useState<WasteListing[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { user } = useAuth();

  useEffect(() => {
    if (!user?.company_id) return;

    fetchApi<{ items: WasteListing[] }>(`/waste-listings?company_id=${user.company_id}`)
      .then((data) => setListings(data.items))
      .catch((err) => setError(err.message || 'Failed to load waste listings'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Waste Listings</h1>
          <p className="text-xs text-industrial-400">Your listed industrial byproducts — buyers will find you through AI recommendations</p>
        </div>
      </div>

      {error && <ErrorBanner message={error} />}
      {!loading && !error && listings.length === 0 && (
        <EmptyState message="No waste listings yet. Create one to start matching with partners." />
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {listings.map((item) => (
          <div key={item.id} className="bg-industrial-900 border border-industrial-800 rounded-2xl p-5 shadow-sm hover:border-industrial-700 transition-all flex flex-col justify-between">
            <div>
              <div className="flex items-start justify-between gap-3 mb-3">
                <div>
                  <span className="bg-eco-500/10 border border-eco-500/20 text-eco-400 text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider">
                    {item.status}
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
                  <span className="text-[10px] text-industrial-400 uppercase font-semibold block">Purity</span>
                  <span className="font-semibold text-white">
                    {item.purity_percentage != null ? `${item.purity_percentage}%` : 'Not specified'}
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-industrial-400 uppercase font-semibold block">Grade</span>
                  <span className="font-semibold text-white">{item.quality_grade || 'Not specified'}</span>
                </div>
                <div>
                  <span className="text-[10px] text-industrial-400 uppercase font-semibold block">Asking Price</span>
                  <span className="font-semibold text-white">
                    {item.price_per_unit != null ? `₹${item.price_per_unit}/${item.unit}` : 'Negotiable'}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center border-t border-industrial-800/80 pt-3.5 mt-2">
              <span className="text-[11px] text-industrial-400 flex items-center gap-1">
                <Calendar className="w-3.5 h-3.5" />
                <span>Expires: {item.available_until}</span>
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
