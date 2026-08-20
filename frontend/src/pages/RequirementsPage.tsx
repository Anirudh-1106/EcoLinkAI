import React, { useEffect, useState } from 'react';
import { ClipboardList, MapPin, Calendar, CheckCircle2 } from 'lucide-react';
import { fetchApi } from '../api/client';
import { Requirement } from '../types';

export const RequirementsPage: React.FC = () => {
  const [requirements, setRequirements] = useState<Requirement[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchApi<{ items: Requirement[] }>('/requirements')
      .then((data) => setRequirements(data.items))
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Material Requirements</h1>
          <p className="text-xs text-industrial-400">Demand posted by industrial consumer plants</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {requirements.map((item) => (
          <div key={item.id} className="bg-industrial-900 border border-industrial-800 rounded-2xl p-5 shadow-sm hover:border-industrial-700 transition-all">
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
                <span className="font-semibold text-white">{item.minimum_purity || 80}%</span>
              </div>
              <div>
                <span className="text-[10px] text-industrial-400 uppercase font-semibold block">Max Budget</span>
                <span className="font-semibold text-white">₹{item.maximum_budget_per_unit || 'Open'}</span>
              </div>
              <div>
                <span className="text-[10px] text-industrial-400 uppercase font-semibold block">Max Distance</span>
                <span className="font-semibold text-white">{item.preferred_max_distance_km || 300} km</span>
              </div>
            </div>

            <div className="text-[11px] text-industrial-400 flex items-center gap-1 border-t border-industrial-800/80 pt-3">
              <Calendar className="w-3.5 h-3.5" />
              <span>Required before: {item.required_before}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
