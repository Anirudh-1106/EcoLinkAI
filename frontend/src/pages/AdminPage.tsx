import React, { useEffect, useState } from 'react';
import { ShieldCheck, Building2, Factory, Trash2, Layers, CheckCircle2, XCircle } from 'lucide-react';
import { fetchApi } from '../api/client';
import { Company, PlatformAnalytics } from '../types';

export const AdminPage: React.FC = () => {
  const [companies, setCompanies] = useState<Company[]>([]);
  const [platform, setPlatform] = useState<PlatformAnalytics | null>(null);

  useEffect(() => {
    fetchApi<{ items: Company[] }>('/companies').then((d) => setCompanies(d.items));
    fetchApi<PlatformAnalytics>('/analytics/platform').then(setPlatform);
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Platform Admin Panel</h1>
          <p className="text-xs text-industrial-400">Verify company credentials, monitor ecosystem statistics, and AI health</p>
        </div>
      </div>

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
