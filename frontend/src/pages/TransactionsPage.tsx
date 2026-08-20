import React, { useEffect, useState } from 'react';
import { Truck, CheckCircle2, Clock, MapPin, DollarSign, Leaf } from 'lucide-react';
import { fetchApi } from '../api/client';
import { Exchange } from '../types';

export const TransactionsPage: React.FC = () => {
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchApi<{ items: Exchange[] }>('/exchanges')
      .then((data) => setExchanges(data.items))
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, []);

  const handleUpdateStatus = async (exchangeId: string, status: string) => {
    try {
      await fetchApi(`/exchanges/${exchangeId}`, {
        method: 'PUT',
        body: JSON.stringify({ exchange_status: status, shipment_status: 'Delivered' }),
      });
      fetchApi<{ items: Exchange[] }>('/exchanges').then((data) => setExchanges(data.items));
    } catch (err: any) {
      alert(`Update error: ${err.message}`);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Transactions & Logistics</h1>
          <p className="text-xs text-industrial-400">Track active shipments, agreed pricing, and carbon savings</p>
        </div>
      </div>

      <div className="space-y-4">
        {exchanges.map((ex) => (
          <div key={ex.id} className="bg-industrial-900 border border-industrial-800 rounded-2xl p-5 shadow-sm space-y-4">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-industrial-800 pb-3">
              <div>
                <span className="bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider">
                  {ex.exchange_status}
                </span>
                <h3 className="font-bold text-white text-base mt-1">
                  {ex.supplier_plant_name || 'Supplier'} ↔ {ex.buyer_plant_name || 'Buyer'}
                </h3>
              </div>
              <div className="text-right">
                <span className="text-sm font-bold text-white block">₹{Number(ex.agreed_price).toLocaleString()}</span>
                <span className="text-[11px] text-industrial-400">Transport: ₹{Number(ex.transport_cost).toLocaleString()}</span>
              </div>
            </div>

            {/* Shipment Status Timeline */}
            <div className="grid grid-cols-4 gap-2 text-center text-xs">
              <div className="bg-industrial-950 p-2.5 rounded-xl border border-industrial-800">
                <span className="text-[10px] text-industrial-400 block font-semibold">INITIATED</span>
                <span className="text-eco-400 font-bold">✓</span>
              </div>
              <div className="bg-industrial-950 p-2.5 rounded-xl border border-industrial-800">
                <span className="text-[10px] text-industrial-400 block font-semibold">SCHEDULED</span>
                <span className="text-eco-400 font-bold">✓</span>
              </div>
              <div className="bg-industrial-950 p-2.5 rounded-xl border border-industrial-800">
                <span className="text-[10px] text-industrial-400 block font-semibold">IN TRANSIT</span>
                <span className="text-eco-400 font-bold">✓</span>
              </div>
              <div className="bg-industrial-950 p-2.5 rounded-xl border border-industrial-800">
                <span className="text-[10px] text-industrial-400 block font-semibold">DELIVERED</span>
                <span className="text-eco-400 font-bold">{ex.exchange_status === 'Completed' ? '✓' : '...'}</span>
              </div>
            </div>

            <div className="flex items-center justify-between text-xs text-industrial-300 pt-1">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1 text-eco-400">
                  <Leaf className="w-3.5 h-3.5" />
                  <span>Carbon Saved: {ex.actual_carbon_saving || 250} kg CO₂e</span>
                </span>
                {ex.delivered_at && (
                  <span className="text-industrial-400">Delivered: {new Date(ex.delivered_at).toLocaleDateString()}</span>
                )}
              </div>

              {ex.exchange_status !== 'Completed' && (
                <button
                  onClick={() => handleUpdateStatus(ex.id, 'Completed')}
                  className="bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs px-3.5 py-1.5 rounded-lg transition-all"
                >
                  Mark as Completed
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
