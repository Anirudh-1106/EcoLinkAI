import React, { useState } from 'react';
import { Sparkles, MapPin, Truck, Leaf, ShieldCheck, ArrowRight, CheckCircle2, ChevronDown, Package } from 'lucide-react';
import { PartnerCard as PartnerCardType, AlternativeLot } from '../types';

interface PartnerCardProps {
  partner: PartnerCardType;
  onSelect: (partner: PartnerCardType) => void;
}

export const PartnerCardComponent: React.FC<PartnerCardProps> = ({ partner, onSelect }) => {
  const exp = partner.explanation;
  const [showLots, setShowLots] = useState(false);
  const alternativeLots = partner.alternative_lots ?? [];

  // Buying an alternative lot is buying a different listing from the same
  // seller, so the card handed onward carries that listing's own figures.
  const selectLot = (lot: AlternativeLot) =>
    onSelect({
      ...partner,
      waste_listing_id: lot.waste_listing_id,
      listing_quantity: lot.quantity ?? undefined,
      listing_unit: lot.unit ?? undefined,
      listing_price_per_unit: lot.price_per_unit ?? undefined,
      listing_purity: lot.purity ?? undefined,
      ai_score: lot.ai_score,
      estimated_transport_cost: lot.estimated_transport_cost,
      estimated_carbon_saving: lot.estimated_carbon_saving,
    });

  return (
    <div className="bg-industrial-900 border border-industrial-800 rounded-2xl p-6 shadow-md hover:border-eco-500/50 transition-all group">
      {/* Top Header Row */}
      <div className="flex items-start justify-between gap-4 border-b border-industrial-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="bg-eco-600/20 border border-eco-500/30 text-eco-400 text-xs font-bold px-2.5 py-1 rounded-md">
              RANK #{partner.rank}
            </span>
            <span className="text-xs font-medium text-industrial-400 uppercase tracking-wider">
              {partner.model_type === 'mc_gnn' ? 'MC-GNN' : 'BASELINE'} MODEL
            </span>
          </div>
          <h3 className="text-lg font-bold text-white mt-1.5 group-hover:text-eco-400 transition-colors">
            {partner.company_name}
          </h3>
          <p className="text-xs text-industrial-400 flex items-center gap-1.5 mt-0.5">
            <MapPin className="w-3.5 h-3.5 text-industrial-400" />
            <span>{partner.plant_name} ({partner.plant_district}, {partner.plant_state})</span>
          </p>
          {partner.listing_quantity != null && (
            <p className="text-xs text-industrial-300 mt-1">
              <span className="text-eco-400 font-semibold">{partner.listing_quantity} {partner.listing_unit || 'KG'}</span>
              {' available'}
              {partner.listing_price_per_unit != null && (
                <span> · ₹{partner.listing_price_per_unit}/{partner.listing_unit || 'unit'}</span>
              )}
              {partner.listing_purity != null && (
                <span> · {partner.listing_purity}% purity</span>
              )}
            </p>
          )}
        </div>

        {/* AI Compatibility Score Badge */}
        <div
          className="text-right bg-industrial-950 border border-industrial-800 p-3 rounded-xl min-w-[120px]"
          title={
            partner.model_type === 'mc_gnn'
              ? 'Likelihood this exchange is accepted, predicted by the MC-GNN from historical exchange outcomes.'
              : 'Weighted compatibility score from the rule-based baseline model.'
          }
        >
          <div className="flex items-center justify-end gap-1 text-eco-400 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5" />
            <span>AI MATCH</span>
          </div>
          <div className="text-2xl font-black text-white mt-0.5">{partner.ai_score}%</div>
          <div className="text-[10px] text-industrial-500 mt-0.5">
            {partner.model_type === 'mc_gnn' ? 'predicted acceptance' : 'rule-based score'}
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 my-4">
        <div className="bg-industrial-950/60 p-3 rounded-xl border border-industrial-800/80">
          <span className="text-[10px] uppercase tracking-wider text-industrial-400 font-semibold block">Material Match</span>
          <span className="text-sm font-bold text-white mt-0.5 block">{exp.material_compatibility}</span>
          <span className="text-[11px] text-eco-400">{exp.quantity_match_pct}% quantity match</span>
        </div>

        <div className="bg-industrial-950/60 p-3 rounded-xl border border-industrial-800/80">
          <span className="text-[10px] uppercase tracking-wider text-industrial-400 font-semibold block">Distance & Transport</span>
          <span className="text-sm font-bold text-white mt-0.5 block">{partner.distance_km} km</span>
          <span className="text-[11px] text-industrial-300 flex items-center gap-1">
            <Truck className="w-3 h-3 text-industrial-400" />
            <span>₹{exp.estimated_transport_cost.toLocaleString()} ({exp.transport_feasibility})</span>
          </span>
        </div>

        <div className="bg-industrial-950/60 p-3 rounded-xl border border-industrial-800/80">
          <span className="text-[10px] uppercase tracking-wider text-industrial-400 font-semibold block">Trust Score</span>
          <span className="text-sm font-bold text-amber-400 mt-0.5 block flex items-center gap-1">
            <ShieldCheck className="w-4 h-4" />
            <span>{exp.trust_score.toFixed(1)} / 100</span>
          </span>
          <span className="text-[11px] text-industrial-400">{exp.historical_exchange_count} past exchanges</span>
        </div>

        <div className="bg-industrial-950/60 p-3 rounded-xl border border-industrial-800/80">
          <span className="text-[10px] uppercase tracking-wider text-industrial-400 font-semibold block">Carbon Savings</span>
          <span className="text-sm font-bold text-eco-400 mt-0.5 block flex items-center gap-1">
            <Leaf className="w-4 h-4" />
            <span>{exp.carbon_benefit_kg.toFixed(0)} kg CO₂e</span>
          </span>
          <span className="text-[11px] text-industrial-400">Avoided virgin impact</span>
        </div>
      </div>

      {/* AI Explanation Box */}
      <div className="bg-eco-950/30 border border-eco-500/20 rounded-xl p-3.5 text-xs text-industrial-200 mb-4 flex items-start gap-2.5">
        <CheckCircle2 className="w-4 h-4 text-eco-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-eco-300 block mb-0.5">Why this partner was recommended:</span>
          <p className="text-industrial-300 leading-relaxed">{exp.recommendation_summary}</p>
        </div>
      </div>

      {/* Other lots of the same material from this seller */}
      {alternativeLots.length > 0 && (
        <div className="mb-4">
          <button
            onClick={() => setShowLots((open) => !open)}
            className="w-full flex items-center justify-between gap-2 bg-industrial-950/60 hover:bg-industrial-950 border border-industrial-800 rounded-xl px-3.5 py-2.5 transition-colors"
          >
            <span className="flex items-center gap-2 text-xs text-industrial-300">
              <Package className="w-3.5 h-3.5 text-industrial-400" />
              <span>
                <span className="font-semibold text-white">{alternativeLots.length} more lot{alternativeLots.length > 1 ? 's' : ''}</span>
                {' from this seller'}
              </span>
            </span>
            <ChevronDown
              className={`w-4 h-4 text-industrial-400 transition-transform ${showLots ? 'rotate-180' : ''}`}
            />
          </button>

          {showLots && (
            <div className="mt-2 space-y-2">
              {alternativeLots.map((lot) => (
                <div
                  key={lot.waste_listing_id}
                  className="flex items-center justify-between gap-3 bg-industrial-950/40 border border-industrial-800/70 rounded-xl px-3.5 py-2.5"
                >
                  <div className="min-w-0">
                    <p className="text-xs text-white font-semibold truncate">
                      {lot.quantity} {lot.unit || 'kg'}
                      {lot.price_per_unit != null && (
                        <span className="text-industrial-300 font-normal"> · ₹{lot.price_per_unit}/{lot.unit || 'unit'}</span>
                      )}
                    </p>
                    <p className="text-[11px] text-industrial-400">
                      {lot.purity != null && <span>{lot.purity}% purity · </span>}
                      <span className="text-eco-400">{lot.quantity_match_pct}% quantity match</span>
                      <span> · AI {lot.ai_score}%</span>
                    </p>
                  </div>
                  <button
                    onClick={() => selectLot(lot)}
                    className="shrink-0 bg-industrial-800 hover:bg-eco-600 text-white font-semibold text-[11px] px-3 py-1.5 rounded-lg transition-colors"
                  >
                    Buy this lot
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Action CTA */}
      <div className="flex justify-end">
        <button
          onClick={() => onSelect(partner)}
          className="bg-eco-600 hover:bg-eco-500 text-white font-semibold text-xs px-4 py-2.5 rounded-xl transition-all shadow-md shadow-eco-600/20 flex items-center gap-2"
        >
          <span>Request to Buy</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
