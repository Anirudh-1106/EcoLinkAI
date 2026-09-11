import React, { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Sparkles, ArrowRight, CheckCircle2, AlertCircle, RefreshCw, Layers } from 'lucide-react';
import { fetchApi } from '../api/client';
import { PartnerCard, RequirementRecommendationResponse, Requirement } from '../types';
import { PartnerCardComponent } from '../components/PartnerCard';
import { MapView } from '../components/MapView';
import { useAuth } from '../context/AuthContext';

export const RecommendationsPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const requirementIdParam = searchParams.get('requirement_id');
  const { user } = useAuth();

  const [requirements, setRequirements] = useState<Requirement[]>([]);
  const [selectedRequirementId, setSelectedRequirementId] = useState<string>(requirementIdParam || '');
  const [recommendations, setRecommendations] = useState<RequirementRecommendationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [requestSuccess, setRequestSuccess] = useState('');

  const navigate = useNavigate();

  useEffect(() => {
    if (!user?.company_id) return;
    
    // Fetch buyer's requirements for dropdown
    fetchApi<{ items: Requirement[] }>(`/requirements?company_id=${user.company_id}`)
      .then((data) => {
        setRequirements(data.items);
        if (!selectedRequirementId && data.items.length > 0) {
          setSelectedRequirementId(data.items[0].id);
        }
      })
      .catch((err) => console.error(err));
  }, []);

  useEffect(() => {
    if (selectedRequirementId) {
      handleFetchRecommendations(selectedRequirementId);
    }
  }, [selectedRequirementId]);

  const handleFetchRecommendations = async (requirementId: string) => {
    setLoading(true);
    setError('');
    setRequestSuccess('');

    try {
      const data = await fetchApi<RequirementRecommendationResponse>('/recommendations/by-requirement', {
        method: 'POST',
        body: JSON.stringify({
          requirement_id: requirementId,
          max_results: 10,
        }),
      });
      setRecommendations(data);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch AI seller recommendations');
    } finally {
      setLoading(false);
    }
  };

  const handleSendExchangeRequest = async (partner: PartnerCard) => {
    const selectedReq = requirements.find((r) => r.id === selectedRequirementId);
    if (!selectedReq) return;

    try {
      await fetchApi('/exchange-requests', {
        method: 'POST',
        body: JSON.stringify({
          waste_listing_id: partner.waste_listing_id,
          buyer_plant_id: selectedReq.plant_id,
          requirement_id: selectedRequirementId,
          requested_quantity: partner.listing_quantity || selectedReq.quantity,
          offered_price_per_unit: partner.listing_price_per_unit,
          recommendation_rank: partner.rank,
          remarks: `Purchase request via AI recommendation (Rank #${partner.rank}, Match ${partner.ai_score}%)`,
        }),
      });

      setRequestSuccess(`Purchase request sent to ${partner.company_name} (${partner.plant_name})! Awaiting seller's response.`);
      setTimeout(() => navigate('/exchange-requests'), 1500);
    } catch (err: any) {
      alert(`Error sending request: ${err.message}`);
    }
  };

  const selectedRequirement = requirements.find((r) => r.id === selectedRequirementId);

  // Map markers — buyer's plant (current user) and recommended seller plants
  const buyerMarker = selectedRequirement && recommendations
    ? {
        id: selectedRequirement.plant_id,
        name: recommendations.buyer_plant_name || 'Your Plant',
        lat: recommendations.buyer_plant_latitude,
        lng: recommendations.buyer_plant_longitude,
        district: '',
        isSupplier: true, // reuse the "supplier" marker styling for "your" plant
      }
    : undefined;

  const sellerMarkers = recommendations
    ? recommendations.recommendations.slice(0, 3).map((p) => ({
        id: p.plant_id,
        name: p.plant_name,
        lat: p.plant_latitude,
        lng: p.plant_longitude,
        district: p.plant_district,
      }))
    : [];

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-eco-950 via-industrial-900 to-industrial-900 border border-eco-500/30 rounded-3xl p-6 sm:p-8 shadow-xl">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-eco-600/20 border border-eco-500/30 text-eco-400 text-xs font-semibold mb-2">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Multi-Channel Graph Neural Network Engine</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-black text-white">AI Seller Recommendations</h1>
            <p className="text-xs sm:text-sm text-industrial-300 max-w-2xl mt-1">
              Select a material requirement to find the best sellers with matching waste listings using MC-GNN inference across material compatibility, logistics, trust, and carbon impact.
            </p>
          </div>

          {/* Requirement Selector Dropdown */}
          <div className="w-full md:w-auto min-w-[280px]">
            <label className="block text-xs font-semibold text-industrial-300 uppercase tracking-wider mb-1.5">
              Select Your Requirement
            </label>
            <select
              value={selectedRequirementId}
              onChange={(e) => setSelectedRequirementId(e.target.value)}
              className="w-full bg-industrial-950 border border-industrial-700 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-eco-500 shadow-inner"
            >
              {requirements.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.material_name} — {r.quantity} {r.unit} ({r.plant_name})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {requestSuccess && (
        <div className="bg-eco-600/15 border border-eco-500/30 rounded-xl p-4 text-xs text-eco-300 flex items-center gap-2 shadow-lg">
          <CheckCircle2 className="w-5 h-5 text-eco-400 shrink-0" />
          <span>{requestSuccess} Redirecting to Exchange Requests...</span>
        </div>
      )}

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-xs text-red-400 flex items-center gap-2">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {loading && (
        <div className="bg-industrial-900 border border-industrial-800 rounded-2xl p-12 text-center text-industrial-400 space-y-3">
          <RefreshCw className="w-8 h-8 text-eco-400 animate-spin mx-auto" />
          <p className="text-sm font-semibold text-white">Running MC-GNN Graph Inference...</p>
          <p className="text-xs text-industrial-400">Finding best sellers for your material requirement</p>
        </div>
      )}

      {!loading && recommendations && (
        <div className="space-y-6">
          {/* Recommendation Metadata Bar */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-2xl p-4 flex flex-wrap items-center justify-between gap-4 text-xs text-industrial-300">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1.5 text-eco-400 font-semibold">
                <Layers className="w-4 h-4" />
                <span>Model: {recommendations.model_version}</span>
              </span>
              <span>Requirement: <strong className="text-white">{recommendations.material_name}</strong></span>
              <span>Sellers Evaluated: <strong className="text-white">{recommendations.total_candidates}</strong></span>
            </div>
            {recommendations.inference_time_ms && (
              <span className="text-industrial-400">
                Inference Latency: <strong className="text-eco-400">{recommendations.inference_time_ms} ms</strong>
              </span>
            )}
          </div>

          {/* Map View of Routes */}
          {sellerMarkers.length > 0 && (
            <div className="space-y-2">
              <h2 className="text-sm font-semibold text-industrial-300 uppercase tracking-wider">
                Geographic Logistics Routes (Your Plant → Top Recommended Sellers)
              </h2>
              <MapView supplier={buyerMarker} partners={sellerMarkers} height="350px" />
            </div>
          )}

          {/* Ranked Seller Cards */}
          <div className="space-y-4">
            <h2 className="text-sm font-semibold text-industrial-300 uppercase tracking-wider">
              Ranked Seller Candidates ({recommendations.recommendations.length})
            </h2>

            {recommendations.recommendations.map((partner) => (
              <PartnerCardComponent
                key={partner.plant_id}
                partner={partner}
                onSelect={handleSendExchangeRequest}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
