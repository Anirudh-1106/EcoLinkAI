import React, { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Sparkles, ArrowRight, CheckCircle2, AlertCircle, RefreshCw, Layers } from 'lucide-react';
import { fetchApi } from '../api/client';
import { PartnerCard, RecommendationResponse, WasteListing } from '../types';
import { PartnerCardComponent } from '../components/PartnerCard';
import { MapView } from '../components/MapView';

export const RecommendationsPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const listingIdParam = searchParams.get('listing_id');

  const [listings, setListings] = useState<WasteListing[]>([]);
  const [selectedListingId, setSelectedListingId] = useState<string>(listingIdParam || '');
  const [recommendations, setRecommendations] = useState<RecommendationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [requestSuccess, setRequestSuccess] = useState('');

  const navigate = useNavigate();

  useEffect(() => {
    // Fetch available listings for dropdown
    fetchApi<{ items: WasteListing[] }>('/waste-listings')
      .then((data) => {
        setListings(data.items);
        if (!selectedListingId && data.items.length > 0) {
          setSelectedListingId(data.items[0].id);
        }
      })
      .catch((err) => console.error(err));
  }, []);

  useEffect(() => {
    if (selectedListingId) {
      handleFetchRecommendations(selectedListingId);
    }
  }, [selectedListingId]);

  const handleFetchRecommendations = async (listingId: string) => {
    setLoading(true);
    setError('');
    setRequestSuccess('');

    try {
      const data = await fetchApi<RecommendationResponse>('/recommendations', {
        method: 'POST',
        body: JSON.stringify({
          waste_listing_id: listingId,
          max_results: 10,
        }),
      });
      setRecommendations(data);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch AI recommendations');
    } finally {
      setLoading(false);
    }
  };

  const handleSendExchangeRequest = async (partner: PartnerCard) => {
    try {
      await fetchApi('/exchange-requests', {
        method: 'POST',
        body: JSON.stringify({
          waste_listing_id: selectedListingId,
          buyer_plant_id: partner.plant_id,
          requirement_id: partner.requirement_id,
          requested_quantity: partner.required_quantity || 100,
          offered_price_per_unit: 100,
          remarks: `Request generated via AI recommendation (Rank #${partner.rank}, Match ${partner.ai_score}%)`,
        }),
      });

      setRequestSuccess(`Exchange request successfully sent to ${partner.company_name} (${partner.plant_name})!`);
      setTimeout(() => navigate('/exchange-requests'), 1500);
    } catch (err: any) {
      alert(`Error sending request: ${err.message}`);
    }
  };

  const selectedListing = listings.find((l) => l.id === selectedListingId);

  // Map markers
  const supplierMarker = selectedListing
    ? {
        id: selectedListing.plant_id,
        name: selectedListing.plant_name || 'Supplier Plant',
        lat: 9.9816, // Default Ernakulam lat
        lng: 76.2999,
        district: 'Ernakulam',
        isSupplier: true,
      }
    : undefined;

  const partnerMarkers = recommendations
    ? recommendations.recommendations.slice(0, 3).map((p, idx) => ({
        id: p.plant_id,
        name: p.plant_name,
        lat: 9.9816 + (idx + 1) * 0.15,
        lng: 76.2999 + (idx + 1) * 0.12,
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
            <h1 className="text-2xl sm:text-3xl font-black text-white">AI Partner Recommendations</h1>
            <p className="text-xs sm:text-sm text-industrial-300 max-w-2xl mt-1">
              Select a waste listing to run MC-GNN inference across multi-view graph relationships (material compatibility, logistics feasibility, trust history, and carbon impact).
            </p>
          </div>

          {/* Waste Listing Selector Dropdown */}
          <div className="w-full md:w-auto min-w-[280px]">
            <label className="block text-xs font-semibold text-industrial-300 uppercase tracking-wider mb-1.5">
              Select Waste Listing
            </label>
            <select
              value={selectedListingId}
              onChange={(e) => setSelectedListingId(e.target.value)}
              className="w-full bg-industrial-950 border border-industrial-700 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-eco-500 shadow-inner"
            >
              {listings.map((l) => (
                <option key={l.id} value={l.id}>
                  {l.material_name} — {l.quantity} {l.unit} ({l.plant_name})
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
          <p className="text-xs text-industrial-400">Extracting multi-view embeddings & computing Hilbert-Schmidt attention scores</p>
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
              <span>Listing: <strong className="text-white">{recommendations.material_name}</strong></span>
              <span>Candidates Evaluated: <strong className="text-white">{recommendations.total_candidates}</strong></span>
            </div>
            {recommendations.inference_time_ms && (
              <span className="text-industrial-400">
                Inference Latency: <strong className="text-eco-400">{recommendations.inference_time_ms} ms</strong>
              </span>
            )}
          </div>

          {/* Map View of Routes */}
          {partnerMarkers.length > 0 && (
            <div className="space-y-2">
              <h2 className="text-sm font-semibold text-industrial-300 uppercase tracking-wider">
                Geographic Logistics Routes (Supplier → Top Recommended Partners)
              </h2>
              <MapView supplier={supplierMarker} partners={partnerMarkers} height="350px" />
            </div>
          )}

          {/* Ranked Partner Cards */}
          <div className="space-y-4">
            <h2 className="text-sm font-semibold text-industrial-300 uppercase tracking-wider">
              Ranked Symbiosis Partner Candidates ({recommendations.recommendations.length})
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
