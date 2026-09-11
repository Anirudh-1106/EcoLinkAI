import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Sparkles, CheckCircle2, AlertCircle, RefreshCw, Layers, Search, ClipboardList, SlidersHorizontal, Save, X } from 'lucide-react';
import { fetchApi } from '../api/client';
import { PartnerCard, RequirementRecommendationResponse, SearchRecommendationResponse, Requirement, Plant } from '../types';
import { PartnerCardComponent } from '../components/PartnerCard';
import { MapView } from '../components/MapView';
import { useAuth } from '../context/AuthContext';

type TabMode = 'discover' | 'requirements';

// Unified response shape for rendering results
type RecommendationResult = {
  material_name: string;
  buyer_plant_name: string;
  buyer_plant_latitude: number;
  buyer_plant_longitude: number;
  total_candidates: number;
  recommendations: PartnerCard[];
  model_version: string;
  inference_time_ms?: number;
};

export const RecommendationsPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const requirementIdParam = searchParams.get('requirement_id');
  const { user } = useAuth();
  const navigate = useNavigate();

  // ── Tab state ──
  const [activeTab, setActiveTab] = useState<TabMode>(requirementIdParam ? 'requirements' : 'discover');

  // ── Shared state ──
  const [recommendations, setRecommendations] = useState<RecommendationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [requestSuccess, setRequestSuccess] = useState('');
  const [plants, setPlants] = useState<Plant[]>([]);
  const [selectedPlantId, setSelectedPlantId] = useState<string>('');

  // ── "My Requirements" tab state ──
  const [requirements, setRequirements] = useState<Requirement[]>([]);
  const [selectedRequirementId, setSelectedRequirementId] = useState<string>(requirementIdParam || '');

  // ── "Discover Sellers" tab state ──
  const [searchQuery, setSearchQuery] = useState('');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [quantityFilter, setQuantityFilter] = useState<number | null>(null);
  const [distanceFilter, setDistanceFilter] = useState<number | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [lastSearchedQuery, setLastSearchedQuery] = useState('');
  const searchRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── "Save as Requirement" state ──
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [saveQuantity, setSaveQuantity] = useState<string>('1000');
  const [saveBudget, setSaveBudget] = useState<string>('');
  const [saving, setSaving] = useState(false);

  // ── Load plants and requirements ──
  useEffect(() => {
    if (!user?.company_id) return;

    // Fetch buyer's plants
    fetchApi<{ items: Plant[] }>(`/plants?company_id=${user.company_id}`)
      .then((data) => {
        setPlants(data.items);
        if (data.items.length > 0 && !selectedPlantId) {
          setSelectedPlantId(data.items[0].id);
        }
      })
      .catch((err) => console.error(err));

    // Fetch buyer's requirements
    fetchApi<{ items: Requirement[] }>(`/requirements?company_id=${user.company_id}`)
      .then((data) => {
        setRequirements(data.items);
        if (!selectedRequirementId && data.items.length > 0) {
          setSelectedRequirementId(data.items[0].id);
        }
      })
      .catch((err) => console.error(err));
  }, []);

  // ── Close suggestions on outside click ──
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setShowSuggestions(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // ── Auto-fetch when requirement changes (requirements tab) ──
  useEffect(() => {
    if (activeTab === 'requirements' && selectedRequirementId) {
      handleFetchByRequirement(selectedRequirementId);
    }
  }, [selectedRequirementId]);

  // ── Debounced autocomplete ──
  const handleSearchInputChange = useCallback((value: string) => {
    setSearchQuery(value);
    if (debounceRef.current) clearTimeout(debounceRef.current);

    if (value.length < 2) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    debounceRef.current = setTimeout(async () => {
      try {
        const data = await fetchApi<{ suggestions: string[] }>(`/materials/autocomplete?q=${encodeURIComponent(value)}`);
        setSuggestions(data.suggestions);
        setShowSuggestions(data.suggestions.length > 0);
      } catch {
        setSuggestions([]);
      }
    }, 300);
  }, []);

  // ── Discover Sellers: Search ──
  const handleSearch = async (query?: string) => {
    const q = query || searchQuery;
    if (!q || q.length < 2 || !selectedPlantId) return;

    setLoading(true);
    setError('');
    setRequestSuccess('');
    setShowSuggestions(false);
    setLastSearchedQuery(q);

    try {
      const body: Record<string, unknown> = {
        material_query: q,
        buyer_plant_id: selectedPlantId,
        max_results: 10,
      };
      if (quantityFilter) body.quantity_needed = quantityFilter;
      if (distanceFilter) body.max_distance_km = distanceFilter;

      const data = await fetchApi<SearchRecommendationResponse>('/recommendations/by-search', {
        method: 'POST',
        body: JSON.stringify(body),
      });
      setRecommendations(data);
    } catch (err: any) {
      setError(err.message || 'Failed to search for sellers');
    } finally {
      setLoading(false);
    }
  };

  // ── My Requirements: Fetch ──
  const handleFetchByRequirement = async (requirementId: string) => {
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

  // ── Exchange Request ──
  const handleSendExchangeRequest = async (partner: PartnerCard) => {
    if (activeTab === 'requirements') {
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
    } else {
      // Discover mode — send with search context
      try {
        await fetchApi('/exchange-requests', {
          method: 'POST',
          body: JSON.stringify({
            waste_listing_id: partner.waste_listing_id,
            buyer_plant_id: selectedPlantId,
            requested_quantity: partner.listing_quantity,
            offered_price_per_unit: partner.listing_price_per_unit,
            recommendation_rank: partner.rank,
            remarks: `Purchase request via AI search "${lastSearchedQuery}" (Rank #${partner.rank}, Match ${partner.ai_score}%)`,
          }),
        });
        setRequestSuccess(`Purchase request sent to ${partner.company_name} (${partner.plant_name})! Awaiting seller's response.`);
        setTimeout(() => navigate('/exchange-requests'), 1500);
      } catch (err: any) {
        alert(`Error sending request: ${err.message}`);
      }
    }
  };

  // ── Save as Requirement ──
  const handleSaveAsRequirement = async () => {
    if (!lastSearchedQuery || !selectedPlantId) return;
    setSaving(true);

    try {
      // First find the material ID by searching materials
      const materialsData = await fetchApi<{ items: { id: string; material_name: string }[] }>(`/materials?search=${encodeURIComponent(lastSearchedQuery)}&page_size=1`);
      if (!materialsData.items.length) {
        alert('Could not find matching material in database');
        setSaving(false);
        return;
      }

      const materialId = materialsData.items[0].id;
      const futureDate = new Date();
      futureDate.setFullYear(futureDate.getFullYear() + 1);

      await fetchApi('/requirements', {
        method: 'POST',
        body: JSON.stringify({
          plant_id: selectedPlantId,
          material_id: materialId,
          quantity: parseFloat(saveQuantity) || 1000,
          unit: 'kg',
          maximum_budget_per_unit: saveBudget ? parseFloat(saveBudget) : null,
          required_before: futureDate.toISOString().split('T')[0],
          notes: `Created from marketplace search: "${lastSearchedQuery}"`,
        }),
      });

      setShowSaveModal(false);
      setRequestSuccess(`"${lastSearchedQuery}" saved as a permanent requirement! You can find it in the "My Requirements" tab.`);

      // Refresh requirements list
      if (user?.company_id) {
        const data = await fetchApi<{ items: Requirement[] }>(`/requirements?company_id=${user.company_id}`);
        setRequirements(data.items);
      }
    } catch (err: any) {
      alert(`Error saving requirement: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  // ── Map markers ──
  const selectedRequirement = requirements.find((r) => r.id === selectedRequirementId);

  const buyerMarker = recommendations
    ? {
        id: activeTab === 'requirements' && selectedRequirement ? selectedRequirement.plant_id : selectedPlantId,
        name: recommendations.buyer_plant_name || 'Your Plant',
        lat: recommendations.buyer_plant_latitude,
        lng: recommendations.buyer_plant_longitude,
        district: '',
        isSupplier: true,
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

  // ── Render ──
  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-eco-950 via-industrial-900 to-industrial-900 border border-eco-500/30 rounded-3xl p-6 sm:p-8 shadow-xl">
        <div className="flex flex-col gap-5">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-eco-600/20 border border-eco-500/30 text-eco-400 text-xs font-semibold mb-2">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Multi-Channel Graph Neural Network Engine</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-black text-white">AI Seller Recommendations</h1>
            <p className="text-xs sm:text-sm text-industrial-300 max-w-2xl mt-1">
              Find the best sellers using MC-GNN inference across material compatibility, logistics, trust, and carbon impact.
            </p>
          </div>

          {/* Tab Switcher */}
          <div className="flex gap-1 bg-industrial-950 rounded-xl p-1 w-fit">
            <button
              onClick={() => { setActiveTab('discover'); setRecommendations(null); setError(''); }}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                activeTab === 'discover'
                  ? 'bg-eco-600 text-white shadow-lg'
                  : 'text-industrial-400 hover:text-white hover:bg-industrial-800'
              }`}
            >
              <Search className="w-4 h-4" />
              Discover Sellers
            </button>
            <button
              onClick={() => { setActiveTab('requirements'); setRecommendations(null); setError(''); }}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                activeTab === 'requirements'
                  ? 'bg-eco-600 text-white shadow-lg'
                  : 'text-industrial-400 hover:text-white hover:bg-industrial-800'
              }`}
            >
              <ClipboardList className="w-4 h-4" />
              My Requirements
            </button>
          </div>

          {/* === DISCOVER TAB: Search Bar + Filters === */}
          {activeTab === 'discover' && (
            <div className="space-y-3">
              <div className="flex flex-col sm:flex-row gap-3">
                {/* Search Bar with Autocomplete */}
                <div ref={searchRef} className="relative flex-1">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-industrial-500" />
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => handleSearchInputChange(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter') handleSearch(); }}
                      placeholder="Search for a material (e.g., Rubber Dust, Aluminium, Plastic...)"
                      className="w-full bg-industrial-950 border border-industrial-700 rounded-xl pl-10 pr-4 py-2.5 text-sm text-white placeholder-industrial-500 focus:outline-none focus:border-eco-500 shadow-inner"
                    />
                  </div>

                  {/* Autocomplete Dropdown */}
                  {showSuggestions && suggestions.length > 0 && (
                    <div className="absolute z-50 w-full mt-1 bg-industrial-900 border border-industrial-700 rounded-xl shadow-2xl overflow-hidden">
                      {suggestions.map((s, i) => (
                        <button
                          key={i}
                          onClick={() => {
                            setSearchQuery(s);
                            setShowSuggestions(false);
                            handleSearch(s);
                          }}
                          className="w-full text-left px-4 py-2.5 text-sm text-industrial-200 hover:bg-eco-600/20 hover:text-white transition-colors flex items-center gap-2"
                        >
                          <Search className="w-3 h-3 text-industrial-500" />
                          {s}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {/* Filter Toggle + Search Button */}
                <div className="flex gap-2">
                  <button
                    onClick={() => setShowFilters(!showFilters)}
                    className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold border transition-all ${
                      showFilters
                        ? 'bg-eco-600/20 border-eco-500/50 text-eco-400'
                        : 'bg-industrial-950 border-industrial-700 text-industrial-400 hover:border-industrial-500'
                    }`}
                  >
                    <SlidersHorizontal className="w-4 h-4" />
                    Filters
                  </button>
                  <button
                    onClick={() => handleSearch()}
                    disabled={searchQuery.length < 2 || !selectedPlantId}
                    className="px-6 py-2.5 bg-eco-600 hover:bg-eco-500 disabled:bg-industrial-700 disabled:text-industrial-500 text-white rounded-xl text-sm font-semibold transition-all shadow-lg disabled:shadow-none"
                  >
                    Search
                  </button>
                </div>
              </div>

              {/* Filter Panel */}
              {showFilters && (
                <div className="bg-industrial-950 border border-industrial-800 rounded-xl p-4 flex flex-wrap items-end gap-6">
                  <div className="flex-1 min-w-[200px]">
                    <label className="block text-xs font-semibold text-industrial-400 uppercase tracking-wider mb-1">
                      Quantity Needed (kg)
                    </label>
                    <div className="flex items-center gap-3">
                      <input
                        type="range"
                        min="0"
                        max="10000"
                        step="100"
                        value={quantityFilter || 0}
                        onChange={(e) => setQuantityFilter(parseInt(e.target.value) || null)}
                        className="flex-1 accent-eco-500"
                      />
                      <span className="text-sm text-white font-mono w-20 text-right">
                        {quantityFilter ? `${quantityFilter.toLocaleString()} kg` : 'Any'}
                      </span>
                    </div>
                  </div>
                  <div className="flex-1 min-w-[200px]">
                    <label className="block text-xs font-semibold text-industrial-400 uppercase tracking-wider mb-1">
                      Max Distance (km)
                    </label>
                    <div className="flex items-center gap-3">
                      <input
                        type="range"
                        min="0"
                        max="500"
                        step="10"
                        value={distanceFilter || 0}
                        onChange={(e) => setDistanceFilter(parseInt(e.target.value) || null)}
                        className="flex-1 accent-eco-500"
                      />
                      <span className="text-sm text-white font-mono w-20 text-right">
                        {distanceFilter ? `${distanceFilter} km` : 'Any'}
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={() => { setQuantityFilter(null); setDistanceFilter(null); }}
                    className="text-xs text-industrial-500 hover:text-industrial-300 transition-colors"
                  >
                    Reset Filters
                  </button>
                </div>
              )}
            </div>
          )}

          {/* === REQUIREMENTS TAB: Dropdown === */}
          {activeTab === 'requirements' && (
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
          )}
        </div>
      </div>

      {/* Success Banner */}
      {requestSuccess && (
        <div className="bg-eco-600/15 border border-eco-500/30 rounded-xl p-4 text-xs text-eco-300 flex items-center gap-2 shadow-lg">
          <CheckCircle2 className="w-5 h-5 text-eco-400 shrink-0" />
          <span>{requestSuccess}</span>
        </div>
      )}

      {/* Error Banner */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-xs text-red-400 flex items-center gap-2">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Loading Spinner */}
      {loading && (
        <div className="bg-industrial-900 border border-industrial-800 rounded-2xl p-12 text-center text-industrial-400 space-y-3">
          <RefreshCw className="w-8 h-8 text-eco-400 animate-spin mx-auto" />
          <p className="text-sm font-semibold text-white">Running MC-GNN Graph Inference...</p>
          <p className="text-xs text-industrial-400">
            {activeTab === 'discover' ? `Searching sellers for "${searchQuery}"` : 'Finding best sellers for your material requirement'}
          </p>
        </div>
      )}

      {/* Results */}
      {!loading && recommendations && (
        <div className="space-y-6">
          {/* Save as Requirement Banner (Discover mode only) */}
          {activeTab === 'discover' && lastSearchedQuery && recommendations.recommendations.length > 0 && (
            <div className="bg-eco-950/50 border border-eco-500/20 rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
              <div>
                <p className="text-sm text-white font-semibold">Like these results?</p>
                <p className="text-xs text-industrial-400">Save "{lastSearchedQuery}" as a permanent requirement to get automatic updates when new sellers join.</p>
              </div>
              <button
                onClick={() => setShowSaveModal(true)}
                className="flex items-center gap-2 px-4 py-2 bg-eco-600 hover:bg-eco-500 text-white rounded-lg text-sm font-semibold transition-all shadow-lg whitespace-nowrap"
              >
                <Save className="w-4 h-4" />
                Save as Requirement
              </button>
            </div>
          )}

          {/* Recommendation Metadata Bar */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-2xl p-4 flex flex-wrap items-center justify-between gap-4 text-xs text-industrial-300">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1.5 text-eco-400 font-semibold">
                <Layers className="w-4 h-4" />
                <span>Model: {recommendations.model_version}</span>
              </span>
              <span>
                {activeTab === 'discover' ? 'Search' : 'Requirement'}: <strong className="text-white">{recommendations.material_name}</strong>
              </span>
              <span>Sellers Evaluated: <strong className="text-white">{recommendations.total_candidates}</strong></span>
            </div>
            {recommendations.inference_time_ms && (
              <span className="text-industrial-400">
                Inference Latency: <strong className="text-eco-400">{recommendations.inference_time_ms} ms</strong>
              </span>
            )}
          </div>

          {/* Map View */}
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

            {recommendations.recommendations.length === 0 && (
              <div className="bg-industrial-900 border border-industrial-800 rounded-2xl p-8 text-center">
                <p className="text-sm text-industrial-400">No matching sellers found. Try a different search term or adjust your filters.</p>
              </div>
            )}

            {recommendations.recommendations.map((partner) => (
              <PartnerCardComponent
                key={`${partner.plant_id}-${partner.waste_listing_id}`}
                partner={partner}
                onSelect={handleSendExchangeRequest}
              />
            ))}
          </div>
        </div>
      )}

      {/* Save as Requirement Modal */}
      {showSaveModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-industrial-900 border border-industrial-700 rounded-2xl p-6 w-full max-w-md shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-bold text-white">Save as Requirement</h3>
              <button onClick={() => setShowSaveModal(false)} className="text-industrial-500 hover:text-white">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3">
              <div>
                <label className="block text-xs font-semibold text-industrial-400 uppercase tracking-wider mb-1">Material</label>
                <input
                  type="text"
                  value={lastSearchedQuery}
                  disabled
                  className="w-full bg-industrial-950 border border-industrial-800 rounded-lg px-3 py-2 text-sm text-industrial-300"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-industrial-400 uppercase tracking-wider mb-1">Quantity Needed (kg)</label>
                <input
                  type="number"
                  value={saveQuantity}
                  onChange={(e) => setSaveQuantity(e.target.value)}
                  className="w-full bg-industrial-950 border border-industrial-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-eco-500"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-industrial-400 uppercase tracking-wider mb-1">Max Budget per Unit (₹, optional)</label>
                <input
                  type="number"
                  value={saveBudget}
                  onChange={(e) => setSaveBudget(e.target.value)}
                  placeholder="No limit"
                  className="w-full bg-industrial-950 border border-industrial-700 rounded-lg px-3 py-2 text-sm text-white placeholder-industrial-600 focus:outline-none focus:border-eco-500"
                />
              </div>
            </div>

            <div className="flex gap-3 pt-2">
              <button
                onClick={() => setShowSaveModal(false)}
                className="flex-1 px-4 py-2.5 bg-industrial-800 text-industrial-300 rounded-lg text-sm font-semibold hover:bg-industrial-700 transition-all"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveAsRequirement}
                disabled={saving}
                className="flex-1 px-4 py-2.5 bg-eco-600 hover:bg-eco-500 disabled:bg-industrial-700 text-white rounded-lg text-sm font-semibold transition-all shadow-lg flex items-center justify-center gap-2"
              >
                {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                {saving ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
