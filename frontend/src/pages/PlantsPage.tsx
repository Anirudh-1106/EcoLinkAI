import React, { useEffect, useState } from 'react';
import { Factory, MapPin, Plus, AlertCircle } from 'lucide-react';
import { fetchApi } from '../api/client';
import { Plant } from '../types';
import { MapView } from '../components/MapView';
import { ErrorBanner, EmptyState } from '../components/ErrorBanner';
import { useAuth } from '../context/AuthContext';

export const PlantsPage: React.FC = () => {
  const [plants, setPlants] = useState<Plant[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { user } = useAuth();

  // Scoped to the signed-in company. Unfiltered, /plants returns every
  // facility on the platform, so this page -- which presents itself as your
  // own facilities -- was listing competitors and plotting their coordinates
  // on the map.
  useEffect(() => {
    if (!user?.company_id) {
      setLoading(false);
      return;
    }
    fetchApi<{ items: Plant[] }>(`/plants?company_id=${user.company_id}`)
      .then((data) => setPlants(data.items))
      .catch((err) => setError(err.message || 'Failed to load plants'))
      .finally(() => setLoading(false));
  }, [user?.company_id]);

  const mapPlants = plants.map((p) => ({
    id: p.id,
    name: p.plant_name,
    lat: Number(p.latitude),
    lng: Number(p.longitude),
    district: p.district,
  }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Manufacturing Plants</h1>
          <p className="text-xs text-industrial-400">Manage physical operating facilities and location coordinates</p>
        </div>
      </div>

      {error && <ErrorBanner message={error} />}
      {!loading && !error && plants.length === 0 && (
        <EmptyState message="No plants registered yet. Add a plant to start listing waste or requirements." />
      )}

      {/* Map View */}
      {plants.length > 0 && (
        <div className="space-y-2">
          <h2 className="text-sm font-semibold text-industrial-300 uppercase tracking-wider">Facility Geographic Locations</h2>
          <MapView partners={mapPlants} height="320px" />
        </div>
      )}

      {/* Plant Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {plants.map((plant) => (
          <div key={plant.id} className="bg-industrial-900 border border-industrial-800 rounded-2xl p-5 shadow-sm hover:border-industrial-700 transition-all">
            <div className="flex items-center gap-3 mb-3">
              <div className="bg-blue-500/10 text-blue-400 border border-blue-500/20 p-2.5 rounded-xl">
                <Factory className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-white text-base">{plant.plant_name}</h3>
                <span className="text-[11px] text-industrial-400 uppercase tracking-wider">{plant.plant_type}</span>
              </div>
            </div>

            <div className="text-xs text-industrial-300 space-y-1.5 border-t border-industrial-800/80 pt-3">
              <p className="flex items-center gap-1.5">
                <MapPin className="w-3.5 h-3.5 text-industrial-400 shrink-0" />
                <span>{plant.address}, {plant.district}, {plant.state}</span>
              </p>
              <p className="text-industrial-400 text-[11px]">
                Lat: {Number(plant.latitude).toFixed(4)}, Lon: {Number(plant.longitude).toFixed(4)}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
