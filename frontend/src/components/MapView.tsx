import React from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import L from 'leaflet';

// Fix default marker icon issue in Leaflet with React
const defaultIcon = L.icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});
L.Marker.prototype.options.icon = defaultIcon;

interface MapPlant {
  id: string;
  name: string;
  lat: number;
  lng: number;
  district: string;
  isSupplier?: boolean;
}

interface MapViewProps {
  supplier?: MapPlant;
  partners?: MapPlant[];
  height?: string;
}

export const MapView: React.FC<MapViewProps> = ({
  supplier,
  partners = [],
  height = '400px',
}) => {
  // Center on Kerala region
  const centerLat = supplier ? supplier.lat : 10.0;
  const centerLng = supplier ? supplier.lng : 76.5;

  return (
    <div className="rounded-2xl overflow-hidden border border-industrial-800 shadow-md" style={{ height }}>
      <MapContainer
        center={[centerLat, centerLng]}
        zoom={8}
        scrollWheelZoom={false}
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {supplier && (
          <Marker position={[supplier.lat, supplier.lng]}>
            <Popup>
              <div className="text-xs font-sans">
                <strong className="text-eco-600 block">{supplier.name} (SUPPLIER)</strong>
                <span>District: {supplier.district}</span>
              </div>
            </Popup>
          </Marker>
        )}

        {partners.map((p) => (
          <React.Fragment key={p.id}>
            <Marker position={[p.lat, p.lng]}>
              <Popup>
                <div className="text-xs font-sans">
                  <strong className="block">{p.name} (PARTNER)</strong>
                  <span>District: {p.district}</span>
                </div>
              </Popup>
            </Marker>

            {supplier && (
              <Polyline
                positions={[
                  [supplier.lat, supplier.lng],
                  [p.lat, p.lng],
                ]}
                color="#22c55e"
                weight={2}
                dashArray="5, 8"
              />
            )}
          </React.Fragment>
        ))}
      </MapContainer>
    </div>
  );
};
