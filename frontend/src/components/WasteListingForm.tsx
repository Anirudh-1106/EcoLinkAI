import React, { useEffect, useMemo, useState } from 'react';
import { X, Loader2 } from 'lucide-react';
import { fetchApi } from '../api/client';
import { Material, Plant, WasteListing } from '../types';

// Mirrors QuantityUnit in the backend. Only kg and ton convert to a mass, so
// the others cannot be compared against a requirement measured by weight --
// the label says so rather than letting someone pick one and wonder why the
// match quality is poor.
const UNITS: { value: string; label: string; comparable: boolean }[] = [
  { value: 'kg', label: 'Kilograms (kg)', comparable: true },
  { value: 'ton', label: 'Tonnes (ton)', comparable: true },
  { value: 'liter', label: 'Litres', comparable: false },
  { value: 'cubic_meter', label: 'Cubic metres', comparable: false },
  { value: 'piece', label: 'Pieces', comparable: false },
  { value: 'meter', label: 'Metres', comparable: false },
];

interface Props {
  plants: Plant[];
  existing?: WasteListing | null;
  onClose: () => void;
  onSaved: () => void;
}

export const WasteListingForm: React.FC<Props> = ({ plants, existing, onClose, onSaved }) => {
  const isEdit = Boolean(existing);
  const today = new Date().toISOString().slice(0, 10);

  const [plantId, setPlantId] = useState(existing?.plant_id ?? plants[0]?.id ?? '');
  const [materialId, setMaterialId] = useState(existing?.material_id ?? '');
  const [materialQuery, setMaterialQuery] = useState(existing?.material_name ?? '');
  const [materials, setMaterials] = useState<Material[]>([]);
  const [showMaterials, setShowMaterials] = useState(false);

  const [quantity, setQuantity] = useState(existing ? String(existing.quantity) : '');
  const [unit, setUnit] = useState(existing?.unit ?? 'kg');
  const [purity, setPurity] = useState(existing?.purity_percentage != null ? String(existing.purity_percentage) : '');
  const [moisture, setMoisture] = useState(existing?.moisture_percentage != null ? String(existing.moisture_percentage) : '');
  const [grade, setGrade] = useState(existing?.quality_grade ?? '');
  const [price, setPrice] = useState(existing?.price_per_unit != null ? String(existing.price_per_unit) : '');
  const [availableFrom, setAvailableFrom] = useState(existing?.available_from ?? today);
  const [availableUntil, setAvailableUntil] = useState(existing?.available_until ?? '');
  const [description, setDescription] = useState(existing?.description ?? '');

  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  // Material lookup. The catalogue is far too large for a dropdown, so it is
  // searched rather than listed.
  useEffect(() => {
    if (isEdit || materialQuery.trim().length < 2) {
      setMaterials([]);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(async () => {
      try {
        const data = await fetchApi<{ items: Material[] }>(
          `/materials?search=${encodeURIComponent(materialQuery)}&page_size=20`,
        );
        if (!cancelled) setMaterials(data.items);
      } catch {
        if (!cancelled) setMaterials([]);
      }
    }, 300);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [materialQuery, isEdit]);

  const validation = useMemo(() => {
    if (!isEdit && !plantId) return 'Choose which plant this waste comes from';
    if (!isEdit && !materialId) return 'Choose a material from the list';
    const q = Number(quantity);
    if (!quantity.trim() || Number.isNaN(q) || q <= 0) return 'Quantity must be greater than zero';
    for (const [label, value] of [['Purity', purity], ['Moisture', moisture]] as const) {
      if (value.trim()) {
        const n = Number(value);
        if (Number.isNaN(n) || n < 0 || n > 100) return `${label} must be between 0 and 100`;
      }
    }
    if (price.trim() && (Number.isNaN(Number(price)) || Number(price) < 0)) {
      return 'Price cannot be negative';
    }
    if (!availableUntil) return 'Set the date this listing stays available until';
    if (!isEdit && availableUntil < availableFrom) return 'The end date cannot be before the start date';
    return null;
  }, [isEdit, plantId, materialId, quantity, purity, moisture, price, availableFrom, availableUntil]);

  const submit = async () => {
    if (validation) return;
    setSaving(true);
    setError('');

    // Optional fields are sent as null rather than empty strings so the
    // column stores "not measured" instead of a zero that would be read as a
    // real reading.
    const optionalNumber = (v: string) => (v.trim() ? Number(v) : null);

    try {
      if (isEdit && existing) {
        await fetchApi(`/waste-listings/${existing.id}`, {
          method: 'PUT',
          body: JSON.stringify({
            description: description.trim() || null,
            quantity: Number(quantity),
            purity_percentage: optionalNumber(purity),
            moisture_percentage: optionalNumber(moisture),
            quality_grade: grade.trim() || null,
            price_per_unit: optionalNumber(price),
            available_until: availableUntil,
          }),
        });
      } else {
        await fetchApi('/waste-listings', {
          method: 'POST',
          body: JSON.stringify({
            plant_id: plantId,
            material_id: materialId,
            description: description.trim() || null,
            quantity: Number(quantity),
            unit,
            purity_percentage: optionalNumber(purity),
            moisture_percentage: optionalNumber(moisture),
            quality_grade: grade.trim() || null,
            price_per_unit: optionalNumber(price),
            available_from: availableFrom,
            available_until: availableUntil,
          }),
        });
      }
      onSaved();
    } catch (err: any) {
      setError(err.message || 'Could not save this listing');
    } finally {
      setSaving(false);
    }
  };

  const selectedUnit = UNITS.find((u) => u.value === unit);
  const field = 'w-full bg-industrial-950 border border-industrial-800 rounded-lg px-3 py-2 text-sm text-white focus:border-eco-500 focus:outline-none';
  const label = 'block text-xs font-semibold text-industrial-400 uppercase tracking-wider mb-1';

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="bg-industrial-900 border border-industrial-700 rounded-2xl p-6 w-full max-w-2xl shadow-2xl space-y-4 my-8">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-lg font-bold text-white">
              {isEdit ? 'Edit listing' : 'List waste for sale'}
            </h3>
            <p className="text-xs text-industrial-400 mt-0.5">
              {isEdit
                ? 'Material, unit and start date are fixed once buyers can see the listing.'
                : 'Buyers find this through AI recommendations, so the more accurate it is the better it matches.'}
            </p>
          </div>
          <button onClick={onClose} className="text-industrial-500 hover:text-white" aria-label="Close">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Plant + material */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <label className={label}>Plant *</label>
            <select
              value={plantId}
              disabled={isEdit}
              onChange={(e) => setPlantId(e.target.value)}
              className={`${field} disabled:text-industrial-500`}
            >
              {plants.length === 0 && <option value="">No plants registered</option>}
              {plants.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.plant_name} ({p.district})
                </option>
              ))}
            </select>
          </div>

          <div className="relative">
            <label className={label}>Material *</label>
            <input
              type="text"
              value={materialQuery}
              disabled={isEdit}
              placeholder="Search, e.g. rubber"
              onChange={(e) => {
                setMaterialQuery(e.target.value);
                setMaterialId('');
                setShowMaterials(true);
              }}
              onFocus={() => setShowMaterials(true)}
              className={`${field} disabled:text-industrial-500`}
            />
            {showMaterials && materials.length > 0 && !materialId && (
              <div className="absolute z-10 mt-1 w-full max-h-48 overflow-y-auto bg-industrial-950 border border-industrial-700 rounded-lg shadow-xl">
                {materials.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => {
                      setMaterialId(m.id);
                      setMaterialQuery(m.material_name);
                      setShowMaterials(false);
                    }}
                    className="w-full text-left px-3 py-2 text-xs text-industrial-200 hover:bg-industrial-800"
                  >
                    {m.material_name}
                    <span className="text-industrial-500"> · {m.material_category}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Quantity + unit */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <label className={label}>Quantity *</label>
            <input type="number" min="0" step="any" value={quantity}
              onChange={(e) => setQuantity(e.target.value)} className={field} />
          </div>
          <div>
            <label className={label}>Unit *</label>
            <select
              value={unit}
              disabled={isEdit}
              onChange={(e) => setUnit(e.target.value)}
              className={`${field} disabled:text-industrial-500`}
            >
              {UNITS.map((u) => (
                <option key={u.value} value={u.value}>{u.label}</option>
              ))}
            </select>
            {selectedUnit && !selectedUnit.comparable && (
              <p className="text-[11px] text-amber-400/90 mt-1">
                Buyers usually ask by weight. This unit can't be converted to one
                without a density, so quantity matching will be approximate.
              </p>
            )}
          </div>
        </div>

        {/* Quality */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          <div>
            <label className={label}>Purity %</label>
            <input type="number" min="0" max="100" step="any" value={purity}
              onChange={(e) => setPurity(e.target.value)} placeholder="optional" className={field} />
          </div>
          <div>
            <label className={label}>Moisture %</label>
            <input type="number" min="0" max="100" step="any" value={moisture}
              onChange={(e) => setMoisture(e.target.value)} placeholder="optional" className={field} />
          </div>
          <div>
            <label className={label}>Grade</label>
            <input type="text" maxLength={20} value={grade}
              onChange={(e) => setGrade(e.target.value)} placeholder="e.g. A" className={field} />
          </div>
        </div>

        {/* Price + dates */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div>
            <label className={label}>Price per {unit}</label>
            <input type="number" min="0" step="any" value={price}
              onChange={(e) => setPrice(e.target.value)} placeholder="optional" className={field} />
          </div>
          <div>
            <label className={label}>Available from *</label>
            <input type="date" value={availableFrom} disabled={isEdit}
              onChange={(e) => setAvailableFrom(e.target.value)}
              className={`${field} disabled:text-industrial-500`} />
          </div>
          <div>
            <label className={label}>Available until *</label>
            <input type="date" value={availableUntil} min={availableFrom}
              onChange={(e) => setAvailableUntil(e.target.value)} className={field} />
          </div>
        </div>

        <div>
          <label className={label}>Description</label>
          <textarea rows={2} maxLength={1000} value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Handling notes, packaging, contamination, collection constraints…"
            className={`${field} resize-none`} />
        </div>

        {(error || validation) && (
          <div className={`rounded-lg px-3 py-2 text-xs ${error ? 'bg-red-950/40 border border-red-500/30 text-red-300' : 'text-industrial-400'}`}>
            {error || validation}
          </div>
        )}

        <div className="flex justify-end gap-2 pt-1">
          <button onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-semibold text-industrial-300 hover:text-white">
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={Boolean(validation) || saving}
            className="bg-eco-600 hover:bg-eco-500 disabled:bg-industrial-700 disabled:text-industrial-500 disabled:cursor-not-allowed text-white font-semibold text-xs px-4 py-2.5 rounded-xl transition-all flex items-center gap-2"
          >
            {saving && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
            {isEdit ? 'Save changes' : 'Publish listing'}
          </button>
        </div>
      </div>
    </div>
  );
};
