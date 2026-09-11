import React, { useEffect, useState } from 'react';
import { Star, MessageSquare, ShieldCheck, CheckCircle2 } from 'lucide-react';
import { fetchApi } from '../api/client';
import { Review, Exchange } from '../types';
import { ErrorBanner, EmptyState } from '../components/ErrorBanner';

export const ReviewsPage: React.FC = () => {
  const [reviews, setReviews] = useState<Review[]>([]);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [selectedExchangeId, setSelectedExchangeId] = useState('');
  const [supplierRating, setSupplierRating] = useState(5);
  const [buyerRating, setBuyerRating] = useState(5);
  const [comment, setComment] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  const [error, setError] = useState('');

  const loadData = () => {
    fetchApi<{ items: Review[] }>('/reviews')
      .then((d) => setReviews(d.items))
      .catch((err) => setError(err.message || 'Failed to load reviews'));
    fetchApi<{ items: Exchange[] }>('/exchanges')
      .then((d) => {
        const completed = d.items.filter((ex) => ex.exchange_status === 'Completed');
        setExchanges(completed);
        if (completed.length > 0) setSelectedExchangeId(completed[0].id);
      })
      .catch((err) => setError(err.message || 'Failed to load exchanges'));
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSuccessMsg('');

    try {
      await fetchApi('/reviews', {
        method: 'POST',
        body: JSON.stringify({
          exchange_id: selectedExchangeId,
          supplier_rating: supplierRating,
          buyer_rating: buyerRating,
          supplier_feedback: comment,
          buyer_feedback: comment,
        }),
      });
      setSuccessMsg('Review submitted! Company trust score updated.');
      loadData();
    } catch (err: any) {
      alert(`Review submission error: ${err.message}`);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Reviews & Trust Feedback</h1>
          <p className="text-xs text-industrial-400">Post-transaction reviews directly inform AI partner trust scoring</p>
        </div>
      </div>

      {error && <ErrorBanner message={error} />}

      {/* Form */}
      <div className="bg-industrial-900 border border-industrial-800 rounded-2xl p-6 shadow-sm max-w-2xl space-y-4">
        <h2 className="text-base font-bold text-white flex items-center gap-2">
          <Star className="w-5 h-5 text-amber-400" />
          <span>Submit Post-Exchange Feedback</span>
        </h2>

        {successMsg && (
          <div className="bg-eco-600/15 border border-eco-500/30 rounded-xl p-3 text-xs text-eco-300 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-eco-400" />
            <span>{successMsg}</span>
          </div>
        )}

        {exchanges.length === 0 ? (
          <p className="text-xs text-industrial-400">
            No completed exchanges yet. Reviews can be submitted once a transaction is marked completed.
          </p>
        ) : (
        <form onSubmit={handleSubmit} className="space-y-4 text-xs">
          <div>
            <label className="block font-semibold text-industrial-300 mb-1">Select Completed Exchange</label>
            <select
              value={selectedExchangeId}
              onChange={(e) => setSelectedExchangeId(e.target.value)}
              className="w-full bg-industrial-950 border border-industrial-800 rounded-xl px-3 py-2.5 text-white"
            >
              {exchanges.map((ex) => (
                <option key={ex.id} value={ex.id}>
                  Exchange #{ex.id.slice(0, 8)} — {ex.supplier_plant_name || 'Supplier'} ↔ {ex.buyer_plant_name || 'Buyer'}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block font-semibold text-industrial-300 mb-1">Supplier Rating (1-5)</label>
              <select
                value={supplierRating}
                onChange={(e) => setSupplierRating(Number(e.target.value))}
                className="w-full bg-industrial-950 border border-industrial-800 rounded-xl px-3 py-2 text-white"
              >
                {[5, 4, 3, 2, 1].map((r) => (
                  <option key={r} value={r}>{r} Stars</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block font-semibold text-industrial-300 mb-1">Buyer Rating (1-5)</label>
              <select
                value={buyerRating}
                onChange={(e) => setBuyerRating(Number(e.target.value))}
                className="w-full bg-industrial-950 border border-industrial-800 rounded-xl px-3 py-2 text-white"
              >
                {[5, 4, 3, 2, 1].map((r) => (
                  <option key={r} value={r}>{r} Stars</option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className="block font-semibold text-industrial-300 mb-1">Review Comments</label>
            <textarea
              rows={3}
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              required
              placeholder="High quality material, prompt logistics pickup..."
              className="w-full bg-industrial-950 border border-industrial-800 rounded-xl p-3 text-white placeholder-industrial-500"
            />
          </div>

          <button
            type="submit"
            className="bg-eco-600 hover:bg-eco-500 text-white font-bold px-5 py-2.5 rounded-xl transition-all shadow-md"
          >
            Submit Feedback
          </button>
        </form>
        )}
      </div>

      {/* Review List */}
      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-industrial-300 uppercase tracking-wider">Historical Exchange Feedback</h2>
        {reviews.length === 0 && <EmptyState message="No reviews submitted yet." />}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {reviews.map((rev) => (
            <div key={rev.id} className="bg-industrial-900 border border-industrial-800 rounded-2xl p-4 space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-bold text-white">Exchange #{rev.exchange_id.slice(0, 8)}</span>
                <span className="text-amber-400 font-bold flex items-center gap-1">
                  <Star className="w-3.5 h-3.5 fill-amber-400" />
                  <span>{rev.supplier_rating}.0 / 5.0</span>
                </span>
              </div>
              <p className="text-industrial-300 italic bg-industrial-950/60 p-2.5 rounded-xl border border-industrial-800">
                "{rev.supplier_feedback || rev.buyer_feedback}"
              </p>
              <p className="text-industrial-400 text-[11px]">{new Date(rev.created_at).toLocaleDateString()}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
