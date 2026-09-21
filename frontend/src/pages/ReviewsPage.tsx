import React, { useEffect, useState } from 'react';
import { Star, MessageSquare, ShieldCheck, CheckCircle2 } from 'lucide-react';
import { fetchApi } from '../api/client';
import { Review, Exchange } from '../types';
import { ErrorBanner, EmptyState } from '../components/ErrorBanner';
import { useAuth } from '../context/AuthContext';

/** One direction of a review. A side that has not rated yet says so. */
const ReviewSide: React.FC<{
  label: string;
  rating?: number | null;
  feedback?: string;
}> = ({ label, rating, feedback }) => (
  <div className="bg-industrial-950/60 border border-industrial-800 rounded-xl p-2.5">
    <div className="flex items-center justify-between gap-2">
      <span className="text-[10px] uppercase tracking-wider text-industrial-400 font-semibold">{label}</span>
      {rating != null ? (
        <span className="text-amber-400 font-bold flex items-center gap-1 shrink-0">
          <Star className="w-3 h-3 fill-amber-400" />
          <span>{rating}.0 / 5.0</span>
        </span>
      ) : (
        <span className="text-industrial-500 shrink-0">Not yet rated</span>
      )}
    </div>
    {rating != null && feedback && (
      <p className="text-industrial-300 italic mt-1">"{feedback}"</p>
    )}
  </div>
);

export const ReviewsPage: React.FC = () => {
  const [reviews, setReviews] = useState<Review[]>([]);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [selectedExchangeId, setSelectedExchangeId] = useState('');
  const [rating, setRating] = useState(5);
  const [comment, setComment] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  const [error, setError] = useState('');
  const { user } = useAuth();

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

  const selected = exchanges.find((ex) => ex.id === selectedExchangeId);

  // You rate the other party, never yourself. Selling this exchange means
  // rating the buyer; buying it means rating the supplier.
  const myRole: 'supplier' | 'buyer' | null = !selected || !user?.company_id
    ? null
    : selected.supplier_company_id === user.company_id
      ? 'supplier'
      : selected.buyer_company_id === user.company_id
        ? 'buyer'
        : null;

  const counterpartyLabel =
    myRole === 'supplier'
      ? selected?.buyer_plant_name || 'the buyer'
      : selected?.supplier_plant_name || 'the supplier';

  const alreadyRated = reviews.some(
    (r) => r.exchange_id === selectedExchangeId && r.my_rating_submitted,
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSuccessMsg('');

    try {
      // Only the rating is sent. Which side it applies to is decided by the
      // server from the caller's role in the exchange.
      await fetchApi('/reviews', {
        method: 'POST',
        body: JSON.stringify({
          exchange_id: selectedExchangeId,
          rating,
          feedback: comment,
        }),
      });
      setSuccessMsg(`Review submitted. ${counterpartyLabel}'s trust score has been updated.`);
      setComment('');
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
                  {ex.material_name || 'Material'}
                  {ex.requested_quantity ? ` · ${ex.requested_quantity} ${ex.unit || ''}` : ''}
                  {' — '}
                  {ex.supplier_plant_name || 'Supplier'} → {ex.buyer_plant_name || 'Buyer'}
                </option>
              ))}
            </select>
          </div>

          {/* You rate the counterparty, not yourself */}
          <div>
            <label className="block font-semibold text-industrial-300 mb-1">
              {myRole === 'supplier' ? 'Rate the buyer' : 'Rate the supplier'}
              <span className="text-industrial-500 font-normal"> — {counterpartyLabel}</span>
            </label>
            <select
              value={rating}
              onChange={(e) => setRating(Number(e.target.value))}
              className="w-full bg-industrial-950 border border-industrial-800 rounded-xl px-3 py-2 text-white"
            >
              {[5, 4, 3, 2, 1].map((r) => (
                <option key={r} value={r}>{r} Stars</option>
              ))}
            </select>
            <p className="text-[11px] text-industrial-500 mt-1">
              {myRole === 'supplier'
                ? 'You sold on this exchange, so you rate how the buyer handled it.'
                : 'You bought on this exchange, so you rate how the supplier handled it.'}
            </p>
          </div>

          <div>
            <label className="block font-semibold text-industrial-300 mb-1">Review Comments</label>
            <textarea
              rows={3}
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              required
              placeholder={
                myRole === 'supplier'
                  ? 'Paid on time, collection went smoothly...'
                  : 'High quality material, prompt logistics pickup...'
              }
              className="w-full bg-industrial-950 border border-industrial-800 rounded-xl p-3 text-white placeholder-industrial-500"
            />
          </div>

          {alreadyRated && (
            <p className="text-[11px] text-amber-400/90">
              You have already rated this exchange. Choose another to review.
            </p>
          )}

          <button
            type="submit"
            disabled={alreadyRated || !myRole}
            className="bg-eco-600 hover:bg-eco-500 disabled:bg-industrial-700 disabled:text-industrial-500 disabled:cursor-not-allowed text-white font-bold px-5 py-2.5 rounded-xl transition-all shadow-md"
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
            <div key={rev.id} className="bg-industrial-900 border border-industrial-800 rounded-2xl p-4 space-y-2.5 text-xs">
              <div>
                <span className="font-bold text-white">{rev.material_name || 'Material'}</span>
                {rev.quantity != null && (
                  <span className="text-industrial-400"> · {rev.quantity} {rev.unit || ''}</span>
                )}
                <p className="text-industrial-400 text-[11px] mt-0.5">
                  {rev.supplier_company_name || rev.supplier_plant_name || 'Supplier'}
                  {' → '}
                  {rev.buyer_company_name || rev.buyer_plant_name || 'Buyer'}
                  {rev.my_role && (
                    <span className="text-eco-400"> · you {rev.my_role === 'supplier' ? 'sold' : 'bought'}</span>
                  )}
                </p>
              </div>

              {/* Each direction shown separately, so it is clear who rated whom */}
              <div className="space-y-1.5">
                <ReviewSide
                  label="Supplier rated by buyer"
                  rating={rev.supplier_rating}
                  feedback={rev.supplier_feedback}
                />
                <ReviewSide
                  label="Buyer rated by supplier"
                  rating={rev.buyer_rating}
                  feedback={rev.buyer_feedback}
                />
              </div>

              <p className="text-industrial-400 text-[11px]">{new Date(rev.created_at).toLocaleDateString()}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
