import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Leaf, UserPlus, AlertCircle } from 'lucide-react';
import { fetchApi } from '../api/client';
import { useAuth } from '../context/AuthContext';

export const RegisterPage: React.FC = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    full_name: '',
    company_name: '',
    industry_type: 'Metal & Metallurgy',
    registration_number: '',
    gst_number: '',
    license_number: '',
  });

  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const { login } = useAuth();
  const navigate = useNavigate();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response: any = await fetchApi('/auth/register', {
        method: 'POST',
        body: JSON.stringify(formData),
      });

      login(response.access_token, {
        id: response.user_id,
        email: response.email,
        full_name: response.full_name,
        role: response.role,
        company_id: response.company_id,
      });

      navigate('/dashboard');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-industrial-950 flex flex-col justify-center items-center p-6 text-white font-sans py-12">
      <div className="w-full max-w-xl bg-industrial-900 border border-industrial-800 rounded-3xl p-8 shadow-2xl">
        <div className="text-center mb-8">
          <div className="inline-flex bg-eco-600 p-3 rounded-2xl text-white shadow-lg shadow-eco-600/30 mb-3">
            <Leaf className="w-8 h-8" />
          </div>
          <h1 className="text-2xl font-bold">Register Your Industrial Entity</h1>
          <p className="text-xs text-industrial-400 mt-1">Join the AI-powered circular waste exchange network</p>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 text-xs text-red-400 mb-6 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-industrial-300 uppercase tracking-wider mb-1">
                Full Name
              </label>
              <input
                type="text"
                name="full_name"
                value={formData.full_name}
                onChange={handleChange}
                required
                placeholder="John Doe"
                className="w-full bg-industrial-950 border border-industrial-800 rounded-xl px-4 py-2.5 text-sm text-white placeholder-industrial-500 focus:outline-none focus:border-eco-500"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-industrial-300 uppercase tracking-wider mb-1">
                Work Email
              </label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                placeholder="manager@company.com"
                className="w-full bg-industrial-950 border border-industrial-800 rounded-xl px-4 py-2.5 text-sm text-white placeholder-industrial-500 focus:outline-none focus:border-eco-500"
              />
            </div>
          </div>

          <div className="grid sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-industrial-300 uppercase tracking-wider mb-1">
                Company Name
              </label>
              <input
                type="text"
                name="company_name"
                value={formData.company_name}
                onChange={handleChange}
                required
                placeholder="Apex Metals Pvt Ltd"
                className="w-full bg-industrial-950 border border-industrial-800 rounded-xl px-4 py-2.5 text-sm text-white placeholder-industrial-500 focus:outline-none focus:border-eco-500"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-industrial-300 uppercase tracking-wider mb-1">
                Industry Sector
              </label>
              <select
                name="industry_type"
                value={formData.industry_type}
                onChange={handleChange}
                className="w-full bg-industrial-950 border border-industrial-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-eco-500"
              >
                <option value="Metal & Metallurgy">Metal & Metallurgy</option>
                <option value="Chemicals & Petrochemicals">Chemicals & Petrochemicals</option>
                <option value="Textiles & Apparel">Textiles & Apparel</option>
                <option value="Food Processing">Food Processing</option>
                <option value="Paper & Packaging">Paper & Packaging</option>
                <option value="Automotive">Automotive</option>
                <option value="General Manufacturing">General Manufacturing</option>
              </select>
            </div>
          </div>

          <div className="grid sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-semibold text-industrial-300 uppercase tracking-wider mb-1">
                CIN / Reg No
              </label>
              <input
                type="text"
                name="registration_number"
                value={formData.registration_number}
                onChange={handleChange}
                required
                placeholder="CIN12345678"
                className="w-full bg-industrial-950 border border-industrial-800 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-eco-500"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-industrial-300 uppercase tracking-wider mb-1">
                GST Number
              </label>
              <input
                type="text"
                name="gst_number"
                value={formData.gst_number}
                onChange={handleChange}
                required
                maxLength={15}
                placeholder="32ABCDE1234F1Z5"
                className="w-full bg-industrial-950 border border-industrial-800 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-eco-500"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-industrial-300 uppercase tracking-wider mb-1">
                Factory License
              </label>
              <input
                type="text"
                name="license_number"
                value={formData.license_number}
                onChange={handleChange}
                required
                placeholder="LIC-998877"
                className="w-full bg-industrial-950 border border-industrial-800 rounded-xl px-3 py-2.5 text-sm text-white focus:outline-none focus:border-eco-500"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-industrial-300 uppercase tracking-wider mb-1">
              Password
            </label>
            <input
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              required
              minLength={8}
              placeholder="••••••••"
              className="w-full bg-industrial-950 border border-industrial-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-eco-500"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-eco-600 hover:bg-eco-500 text-white font-semibold py-3 rounded-xl transition-all shadow-lg shadow-eco-600/20 flex items-center justify-center gap-2 mt-6 text-sm"
          >
            <UserPlus className="w-4 h-4" />
            <span>{loading ? 'Creating Account...' : 'Register Company'}</span>
          </button>
        </form>

        <div className="mt-6 border-t border-industrial-800 pt-6 text-center text-xs text-industrial-400">
          Already registered?{' '}
          <Link to="/login" className="text-eco-400 font-semibold hover:underline">
            Sign In
          </Link>
        </div>
      </div>
    </div>
  );
};
