import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Leaf, LogIn, AlertCircle } from 'lucide-react';
import { fetchApi } from '../api/client';
import { useAuth } from '../context/AuthContext';

export const LoginPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const formData = new URLSearchParams();
      formData.append('username', email);
      formData.append('password', password);

      const response = await fetch('http://localhost:8000/api/v1/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Login failed');
      }

      const data = await response.json();
      login(data.access_token, {
        id: data.user_id,
        email: data.email,
        full_name: data.full_name,
        role: data.role,
        company_id: data.company_id,
      });
      navigate('/dashboard');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-industrial-950 flex flex-col justify-center items-center p-6 text-white font-sans">
      <div className="w-full max-w-md bg-industrial-900 border border-industrial-800 rounded-3xl p-8 shadow-2xl">
        <div className="text-center mb-8">
          <div className="inline-flex bg-eco-600 p-3 rounded-2xl text-white shadow-lg shadow-eco-600/30 mb-3">
            <Leaf className="w-8 h-8" />
          </div>
          <h1 className="text-2xl font-bold">Welcome Back</h1>
          <p className="text-xs text-industrial-400 mt-1">Sign in to your EcoLinkAI company portal</p>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 text-xs text-red-400 mb-6 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-industrial-300 uppercase tracking-wider mb-1.5">
              Work Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              placeholder="manager@company.com"
              className="w-full bg-industrial-950 border border-industrial-800 rounded-xl px-4 py-3 text-sm text-white placeholder-industrial-500 focus:outline-none focus:border-eco-500 transition-colors"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-industrial-300 uppercase tracking-wider mb-1.5">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              placeholder="••••••••"
              className="w-full bg-industrial-950 border border-industrial-800 rounded-xl px-4 py-3 text-sm text-white placeholder-industrial-500 focus:outline-none focus:border-eco-500 transition-colors"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-eco-600 hover:bg-eco-500 text-white font-semibold py-3 rounded-xl transition-all shadow-lg shadow-eco-600/20 flex items-center justify-center gap-2 mt-6 text-sm"
          >
            <LogIn className="w-4 h-4" />
            <span>{loading ? 'Signing in...' : 'Sign In'}</span>
          </button>
        </form>

        <div className="mt-6 text-center text-xs text-industrial-400">
          Demo company login: <code className="text-eco-400 bg-industrial-950 px-2 py-1 rounded">saran-sankaran@industry.in / password123</code>
        </div>

        <div className="mt-6 border-t border-industrial-800 pt-6 text-center text-xs text-industrial-400">
          Don't have a company account?{' '}
          <Link to="/register" className="text-eco-400 font-semibold hover:underline">
            Register Company
          </Link>
        </div>
      </div>
    </div>
  );
};
