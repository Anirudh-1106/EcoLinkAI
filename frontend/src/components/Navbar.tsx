import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Leaf, LogOut, User as UserIcon, ShieldCheck } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export const Navbar: React.FC = () => {
  const { user, logout, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  return (
    <nav className="bg-industrial-900 border-b border-industrial-800 text-white px-6 py-3.5 flex items-center justify-between sticky top-0 z-50 shadow-md">
      <Link to="/" className="flex items-center gap-2.5 group">
        <div className="bg-eco-600 p-2 rounded-xl text-white shadow-lg shadow-eco-600/30 group-hover:scale-105 transition-transform">
          <Leaf className="w-5 h-5" />
        </div>
        <div>
          <span className="font-bold text-xl tracking-tight text-white">EcoLink<span className="text-eco-400">AI</span></span>
          <span className="block text-[10px] uppercase tracking-wider text-industrial-400 font-semibold -mt-1">Industrial Symbiosis Platform</span>
        </div>
      </Link>

      <div className="flex items-center gap-4">
        {isAuthenticated && user ? (
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2.5 bg-industrial-800/80 border border-industrial-700 px-3.5 py-1.5 rounded-full text-xs text-industrial-200">
              <UserIcon className="w-4 h-4 text-eco-400" />
              <div>
                <span className="font-semibold block text-white">{user.full_name}</span>
                <span className="text-[10px] text-industrial-400 uppercase tracking-wider">{user.role}</span>
              </div>
            </div>
            <button
              onClick={() => { logout(); navigate('/login'); }}
              className="p-2 text-industrial-400 hover:text-red-400 hover:bg-industrial-800 rounded-lg transition-colors"
              title="Logout"
            >
              <LogOut className="w-5 h-5" />
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-3">
            <Link
              to="/login"
              className="text-sm font-medium text-industrial-300 hover:text-white px-3 py-1.5 transition-colors"
            >
              Sign In
            </Link>
            <Link
              to="/register"
              className="bg-eco-600 hover:bg-eco-500 text-white text-sm font-semibold px-4 py-2 rounded-lg transition-all shadow-md shadow-eco-600/20"
            >
              Register Company
            </Link>
          </div>
        )}
      </div>
    </nav>
  );
};
