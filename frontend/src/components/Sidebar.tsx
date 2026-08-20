import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Factory,
  Trash2,
  ClipboardList,
  Sparkles,
  ArrowRightLeft,
  Truck,
  Star,
  BarChart3,
  ShieldCheck,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export const Sidebar: React.FC = () => {
  const { user } = useAuth();

  const navItems = [
    { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { to: '/plants', label: 'Plants', icon: Factory },
    { to: '/waste-listings', label: 'Waste Listings', icon: Trash2 },
    { to: '/requirements', label: 'Requirements', icon: ClipboardList },
    { to: '/recommendations', label: 'AI Recommendations', icon: Sparkles, highlight: true },
    { to: '/exchange-requests', label: 'Exchange Requests', icon: ArrowRightLeft },
    { to: '/transactions', label: 'Transactions', icon: Truck },
    { to: '/reviews', label: 'Reviews', icon: Star },
    { to: '/analytics', label: 'Analytics', icon: BarChart3 },
  ];

  if (user?.role === 'admin') {
    navItems.push({ to: '/admin', label: 'Admin Panel', icon: ShieldCheck });
  }

  return (
    <aside className="w-64 bg-industrial-900 border-r border-industrial-800 p-4 flex flex-col justify-between min-h-[calc(100vh-61px)] text-industrial-300">
      <div className="space-y-1">
        <div className="px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-industrial-400">
          Navigation
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-eco-600/15 text-eco-400 border border-eco-500/30 font-semibold shadow-sm'
                    : item.highlight
                    ? 'text-eco-300 hover:bg-eco-950/40 hover:text-eco-400'
                    : 'text-industrial-300 hover:bg-industrial-800 hover:text-white'
                }`
              }
            >
              <Icon className={`w-4 h-4 ${item.highlight ? 'text-eco-400 animate-pulse' : ''}`} />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </div>

      <div className="bg-industrial-950/60 border border-industrial-800 rounded-xl p-3.5 text-xs text-industrial-400 space-y-1">
        <div className="flex items-center gap-1.5 text-eco-400 font-semibold">
          <Sparkles className="w-3.5 h-3.5" />
          <span>MC-GNN v1.0 Active</span>
        </div>
        <p className="text-[11px] leading-relaxed text-industrial-400">
          Multi-Channel Graph Neural Network scoring & adaptive fusion enabled.
        </p>
      </div>
    </aside>
  );
};
