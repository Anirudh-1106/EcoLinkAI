import React from 'react';
import { LucideIcon } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  color?: 'eco' | 'blue' | 'amber' | 'emerald' | 'purple';
}

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  color = 'eco',
}) => {
  const colorMap = {
    eco: 'bg-eco-500/10 text-eco-400 border-eco-500/20',
    blue: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    amber: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    emerald: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    purple: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
  };

  return (
    <div className="bg-industrial-900 border border-industrial-800 rounded-2xl p-5 shadow-sm hover:border-industrial-700 transition-all">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-industrial-400 uppercase tracking-wider">{title}</span>
        <div className={`p-2.5 rounded-xl border ${colorMap[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
      <div className="mt-3">
        <div className="text-2xl font-bold text-white tracking-tight">{value}</div>
        {subtitle && <p className="text-xs text-industrial-400 mt-1">{subtitle}</p>}
      </div>
    </div>
  );
};
