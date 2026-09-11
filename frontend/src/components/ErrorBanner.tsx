import React from 'react';
import { AlertCircle } from 'lucide-react';

export const ErrorBanner: React.FC<{ message: string }> = ({ message }) => (
  <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 text-xs text-red-400 flex items-center gap-2">
    <AlertCircle className="w-4 h-4 shrink-0" />
    <span>{message}</span>
  </div>
);

export const EmptyState: React.FC<{ message: string }> = ({ message }) => (
  <div className="bg-industrial-900 border border-industrial-800 rounded-2xl p-10 text-center text-sm text-industrial-400">
    {message}
  </div>
);
