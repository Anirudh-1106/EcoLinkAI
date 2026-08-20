import React from 'react';
import { Link } from 'react-router-dom';
import { Sparkles, ArrowRight, Leaf, ShieldCheck, Truck, Factory, BarChart3, Recycle, Database } from 'lucide-react';
import { Navbar } from '../components/Navbar';

export const LandingPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-industrial-950 text-white flex flex-col font-sans">
      <Navbar />

      {/* Hero Section */}
      <section className="relative py-20 px-6 max-w-6xl mx-auto text-center overflow-hidden">
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-eco-900/30 via-industrial-950 to-industrial-950"></div>
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-eco-950/80 border border-eco-500/30 text-eco-400 text-xs font-semibold uppercase tracking-wider mb-6">
          <Sparkles className="w-4 h-4" />
          <span>AI-Powered Circular Economy Platform</span>
        </div>

        <h1 className="text-4xl sm:text-6xl font-black tracking-tight text-white max-w-4xl mx-auto leading-tight">
          Turn Industrial Waste Into <span className="text-transparent bg-clip-text bg-gradient-to-r from-eco-400 to-emerald-300">Industrial Value</span>
        </h1>

        <p className="mt-6 text-lg sm:text-xl text-industrial-300 max-w-2xl mx-auto leading-relaxed">
          EcoLinkAI uses Multi-Channel Graph Neural Networks (MC-GNN) to discover optimal industrial symbiosis partners based on material compatibility, logistics feasibility, trust, and carbon benefits.
        </p>

        <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
          <Link
            to="/register"
            className="bg-eco-600 hover:bg-eco-500 text-white font-bold px-8 py-3.5 rounded-xl shadow-lg shadow-eco-600/30 transition-all flex items-center gap-2 text-base"
          >
            <span>Get Started</span>
            <ArrowRight className="w-5 h-5" />
          </Link>
          <Link
            to="/login"
            className="bg-industrial-800 hover:bg-industrial-700 text-industrial-200 font-semibold px-8 py-3.5 rounded-xl border border-industrial-700 transition-all text-base"
          >
            Explore Platform
          </Link>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto mt-16 text-center border-t border-industrial-800 pt-8">
          <div>
            <div className="text-3xl font-black text-white">40+</div>
            <div className="text-xs text-industrial-400 font-medium uppercase mt-1">Active Plants</div>
          </div>
          <div>
            <div className="text-3xl font-black text-eco-400">140+</div>
            <div className="text-xs text-industrial-400 font-medium uppercase mt-1">Material Types</div>
          </div>
          <div>
            <div className="text-3xl font-black text-white">345+</div>
            <div className="text-xs text-industrial-400 font-medium uppercase mt-1">Completed Exchanges</div>
          </div>
          <div>
            <div className="text-3xl font-black text-eco-400">89.5%</div>
            <div className="text-xs text-industrial-400 font-medium uppercase mt-1">AI Recommendation Precision</div>
          </div>
        </div>
      </section>

      {/* How it Works / MC-GNN Explanation */}
      <section className="py-16 px-6 bg-industrial-900/60 border-y border-industrial-800">
        <div className="max-w-6xl mx-auto">
          <div className="text-center max-w-2xl mx-auto mb-12">
            <h2 className="text-3xl font-bold text-white">Multi-Channel Graph Neural Network (MC-GNN)</h2>
            <p className="text-sm text-industrial-300 mt-2">
              Why traditional matching fails: Simple keyword or rule-based matching ignores complex multi-factor relationships like plant distance, capacity, transport cost, and trust history.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-industrial-900 border border-industrial-800 p-6 rounded-2xl">
              <div className="bg-eco-600/10 text-eco-400 border border-eco-500/20 p-3 rounded-xl w-fit mb-4">
                <Database className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-white mb-2">1. Dynamic Graph Representation</h3>
              <p className="text-xs text-industrial-300 leading-relaxed">
                Industrial plants are modeled as graph nodes, while historical exchanges, logistics routes, and material compatibilities form multi-dimensional graph edges.
              </p>
            </div>

            <div className="bg-industrial-900 border border-industrial-800 p-6 rounded-2xl">
              <div className="bg-blue-500/10 text-blue-400 border border-blue-500/20 p-3 rounded-xl w-fit mb-4">
                <Sparkles className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-white mb-2">2. Multi-Channel & HSIC Fusion</h3>
              <p className="text-xs text-industrial-300 leading-relaxed">
                Multiple GNN backbones (GCN, GAT, GraphSAGE) extract diverse structural views, regularized by HSIC to prevent over-squashing and adaptively fused.
              </p>
            </div>

            <div className="bg-industrial-900 border border-industrial-800 p-6 rounded-2xl">
              <div className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 p-3 rounded-xl w-fit mb-4">
                <Recycle className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-white mb-2">3. Explainable Partner Ranking</h3>
              <p className="text-xs text-industrial-300 leading-relaxed">
                Returns ranked partner recommendations backed by feature-level explanations: material compatibility, distance, transport cost, trust, and carbon benefit.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="mt-auto border-t border-industrial-800 py-6 px-6 text-center text-xs text-industrial-400">
        <p>© 2026 EcoLinkAI — AI-Powered Industrial Symbiosis Recommendation Network.</p>
      </footer>
    </div>
  );
};
