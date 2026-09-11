import React from 'react';
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';

import { LandingPage } from './pages/LandingPage';
import { LoginPage } from './pages/LoginPage';
import { RegisterPage } from './pages/RegisterPage';
import { DashboardPage } from './pages/DashboardPage';
import { PlantsPage } from './pages/PlantsPage';
import { WasteListingsPage } from './pages/WasteListingsPage';
import { RequirementsPage } from './pages/RequirementsPage';
import { RecommendationsPage } from './pages/RecommendationsPage';
import { ExchangeRequestsPage } from './pages/ExchangeRequestsPage';
import { TransactionsPage } from './pages/TransactionsPage';
import { ReviewsPage } from './pages/ReviewsPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { AdminPage } from './pages/AdminPage';

const AppLayout: React.FC = () => {
  return (
    <div className="min-h-screen bg-industrial-950 text-white flex flex-col font-sans">
      <Navbar />
      <div className="flex flex-1">
        <Sidebar />
        <main className="flex-1 p-6 overflow-y-auto max-w-7xl">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

const ProtectedRoute: React.FC = () => {
  const { isAuthenticated } = useAuth();
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return <AppLayout />;
};

const AdminRoute: React.FC = () => {
  const { user } = useAuth();
  if (user && user.role !== 'admin') return <Navigate to="/dashboard" replace />;
  return <AdminPage />;
};

export const App: React.FC = () => {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Routes */}
          <Route path="/" element={<LandingPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          {/* Protected Portal Routes */}
          <Route element={<ProtectedRoute />}>
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/plants" element={<PlantsPage />} />
            <Route path="/waste-listings" element={<WasteListingsPage />} />
            <Route path="/requirements" element={<RequirementsPage />} />
            <Route path="/recommendations" element={<RecommendationsPage />} />
            <Route path="/exchange-requests" element={<ExchangeRequestsPage />} />
            <Route path="/transactions" element={<TransactionsPage />} />
            <Route path="/reviews" element={<ReviewsPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/admin" element={<AdminRoute />} />
          </Route>

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
};

export default App;
