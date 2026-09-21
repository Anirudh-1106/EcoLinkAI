export interface User {
  id: string;
  email: string;
  full_name: string;
  role: string;
  company_id?: string;
}

export interface Company {
  id: string;
  company_name: string;
  industry_type: string;
  registration_number: string;
  gst_number: string;
  license_number: string;
  verification_status: string;
  trust_score: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  plant_count?: number;
}

export interface Plant {
  id: string;
  company_id: string;
  plant_name: string;
  plant_type: string;
  address: string;
  district: string;
  state: string;
  country: string;
  latitude: number;
  longitude: number;
  is_active: boolean;
  company_name?: string;
}

export interface Material {
  id: string;
  material_name: string;
  material_category: string;
  hazard_class: string;
  default_density?: number;
  carbon_factor?: number;
  description?: string;
  is_active: boolean;
}

export interface WasteListing {
  id: string;
  plant_id: string;
  material_id: string;
  description?: string;
  quantity: number;
  unit: string;
  purity_percentage?: number;
  moisture_percentage?: number;
  quality_grade?: string;
  price_per_unit?: number;
  available_from: string;
  available_until: string;
  status: string;
  created_at: string;
  plant_name?: string;
  plant_district?: string;
  plant_latitude?: number;
  plant_longitude?: number;
  material_name?: string;
  company_name?: string;
  company_id?: string;
}

export interface Requirement {
  id: string;
  plant_id: string;
  material_id: string;
  quantity: number;
  unit: string;
  minimum_purity?: number;
  maximum_budget_per_unit?: number;
  required_before: string;
  preferred_max_distance_km?: number;
  notes?: string;
  status: string;
  created_at: string;
  plant_name?: string;
  material_name?: string;
  company_name?: string;
}

export interface PartnerExplanation {
  material_compatibility: string;
  quantity_match_pct: number;
  quality_match_pct?: number;
  distance_km: number;
  transport_feasibility: string;
  estimated_transport_cost: number;
  trust_score: number;
  carbon_benefit_kg: number;
  historical_exchange_count: number;
  recommendation_summary: string;
}

export interface PartnerCard {
  rank: number;
  ai_score: number;
  model_type: string;
  company_id: string;
  company_name: string;
  plant_id: string;
  plant_name: string;
  plant_district: string;
  plant_state: string;
  plant_latitude: number;
  plant_longitude: number;
  requirement_id?: string;
  required_quantity?: number;
  required_purity?: number;
  waste_listing_id?: string;
  listing_quantity?: number;
  listing_purity?: number;
  listing_price_per_unit?: number;
  listing_unit?: string;
  listing_quality_grade?: string;
  material_name: string;
  compatibility_score: number;
  distance_km: number;
  estimated_transport_cost: number;
  estimated_carbon_saving: number;
  explanation: PartnerExplanation;
  alternative_lots?: AlternativeLot[];
}

/** Another lot of the same material from a seller already on the shortlist. */
export interface AlternativeLot {
  waste_listing_id: string;
  ai_score: number;
  quantity?: number | null;
  unit?: string | null;
  price_per_unit?: number | null;
  purity?: number | null;
  quantity_match_pct: number;
  estimated_transport_cost: number;
  estimated_carbon_saving: number;
}

/** What a given quantity would cost and save, priced by the server. */
export interface ExchangeRequestPreview {
  requested_quantity: number;
  unit: string;
  available_quantity: number;
  quantity_kg?: number | null;
  estimated_transport_cost: number;
  estimated_carbon_emission: number;
  estimated_carbon_saving: number;
  estimated_total_price?: number | null;
  distance_km: number;
}

export interface RecommendationResponse {
  waste_listing_id: string;
  material_name: string;
  supplier_plant_name: string;
  supplier_plant_latitude: number;
  supplier_plant_longitude: number;
  total_candidates: number;
  recommendations: PartnerCard[];
  model_version: string;
  inference_time_ms?: number;
}

export interface RequirementRecommendationResponse {
  requirement_id: string;
  material_name: string;
  buyer_plant_name: string;
  buyer_plant_latitude: number;
  buyer_plant_longitude: number;
  total_candidates: number;
  recommendations: PartnerCard[];
  model_version: string;
  inference_time_ms?: number;
}

export interface SearchRecommendationResponse {
  material_query: string;
  material_name: string;
  buyer_plant_name: string;
  buyer_plant_latitude: number;
  buyer_plant_longitude: number;
  total_candidates: number;
  recommendations: PartnerCard[];
  model_version: string;
  inference_time_ms?: number;
}

export interface ExchangeRequest {
  id: string;
  supplier_plant_id: string;
  buyer_plant_id: string;
  waste_listing_id: string;
  requirement_id?: string;
  compatibility_score: number;
  ai_confidence_score: number;
  recommendation_rank: number;
  distance_km: number;
  estimated_transport_cost: number;
  estimated_carbon_emission: number;
  estimated_carbon_saving?: number;
  recommendation_reason?: string;
  status: string;
  created_at: string;
  supplier_plant_name?: string;
  buyer_plant_name?: string;
  supplier_company_id?: string;
  buyer_company_id?: string;
  supplier_company_name?: string;
  buyer_company_name?: string;
  material_name?: string;
  waste_quantity?: number;
  /** How much this request is for, not the listing's whole stock. */
  requested_quantity?: number;
  listing_unit?: string;
}

export interface Exchange {
  id: string;
  exchange_request_id: string;
  exchange_status: string;
  shipment_status: string;
  agreed_price: number;
  transport_cost: number;
  actual_quantity?: number;
  actual_carbon_emission?: number;
  actual_carbon_saving?: number;
  expected_delivery_date?: string;
  delivered_at?: string;
  completion_notes?: string;
  created_at: string;
  supplier_plant_name?: string;
  buyer_plant_name?: string;
  material_name?: string;
  supplier_company_id?: string;
  buyer_company_id?: string;
  requested_quantity?: number;
  unit?: string;
}

export interface Review {
  id: string;
  exchange_id: string;
  /** Rating *of* the supplier, given by the buyer. Null until they leave it. */
  supplier_rating?: number | null;
  /** Rating *of* the buyer, given by the supplier. */
  buyer_rating?: number | null;
  supplier_feedback?: string;
  buyer_feedback?: string;
  created_at: string;
  supplier_plant_name?: string;
  buyer_plant_name?: string;
  supplier_company_name?: string;
  buyer_company_name?: string;
  material_name?: string;
  quantity?: number;
  unit?: string;
  /** Which side the viewer was on: decides which rating they may give. */
  my_role?: 'supplier' | 'buyer' | null;
  my_rating_submitted: boolean;
  counterparty_rating_submitted: boolean;
}

export interface DashboardStats {
  total_plants: number;
  active_waste_listings: number;
  open_requirements: number;
  pending_exchange_requests: number;
  completed_exchanges: number;
  total_carbon_saved_kg: number;
  trust_score: number;
  average_rating: number;
}

export interface PlatformAnalytics {
  total_companies: number;
  total_plants: number;
  total_materials: number;
  total_waste_listings: number;
  total_requirements: number;
  total_exchange_requests: number;
  total_exchanges: number;
  total_reviews: number;
  total_waste_exchanged_tons: number;
  total_carbon_saved_kg: number;
  total_revenue: number;
  average_rating: number;
  exchange_success_rate: number;
}

export interface MaterialDistribution {
  category: string;
  count: number;
  total_quantity: number;
}

export interface AIRuntimeStatus {
  model_loaded: boolean;
  graph_cached: boolean;
  node_count: number;
  edge_count: number;
  age_seconds: number | null;
  ttl_seconds: number;
  last_error: string | null;
}

export interface AIModelMetrics {
  model_version: string;
  last_trained?: string;
  training_samples: number;
  precision_at_5: number;
  recall_at_5: number;
  ndcg_at_5: number;
  baseline_ndcg_at_5: number;
  total_recommendations: number;
  recommendation_to_exchange_rate: number;
}
