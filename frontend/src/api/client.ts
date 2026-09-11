const API_BASE = 'http://localhost:8000/api/v1';

function formatErrorDetail(detail: unknown): string {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === 'object' && 'msg' in item) {
          const field = Array.isArray((item as any).loc) ? (item as any).loc.at(-1) : undefined;
          return field ? `${field}: ${(item as any).msg}` : String((item as any).msg);
        }
        return String(item);
      })
      .join('; ');
  }
  return '';
}

export async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = localStorage.getItem('ecolink_token');

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'An error occurred' }));
    throw new Error(formatErrorDetail(errorData.detail) || `HTTP Error ${response.status}`);
  }

  return response.json();
}
