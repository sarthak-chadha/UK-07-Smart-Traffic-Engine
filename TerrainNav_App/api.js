// TerrainNav | api.js

const API = {

  // Route Prediction
  async predictRoute({ startName, endName, startLat, startLon, endLat, endLon }) {
    const payload = {};
    if (startLat !== undefined && startLon !== undefined) {
      payload.start_lat = parseFloat(startLat);
      payload.start_lon = parseFloat(startLon);
    } else {
      payload.start_name = startName;
    }
    if (endLat !== undefined && endLon !== undefined) {
      payload.end_lat = parseFloat(endLat);
      payload.end_lon = parseFloat(endLon);
    } else {
      payload.end_name = endName;
    }

    const res = await fetch(`${CONFIG.BACKEND_URL}/predict_route`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }
    return res.json();
  },

  // Location Search Autocomplete
  async searchLocations(query, retryCount = 0) {
    if (!query || query.length < 2) return [];
    const key = retryCount > 0 ? getNextTomTomKey() : CONFIG.TOMTOM_KEY;
    const searchString = query.toLowerCase().includes('dehradun') ? query : query + ' Dehradun';
    const url = `https://api.tomtom.com/search/2/search/${encodeURIComponent(searchString)}.json` +
      `?key=${key}&limit=6&lat=30.3226&lon=78.0411&radius=60000&countrySet=IN&language=en-GB`;

    try {
      const res = await fetch(url);
      if (res.status === 403 && retryCount < 2) {
        return this.searchLocations(query, retryCount + 1);
      }
      const data = await res.json();
      return (data.results || []).map(r => ({
        name: r.poi?.name || r.address?.freeformAddress || '',
        address: r.address?.freeformAddress || '',
        lat: r.position?.lat,
        lon: r.position?.lon,
      })).filter(r => r.name);
    } catch {
      return [];
    }
  },

  // Live Traffic Flow
  async getTrafficFlow(lat, lon, retryCount = 0) {
    const key = retryCount > 0 ? getNextTomTomKey() : CONFIG.TOMTOM_KEY;
    const url = `https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json` +
      `?key=${key}&point=${lat},${lon}&unit=KMPH`;
    try {
      const res = await fetch(url);
      if (res.status === 403 && retryCount < 2) return this.getTrafficFlow(lat, lon, retryCount + 1);
      return res.json();
    } catch {
      return null;
    }
  }
};

window.API = API;
