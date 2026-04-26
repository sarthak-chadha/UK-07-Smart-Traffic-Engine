// TerrainNav | map.js

let ttMap = null;
let startMarker = null;
let endMarker = null;
let routeMarkers = [];
let currentStyle = 'night';
let trafficVisible = true;

// Graph Node Definitions
const KNOWN_NODES = [
  { name: 'Clock Tower', lat: 30.3242, lon: 78.0412, type: 'landmark' },
  { name: 'ISBT', lat: 30.2988, lon: 78.0491, type: 'transit' },
  { name: 'Rajpur Road', lat: 30.3465, lon: 78.0561, type: 'road' },
  { name: 'Sahastradhara', lat: 30.3816, lon: 78.1090, type: 'landmark' },
  { name: 'Mussoorie Library', lat: 30.4592, lon: 78.0663, type: 'landmark' },
  { name: 'Gun Hill', lat: 30.4622, lon: 78.0812, type: 'landmark' },
  { name: 'Kempty Falls', lat: 30.4817, lon: 78.0227, type: 'landmark' },
  { name: 'Jolly Grant Airport', lat: 30.1897, lon: 78.1803, type: 'transit' },
];

// Map Initialisation
function initMap() {
  ttMap = tt.map({
    key: CONFIG.TOMTOM_KEY,
    container: 'tt-map',
    center: [CONFIG.MAP_CENTER[1], CONFIG.MAP_CENTER[0]],
    zoom: CONFIG.DEFAULT_ZOOM,
    style: {
      map: 'basic_night',
      poi: 'poi_main',
      trafficFlow: 'flow_absolute',
      trafficIncidents: 'incidents_s1',
    },
    language: 'en-GB',
  });

  ttMap.addControl(new tt.FullscreenControl(), 'top-right');
  ttMap.addControl(new tt.ScaleControl({ unit: 'metric' }), 'bottom-right');

  ttMap.on('load', () => {
    setTimeout(() => {
      ttMap.resize();
      addTrafficLayers();
      addGraphNodePins();
      window.dispatchEvent(new CustomEvent('mapReady'));
    }, 100);
  });

  window.addEventListener('resize', () => { if (ttMap) ttMap.resize(); });

  return ttMap;
}

// Traffic Overlay Layers
function addTrafficLayers() {
  try {
    if (!ttMap.getSource('traffic-flow-source')) {
      ttMap.addSource('traffic-flow-source', {
        type: 'raster',
        tiles: [
          `https://api.tomtom.com/traffic/map/4/tile/flow/relative-delay/{z}/{x}/{y}.png?key=${CONFIG.TOMTOM_KEY}&tileSize=256`
        ],
        tileSize: 256,
      });
      ttMap.addLayer({
        id: 'traffic-flow-layer',
        type: 'raster',
        source: 'traffic-flow-source',
        paint: { 'raster-opacity': 0.7 },
      });
    }
    if (!ttMap.getSource('traffic-incident-source')) {
      ttMap.addSource('traffic-incident-source', {
        type: 'raster',
        tiles: [
          `https://api.tomtom.com/traffic/map/4/tile/incidents/s1/{z}/{x}/{y}.png?key=${CONFIG.TOMTOM_KEY}&tileSize=256`
        ],
        tileSize: 256,
      });
      ttMap.addLayer({
        id: 'traffic-incident-layer',
        type: 'raster',
        source: 'traffic-incident-source',
        paint: { 'raster-opacity': 0.85 },
      });
    }
  } catch (e) {
    console.warn('Traffic layer init:', e);
  }
}

// Traffic Toggle
function toggleTraffic() {
  trafficVisible = !trafficVisible;
  const vis = trafficVisible ? 'visible' : 'none';
  ['traffic-flow-layer', 'traffic-incident-layer'].forEach(id => {
    if (ttMap.getLayer(id)) ttMap.setLayoutProperty(id, 'visibility', vis);
  });
  return trafficVisible;
}

// Map Style Toggle
function toggleMapStyle() {
  const styles = ['basic_night', 'basic_main', 'hybrid_main'];
  const labels = ['night', 'day', 'satellite'];
  const idx = styles.indexOf(currentStyle === 'night' ? 'basic_night' :
    currentStyle === 'day' ? 'basic_main' : 'hybrid_main');
  const nextIdx = (idx + 1) % styles.length;
  currentStyle = labels[nextIdx];
  ttMap.setStyle({ map: styles[nextIdx] });
  return currentStyle;
}

// Graph Node Pins
function addGraphNodePins() {
  KNOWN_NODES.forEach(node => {
    const el = document.createElement('div');
    el.className = `graph-node-marker node-${node.type}`;
    el.innerHTML = `<div class="node-dot"></div><div class="node-label">${node.name}</div>`;

    new tt.Marker({ element: el, anchor: 'center' })
      .setLngLat([node.lon, node.lat])
      .addTo(ttMap);
  });
}

// Route Rendering
function drawRoute(path) {
  clearRoute();

  if (!path || path.length < 2) return;

  const coords = path.map(([lat, lon]) => [lon, lat]);

  // Glow Layer
  if (ttMap.getSource('terrain-route-glow')) {
    ttMap.removeLayer('terrain-route-glow-layer');
    ttMap.removeSource('terrain-route-glow');
  }
  ttMap.addSource('terrain-route-glow', {
    type: 'geojson',
    data: { type: 'Feature', geometry: { type: 'LineString', coordinates: coords } }
  });
  ttMap.addLayer({
    id: 'terrain-route-glow-layer',
    type: 'line',
    source: 'terrain-route-glow',
    layout: { 'line-join': 'round', 'line-cap': 'round' },
    paint: {
      'line-color': '#6EE7F7',
      'line-width': 14,
      'line-opacity': 0.2,
      'line-blur': 6,
    }
  });

  // Border Layer
  if (ttMap.getSource('terrain-route-border')) {
    ttMap.removeLayer('terrain-route-border-layer');
    ttMap.removeSource('terrain-route-border');
  }
  ttMap.addSource('terrain-route-border', {
    type: 'geojson',
    data: { type: 'Feature', geometry: { type: 'LineString', coordinates: coords } }
  });
  ttMap.addLayer({
    id: 'terrain-route-border-layer',
    type: 'line',
    source: 'terrain-route-border',
    layout: { 'line-join': 'round', 'line-cap': 'round' },
    paint: {
      'line-color': '#0d4f6e',
      'line-width': 9,
      'line-opacity': 0.9,
    }
  });

  // Main Route Line
  if (ttMap.getSource('terrain-route')) {
    ttMap.removeLayer('terrain-route-layer');
    ttMap.removeSource('terrain-route');
  }
  ttMap.addSource('terrain-route', {
    type: 'geojson',
    data: { type: 'Feature', geometry: { type: 'LineString', coordinates: coords } }
  });
  ttMap.addLayer({
    id: 'terrain-route-layer',
    type: 'line',
    source: 'terrain-route',
    layout: { 'line-join': 'round', 'line-cap': 'round' },
    paint: {
      'line-color': '#6EE7F7',
      'line-width': 5,
      'line-opacity': 1,
      'line-dasharray': [1, 0],
    }
  });

  // Animated Dash Layer
  if (ttMap.getSource('terrain-route-dash')) {
    ttMap.removeLayer('terrain-route-dash-layer');
    ttMap.removeSource('terrain-route-dash');
  }
  ttMap.addSource('terrain-route-dash', {
    type: 'geojson',
    data: { type: 'Feature', geometry: { type: 'LineString', coordinates: coords } }
  });
  ttMap.addLayer({
    id: 'terrain-route-dash-layer',
    type: 'line',
    source: 'terrain-route-dash',
    layout: { 'line-join': 'round', 'line-cap': 'round' },
    paint: {
      'line-color': '#ffffff',
      'line-width': 2,
      'line-opacity': 0.5,
      'line-dasharray': [0, 4, 3],
    }
  });

  animateDashLine();

  placeStartMarker(path[0][0], path[0][1]);
  placeEndMarker(path[path.length - 1][0], path[path.length - 1][1]);

  const bounds = coords.reduce(
    (b, c) => b.extend(c),
    new tt.LngLatBounds(coords[0], coords[0])
  );
  ttMap.fitBounds(bounds, { padding: { top: 140, bottom: 320, left: 60, right: 80 }, duration: 1200 });
}

// Dash Animation
let dashOffset = 0;
let dashAnimId = null;
function animateDashLine() {
  if (dashAnimId) cancelAnimationFrame(dashAnimId);
  function step() {
    dashOffset = (dashOffset - 1) % 7;
    if (ttMap.getLayer('terrain-route-dash-layer')) {
      ttMap.setPaintProperty('terrain-route-dash-layer', 'line-dasharray', [dashOffset, 4, 3]);
    }
    dashAnimId = requestAnimationFrame(step);
  }
  dashAnimId = requestAnimationFrame(step);
}

// Route Clear
function clearRoute() {
  ['terrain-route-glow', 'terrain-route-border', 'terrain-route', 'terrain-route-dash'].forEach(id => {
    if (ttMap.getLayer(id + '-layer')) ttMap.removeLayer(id + '-layer');
    if (ttMap.getSource(id)) ttMap.removeSource(id);
  });
  if (startMarker) { startMarker.remove(); startMarker = null; }
  if (endMarker) { endMarker.remove(); endMarker = null; }
  routeMarkers.forEach(m => m.remove());
  routeMarkers = [];
  if (dashAnimId) { cancelAnimationFrame(dashAnimId); dashAnimId = null; }
}

// Start Marker
function placeStartMarker(lat, lon) {
  const el = document.createElement('div');
  el.className = 'marker-wrap start-marker-wrap';
  el.innerHTML = `
    <div class="marker-blob start-blob">
      <div class="marker-pulse"></div>
      <div class="marker-dot"></div>
    </div>
    <div class="marker-label-tag">Start</div>
  `;
  startMarker = new tt.Marker({ element: el, anchor: 'bottom' })
    .setLngLat([lon, lat])
    .addTo(ttMap);
}

// End Marker
function placeEndMarker(lat, lon) {
  const el = document.createElement('div');
  el.className = 'marker-wrap end-marker-wrap';
  el.innerHTML = `
    <div class="marker-blob end-blob">
      <i class="ph ph-map-pin-simple-area"></i>
    </div>
    <div class="marker-label-tag end-tag">Destination</div>
  `;
  endMarker = new tt.Marker({ element: el, anchor: 'bottom' })
    .setLngLat([lon, lat])
    .addTo(ttMap);
}

// Camera Fly-To
function flyToLocation(lat, lon, zoom = 15) {
  ttMap.flyTo({ center: [lon, lat], zoom, speed: 1.4, curve: 1.4 });
}

window.MapEngine = {
  initMap,
  drawRoute,
  clearRoute,
  toggleTraffic,
  toggleMapStyle,
  flyToLocation,
  getMap: () => ttMap,
};
