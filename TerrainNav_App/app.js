// TerrainNav | app.js

document.addEventListener('DOMContentLoaded', () => {

  // Map Init
  MapEngine.initMap();

  // DOM References
  const splash        = document.getElementById('splash-screen');
  const app           = document.getElementById('app');
  const loaderFill    = document.getElementById('loader-fill');
  const loaderText    = document.getElementById('loader-text');
  const panel         = document.getElementById('bottom-panel');
  const panelHandle   = document.getElementById('panel-handle');

  const startInput    = document.getElementById('start-input');
  const endInput      = document.getElementById('end-input');
  const startSugg     = document.getElementById('start-suggestions');
  const endSugg       = document.getElementById('end-suggestions');
  const clearStart    = document.getElementById('clear-start');
  const clearEnd      = document.getElementById('clear-end');
  const swapBtn       = document.getElementById('swap-btn');
  const coordToggle   = document.getElementById('coord-toggle');
  const coordSection  = document.getElementById('coord-section');
  const predictBtn    = document.getElementById('predict-btn');
  const predictLoader = document.getElementById('predict-loader');
  const routeResult   = document.getElementById('route-result');
  const errorState    = document.getElementById('error-state');
  const resultClose   = document.getElementById('result-close');
  const retryBtn      = document.getElementById('retry-btn');
  const startNavBtn   = document.getElementById('start-nav-btn');

  const resultEta     = document.getElementById('result-eta');
  const resultNodes   = document.getElementById('result-nodes');
  const resultTime    = document.getElementById('result-time');
  const errorMsg      = document.getElementById('error-msg');

  const zoomIn        = document.getElementById('zoom-in-btn');
  const zoomOut       = document.getElementById('zoom-out-btn');
  const myLoc         = document.getElementById('my-location-btn');
  const navToggle     = document.getElementById('navigate-btn');
  const mapStyleBtn   = document.getElementById('map-style-btn');
  const trafficBtn    = document.getElementById('traffic-toggle-btn');
  const catItems      = document.querySelectorAll('.cat-item');

  // State
  let startCoords  = null;
  let endCoords    = null;
  let useCoords    = false;
  let lastPath     = null;
  let panelExpanded = false;
  let panelLocked   = false;

  // Splash Loader
  const loadMessages = [
    'Initialising engine…',
    'Loading terrain graph…',
    'Calibrating GATv2 model…',
    'Almost ready…',
  ];
  let loadStep = 0;
  const loadInterval = setInterval(() => {
    loadStep++;
    const pct = Math.min(loadStep * 28, 95);
    loaderFill.style.width = pct + '%';
    if (loadMessages[loadStep]) loaderText.textContent = loadMessages[loadStep];
  }, 600);

  window.addEventListener('mapReady', () => {
    clearInterval(loadInterval);
    loaderFill.style.width = '100%';
    loaderText.textContent = 'Ready!';
    setTimeout(() => {
      splash.classList.add('hidden');
      app.classList.remove('hidden');
      fetchLiveTraffic();
    }, 500);
  });

  setTimeout(() => {
    if (!app.classList.contains('visible')) {
      clearInterval(loadInterval);
      loaderFill.style.width = '100%';
      splash.classList.add('hidden');
      app.classList.remove('hidden');
    }
  }, 5000);

  // Bottom Panel Logic
  let dragStartY = 0;
  let isDragging = false;

  function expandPanel() {
    panel.style.transform = 'translateY(0)';
    panelExpanded = true;
  }

  function collapsePanel() {
    if (panelLocked) return;
    const peekPx = window.innerHeight * 0.3;
    panel.style.transform = `translateY(calc(100% - ${peekPx}px))`;
    panelExpanded = false;
  }

  function setInitialPanelPos() {
    const peekPx = window.innerHeight * 0.3;
    panel.style.transform = `translateY(calc(100% - ${peekPx}px))`;
    panel.style.transition = 'transform 0.35s cubic-bezier(0.2, 0.8, 0.2, 1)';
  }
  setInitialPanelPos();

  panelHandle.addEventListener('click', () => {
    if (panelExpanded) collapsePanel();
    else expandPanel();
  });

  panelHandle.addEventListener('mousedown', (e) => {
    dragStartY = e.clientY;
    isDragging = true;
    panel.style.transition = 'none';
    document.addEventListener('mousemove', onDragMove);
    document.addEventListener('mouseup', onDragEnd);
  });

  panelHandle.addEventListener('touchstart', (e) => {
    dragStartY = e.touches[0].clientY;
    isDragging = true;
    panel.style.transition = 'none';
  }, { passive: true });

  panel.addEventListener('touchmove', (e) => {
    if (!isDragging) return;
    const dy = e.touches[0].clientY - dragStartY;
    const rect = panel.getBoundingClientRect();
    const newTop = Math.max(60, rect.top + dy);
    panel.style.transform = `translateY(${newTop - (window.innerHeight - panel.offsetHeight)}px)`;
    dragStartY = e.touches[0].clientY;
  }, { passive: true });

  panel.addEventListener('touchend', () => {
    isDragging = false;
    panel.style.transition = 'transform 0.35s cubic-bezier(0.2, 0.8, 0.2, 1)';
    const rect = panel.getBoundingClientRect();
    if (rect.top < window.innerHeight * 0.45) expandPanel();
    else collapsePanel();
  });

  function onDragMove(e) {
    if (!isDragging) return;
    const dy = e.clientY - dragStartY;
    const rect = panel.getBoundingClientRect();
    const newTop = Math.max(60, rect.top + dy);
    panel.style.transform = `translateY(${newTop - (window.innerHeight - panel.offsetHeight)}px)`;
    dragStartY = e.clientY;
  }

  function onDragEnd() {
    isDragging = false;
    panel.style.transition = 'transform 0.35s cubic-bezier(0.2, 0.8, 0.2, 1)';
    document.removeEventListener('mousemove', onDragMove);
    document.removeEventListener('mouseup', onDragEnd);
    const rect = panel.getBoundingClientRect();
    if (rect.top < window.innerHeight * 0.45) expandPanel();
    else collapsePanel();
  }

  // Autocomplete
  let acTimer = null;
  function setupAutocomplete(input, suggBox, onSelect) {
    input.addEventListener('input', () => {
      clearTimeout(acTimer);
      const q = input.value.trim();
      if (q.length < 2) { suggBox.innerHTML = ''; suggBox.classList.remove('open'); return; }
      acTimer = setTimeout(async () => {
        const results = await API.searchLocations(q);
        if (!results.length) { suggBox.innerHTML = ''; suggBox.classList.remove('open'); return; }
        suggBox.innerHTML = results.map((r, i) =>
          `<div class="sugg-item" data-idx="${i}" tabindex="0">
            <i class="ph ph-map-pin"></i>
            <div>
              <div class="sugg-name">${r.name}</div>
              <div class="sugg-addr">${r.address}</div>
            </div>
          </div>`
        ).join('');
        suggBox.classList.add('open');
        suggBox.__results = results;

        suggBox.querySelectorAll('.sugg-item').forEach(el => {
          el.addEventListener('click', () => {
            const r = results[parseInt(el.dataset.idx)];
            onSelect(r);
            suggBox.innerHTML = '';
            suggBox.classList.remove('open');
          });
        });
      }, 280);
    });

    input.addEventListener('blur', () => {
      setTimeout(() => { suggBox.innerHTML = ''; suggBox.classList.remove('open'); }, 200);
    });
  }

  setupAutocomplete(startInput, startSugg, (r) => {
    startInput.value = r.name;
    startCoords = { lat: r.lat, lon: r.lon };
    MapEngine.flyToLocation(r.lat, r.lon, 14);
  });

  setupAutocomplete(endInput, endSugg, (r) => {
    endInput.value = r.name;
    endCoords = { lat: r.lat, lon: r.lon };
  });

  // Input Controls
  clearStart.addEventListener('click', () => { startInput.value = ''; startCoords = null; });
  clearEnd.addEventListener('click', () => { endInput.value = ''; endCoords = null; });

  swapBtn.addEventListener('click', () => {
    [startInput.value, endInput.value] = [endInput.value, startInput.value];
    [startCoords, endCoords] = [endCoords, startCoords];
    swapBtn.classList.add('rotating');
    setTimeout(() => swapBtn.classList.remove('rotating'), 400);
  });

  coordToggle.addEventListener('click', () => {
    useCoords = !useCoords;
    coordSection.classList.toggle('hidden', !useCoords);
    coordToggle.innerHTML = useCoords
      ? '<i class="ph ph-text-aa"></i> Use Location Name Instead'
      : '<i class="ph ph-map-trifold"></i> Use Coordinates Instead';
  });

  // Route Prediction
  predictBtn.addEventListener('click', doPredict);

  async function doPredict() {
    let payload = {};
    if (useCoords) {
      const sl = document.getElementById('start-lat').value;
      const sn = document.getElementById('start-lon').value;
      const el = document.getElementById('end-lat').value;
      const en = document.getElementById('end-lon').value;
      if (!sl || !sn || !el || !en) {
        showError('Please enter all four coordinates.');
        return;
      }
      payload = { startLat: sl, startLon: sn, endLat: el, endLon: en };
    } else {
      if (!startInput.value.trim()) { showError('Please enter a start location.'); return; }
      if (!endInput.value.trim()) { showError('Please enter a destination.'); return; }

      if (startCoords) {
        payload.startLat = startCoords.lat;
        payload.startLon = startCoords.lon;
      } else {
        payload.startName = startInput.value.trim();
      }
      if (endCoords) {
        payload.endLat = endCoords.lat;
        payload.endLon = endCoords.lon;
      } else {
        payload.endName = endInput.value.trim();
      }
    }

    setPanel('loading');
    expandPanel();

    try {
      const data = await API.predictRoute(payload);
      if (data.status !== 'success') throw new Error(data.detail || 'Unknown error');
      lastPath = data.path;
      MapEngine.drawRoute(data.path);
      showResult(data);
      panelLocked = true;
    } catch (err) {
      showError(err.message || 'Route prediction failed.');
    }
  }

  // Panel State Manager
  function setPanel(state) {
    document.getElementById('route-form-section').classList.toggle('hidden', state !== 'form');
    predictLoader.classList.toggle('hidden', state !== 'loading');
    routeResult.classList.toggle('hidden', state !== 'result');
    errorState.classList.toggle('hidden', state !== 'error');
    document.getElementById('quick-cats').classList.toggle('hidden', state !== 'form');
    document.getElementById('live-info-section').classList.toggle('hidden', state !== 'form');
  }

  // Result Display
  function showResult(data) {
    resultEta.textContent = data.eta_mins ? data.eta_mins.toFixed(1) : '--';
    resultNodes.textContent = data.path ? data.path.length : '--';
    resultTime.textContent = data.prediction_time_ist
      ? data.prediction_time_ist.replace('_', ' at ').replace(/-/g, ':')
      : 'Just now';
    drawElevationChart(data.path);
    setPanel('result');
  }

  function showError(msg) {
    errorMsg.textContent = msg;
    setPanel('error');
  }

  resultClose.addEventListener('click', () => {
    setPanel('form');
    panelLocked = false;
    MapEngine.clearRoute();
    collapsePanel();
  });

  retryBtn.addEventListener('click', () => {
    setPanel('form');
    panelLocked = false;
    expandPanel();
  });

  startNavBtn.addEventListener('click', () => {
    if (lastPath && lastPath.length > 0) {
      const p = lastPath[lastPath.length - 1];
      MapEngine.flyToLocation(p[0], p[1], 16);
    }
  });

  // Elevation Chart
  function drawElevationChart(path) {
    const canvas = document.getElementById('elevation-chart');
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    if (!path || path.length < 2) return;

    const lats = path.map(p => p[0]);
    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const range = maxLat - minLat || 0.001;

    const elevs = lats.map(lat => 450 + ((lat - minLat) / range) * 1650);
    const minE = Math.min(...elevs);
    const maxE = Math.max(...elevs);
    const eRange = maxE - minE || 1;

    const pts = elevs.map((e, i) => ({
      x: (i / (elevs.length - 1)) * W,
      y: H - 10 - ((e - minE) / eRange) * (H - 20)
    }));

    // Fill
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, 'rgba(110, 231, 247, 0.8)');
    grad.addColorStop(0.5, 'rgba(139, 92, 246, 0.6)');
    grad.addColorStop(1, 'rgba(252, 211, 77, 0.8)');

    ctx.beginPath();
    ctx.moveTo(pts[0].x, H);
    pts.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.lineTo(pts[pts.length - 1].x, H);
    ctx.closePath();
    const fillGrad = ctx.createLinearGradient(0, 0, 0, H);
    fillGrad.addColorStop(0, 'rgba(110, 231, 247, 0.35)');
    fillGrad.addColorStop(1, 'rgba(110, 231, 247, 0)');
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Stroke
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    pts.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Labels
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillText(`${Math.round(minE)}m`, 4, H - 4);
    ctx.fillText(`${Math.round(maxE)}m`, 4, 14);
  }

  // Map Controls
  zoomIn.addEventListener('click', () => MapEngine.getMap()?.zoomIn());
  zoomOut.addEventListener('click', () => MapEngine.getMap()?.zoomOut());

  myLoc.addEventListener('click', () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(pos => {
        MapEngine.flyToLocation(pos.coords.latitude, pos.coords.longitude, 15);
      }, () => MapEngine.flyToLocation(CONFIG.MAP_CENTER[0], CONFIG.MAP_CENTER[1], 13));
    }
  });

  navToggle.addEventListener('click', () => {
    expandPanel();
    startInput.focus();
  });

  // Set initial colors for dark mode
  mapStyleBtn.style.background = '#0a0a0a';
  mapStyleBtn.style.color = '#ffffff';
  mapStyleBtn.style.border = '1px solid #333';

  mapStyleBtn.addEventListener('click', () => {
    const style = MapEngine.toggleMapStyle();
    if (style === 'night') {
      mapStyleBtn.style.background = '#0a0a0a';
      mapStyleBtn.style.color = '#ffffff';
      mapStyleBtn.style.border = '1px solid #333';
    } else if (style === 'day') {
      mapStyleBtn.style.background = '#ffffff';
      mapStyleBtn.style.color = '#000000';
      mapStyleBtn.style.border = '1px solid #ddd';
    } else {
      mapStyleBtn.style.background = '#0d4f6e';
      mapStyleBtn.style.color = '#ffffff';
      mapStyleBtn.style.border = '1px solid #6EE7F7';
    }
  });

  trafficBtn.addEventListener('click', () => {
    const visible = MapEngine.toggleTraffic();
    trafficBtn.classList.toggle('active', !visible);
    trafficBtn.title = visible ? 'Hide Traffic' : 'Show Traffic';
  });

  // Quick Place Categories
  catItems.forEach(btn => {
    btn.addEventListener('click', () => {
      const dest = btn.dataset.dest;
      endInput.value = dest;
      
      const lat = btn.dataset.lat;
      const lon = btn.dataset.lon;
      
      if (lat && lon) {
        endCoords = { lat: parseFloat(lat), lon: parseFloat(lon) };
      } else {
        endCoords = null;
      }
      
      expandPanel();
      endInput.focus();
    });
  });

  // Live Traffic Status
  async function fetchLiveTraffic() {
    const el = document.getElementById('traffic-status');
    const badge = document.getElementById('traffic-badge');
    try {
      const data = await API.getTrafficFlow(30.3226, 78.0411);
      if (data && data.flowSegmentData) {
        const fsd = data.flowSegmentData;
        const current = fsd.currentSpeed;
        const free = fsd.freeFlowSpeed;
        const ratio = current / free;
        let label, color;
        if (ratio > 0.8) { label = 'Free Flow'; color = '#1DD1A1'; }
        else if (ratio > 0.5) { label = 'Moderate'; color = '#FFC312'; }
        else { label = 'Heavy'; color = '#FF5252'; }
        el.textContent = `${current} km/h (free: ${free} km/h)`;
        badge.textContent = label;
        badge.style.background = color + '22';
        badge.style.color = color;
        badge.style.borderColor = color + '55';
      }
    } catch {
      el.textContent = 'Unable to fetch traffic data';
    }
  }

  // Initial State
  setPanel('form');
});
