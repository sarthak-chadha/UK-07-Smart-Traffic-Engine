// TerrainNav | config.js
//
// ⚠️  API KEY SETUP  ⚠️
// Replace the placeholder strings below with your real TomTom API keys.
// You can obtain free keys at https://developer.tomtom.com/
// Three separate keys are used to distribute daily quota across time-slots
// (morning / afternoon / evening ≈ 2,500 calls each per free tier).
// DO NOT commit real keys to version control.

const TOMTOM_KEYS = [
  'YOUR_TOMTOM_KEY_MORNING',    // active 06:00 – 11:59 IST
  'YOUR_TOMTOM_KEY_AFTERNOON',  // active 12:00 – 17:59 IST
  'YOUR_TOMTOM_KEY_EVENING',    // active 18:00 – 05:59 IST
];

// IST Hour Utility
function getISTHour() {
  const now = new Date();
  const utcMs = now.getTime() + now.getTimezoneOffset() * 60000;
  const istMs = utcMs + 330 * 60000;
  return new Date(istMs).getHours();
}

// Key Rotation by IST Time-of-Day
function getTomTomKey() {
  const hour = getISTHour();
  if (hour >= 6 && hour < 12) return TOMTOM_KEYS[0];
  if (hour >= 12 && hour < 18) return TOMTOM_KEYS[1];
  return TOMTOM_KEYS[2];
}

// Sequential Key Fallback
let _keyIndex = 0;
function getNextTomTomKey() {
  _keyIndex = (_keyIndex + 1) % TOMTOM_KEYS.length;
  return TOMTOM_KEYS[_keyIndex];
}

// App Config
const CONFIG = {
  BACKEND_URL: 'https://sarthak2606-uk-07-smart-traffic-engine.hf.space',
  TOMTOM_KEY: getTomTomKey(),
  MAP_CENTER: [30.3226, 78.0411],
  DEFAULT_ZOOM: 13,
  MAP_STYLES: {
    dark: `https://api.tomtom.com/maps-sdk-for-web/cdn/6.x/6.23.0/maps/maps.css`,
    STYLE_NIGHT: 'tomtom://vector/1/basic-night',
    STYLE_DAY: 'tomtom://vector/1/basic-main',
    STYLE_SAT: 'tomtom://vector/1/hybrid-main',
  }
};

window.CONFIG = CONFIG;
window.getTomTomKey = getTomTomKey;
window.getNextTomTomKey = getNextTomTomKey;
