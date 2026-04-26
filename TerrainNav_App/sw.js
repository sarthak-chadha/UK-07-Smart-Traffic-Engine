const CACHE_NAME = 'terrainnav-v2';
const urlsToCache = [
  './',
  './index.html',
  './styles.css',
  './config.js',
  './api.js',
  './map.js',
  './app.js',
  './manifest.json',
  './icon.svg',
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(urlsToCache))
  );
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  if (event.request.url.includes('tomtom.com') ||
      event.request.url.includes('googleapis.com') ||
      event.request.url.includes('hf.space')) {
    return;
  }
  event.respondWith(
    caches.match(event.request).then(response => response || fetch(event.request))
  );
});
