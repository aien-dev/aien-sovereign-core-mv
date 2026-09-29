// Retire cached operator pages. Authentication always reaches the server.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    for (const name of await caches.keys()) {
      if (name.startsWith("aien-cockpit-")) await caches.delete(name);
    }
    await self.clients.claim();
  })());
});
