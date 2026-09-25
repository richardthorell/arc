export type SearchDrawerMode = 'assets' | 'commands';

const searchDrawerRequestEvent = 'arc:search-drawer-request';

type SearchDrawerRequest = {
  mode: SearchDrawerMode;
};

export const requestSearchDrawer = (mode: SearchDrawerMode) => {
  window.dispatchEvent(new CustomEvent<SearchDrawerRequest>(searchDrawerRequestEvent, { detail: { mode } }));
};

export const subscribeSearchDrawerRequests = (listener: (mode: SearchDrawerMode) => void) => {
  const onRequest = (event: Event) => listener((event as CustomEvent<SearchDrawerRequest>).detail.mode);
  window.addEventListener(searchDrawerRequestEvent, onRequest);
  return () => window.removeEventListener(searchDrawerRequestEvent, onRequest);
};
