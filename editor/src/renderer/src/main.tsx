import React from 'react';
import ReactDOM from 'react-dom/client';

import './theme.css';
import './tokens.css';
import './typography.css';
import './components.css';
import './radio.css';
import './styles.css';
import './layout/titlebar.css';
import './layout/toolbar.css';
import './ui-polish.css';
import './viewport/viewportChromeOverrides.css';
import './viewport/viewportChromeBehavior';
import { App } from './App';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
