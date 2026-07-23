import React from 'react';
import ReactDOM from 'react-dom/client';

import './theme.css';
import './tokens.css';
import './components.css';
import './styles.css';
import './layout/titlebar.css';
import './layout/toolbar.css';
import { App } from './App';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
