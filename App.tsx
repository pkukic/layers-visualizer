import React, { useState, useCallback } from 'react';
import { ModelSidebar } from './components/ModelSidebar';
import { ModelView } from './components/ModelView';
import { ALL_MODELS } from './data';

export default function App() {
  const [selectedId, setSelectedId] = useState<string>(ALL_MODELS[0].id);

  const selectedModel = ALL_MODELS.find((m) => m.id === selectedId) ?? ALL_MODELS[0];

  const handleSelect = useCallback((id: string) => {
    setSelectedId(id);
    // Scroll main content back to top when switching models
    const main = document.getElementById('main-content');
    if (main) main.scrollTop = 0;
  }, []);

  return (
    <div
      style={{
        display: 'flex',
        height: '100vh',
        overflow: 'hidden',
        background: '#0d1117',
        color: '#e5e7eb',
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif',
      }}
    >
      {/* Sidebar */}
      <ModelSidebar
        models={ALL_MODELS}
        selectedId={selectedId}
        onSelect={handleSelect}
      />

      {/* Main content */}
      <div
        id="main-content"
        style={{
          flex: 1,
          overflowY: 'auto',
          overflowX: 'auto',
          minWidth: 0,
          background: '#0d1117',
        }}
      >
        <ModelView key={selectedId} model={selectedModel} />
      </div>
    </div>
  );
}
