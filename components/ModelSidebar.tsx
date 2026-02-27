import React from 'react';
import { ModelSpec, ModelCategory } from '../types';
import { CATEGORY_LABELS, CATEGORY_ORDER } from '../data';

interface Props {
  models: ModelSpec[];
  selectedId: string;
  onSelect: (id: string) => void;
}

const CATEGORY_ACCENT: Record<ModelCategory, string> = {
  llm:    '#818cf8',
  vision: '#34d399',
  audio:  '#fb923c',
  hybrid: '#f472b6',
};

export const ModelSidebar: React.FC<Props> = ({ models, selectedId, onSelect }) => {
  const byCategory: Partial<Record<ModelCategory, ModelSpec[]>> = {};
  for (const m of models) {
    if (!byCategory[m.category]) byCategory[m.category] = [];
    byCategory[m.category]!.push(m);
  }

  return (
    <div className="model-sidebar">
      {/* App title */}
      <div
        style={{
          padding: '16px 14px 12px',
          borderBottom: '1px solid #1f2937',
        }}
      >
        <div style={{ fontSize: '1rem', fontWeight: 800, color: '#f9fafb', letterSpacing: '-0.01em' }}>
          Layers Visualizer
        </div>
        <div style={{ fontSize: '0.65rem', color: '#4b5563', marginTop: '2px' }}>
          Transformer architecture explorer
        </div>
      </div>

      {/* Model list grouped by category */}
      {CATEGORY_ORDER.map((cat) => {
        const catModels = byCategory[cat];
        if (!catModels || catModels.length === 0) return null;
        const accent = CATEGORY_ACCENT[cat];
        return (
          <div key={cat}>
            <div
              className="model-category-label"
              style={{ color: `${accent}80` }}
            >
              {CATEGORY_LABELS[cat]}
            </div>
            {catModels.map((m) => (
              <button
                key={m.id}
                className={`model-btn${selectedId === m.id ? ' active' : ''}`}
                onClick={() => onSelect(m.id)}
                style={selectedId === m.id ? { borderLeftColor: accent } : {}}
              >
                <div
                  className="model-btn-name"
                  style={selectedId === m.id ? { color: accent } : {}}
                >
                  {m.name}
                </div>
                <div className="model-btn-sub">{m.paramCount}</div>
              </button>
            ))}
          </div>
        );
      })}

      {/* Bottom info */}
      <div
        style={{
          padding: '16px 14px',
          marginTop: 'auto',
          borderTop: '1px solid #1f2937',
          fontSize: '0.62rem',
          color: '#374151',
          lineHeight: 1.6,
        }}
      >
        10 architectures · LLM / Vision / Audio / Hybrid
      </div>
    </div>
  );
};
