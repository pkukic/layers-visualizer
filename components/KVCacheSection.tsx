import React from 'react';
import { KVCacheSpec } from '../types';

interface Props {
  spec: KVCacheSpec;
}

export const KVCacheSection: React.FC<Props> = ({ spec }) => {
  const colCount = spec.columns.length;
  const gridCols = colCount === 2 ? '1fr 1fr' : colCount === 3 ? '1fr 1fr 1fr' : '1fr';

  return (
    <div>
      <div
        className="section-title"
        style={{ color: '#a78bfa', borderTop: '1px solid #1f2937', marginTop: '0' }}
      >
        <span style={{ color: '#c4b5fd', textTransform: 'none', fontSize: '0.78rem' }}>
          ⬡ {spec.title}
        </span>
      </div>
      <div
        className="kv-grid"
        style={{ gridTemplateColumns: gridCols }}
      >
        {spec.columns.map((col, i) => (
          <div key={i} className="kv-col">
            <div className="kv-col-title">{col.title}</div>
            {col.lines.map((line, j) => (
              <div key={j} className="kv-row">
                <span className="kv-label">{line.label}</span>
                <span className="kv-value">{line.value}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};
