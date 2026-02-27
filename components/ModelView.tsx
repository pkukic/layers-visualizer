import React from 'react';
import { ModelSpec } from '../types';
import { OpTableView } from './OpTableView';
import { KVCacheSection } from './KVCacheSection';

interface Props {
  model: ModelSpec;
}

const NOTE_CLASS: Record<string, string> = {
  gemv: 'note-box note-box-gemv',
  info: 'note-box note-box-info',
  warning: 'note-box note-box-warning',
};

const NOTE_ICON: Record<string, string> = {
  gemv: '⚡',
  info: 'ℹ',
  warning: '⚠',
};

const CATEGORY_COLOR: Record<string, string> = {
  llm:    '#818cf8',
  vision: '#34d399',
  audio:  '#fb923c',
  hybrid: '#f472b6',
};

export const ModelView: React.FC<Props> = ({ model }) => {
  const accent = CATEGORY_COLOR[model.category] ?? '#818cf8';

  return (
    <div className="fade-in" style={{ minWidth: 0 }}>
      {/* ── Header ── */}
      <div
        style={{
          padding: '18px 20px 14px',
          borderBottom: '1px solid #1f2937',
          background: 'linear-gradient(to bottom, #0d1117, #0a0d14)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            <h1 style={{ fontSize: '1.4rem', fontWeight: 800, color: '#f9fafb', margin: 0, lineHeight: 1.2 }}>
              {model.name}
            </h1>
            <div style={{ fontSize: '0.72rem', color: '#6b7280', marginTop: '2px', fontFamily: 'ui-monospace, monospace' }}>
              {model.family} · {model.paramCount} · {model.category.toUpperCase()}
            </div>
          </div>
          <div style={{ display: 'flex', flex: '1', justifyContent: 'flex-end' }}>
            <div
              style={{
                fontSize: '0.68rem',
                fontWeight: 700,
                padding: '3px 10px',
                borderRadius: '12px',
                background: `${accent}18`,
                color: accent,
                border: `1px solid ${accent}35`,
                letterSpacing: '0.05em',
                alignSelf: 'flex-start',
              }}
            >
              {model.paramCount}
            </div>
          </div>
        </div>
        <p style={{ fontSize: '0.77rem', color: '#9ca3af', margin: '10px 0 10px', lineHeight: 1.55, maxWidth: '720px' }}>
          {model.subtitle}
        </p>
        {/* Technique tags */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
          {model.techniques.map((t, i) => (
            <span key={i} className="tech-tag">{t}</span>
          ))}
        </div>
      </div>

      {/* ── Hyperparameters ── */}
      <div style={{ borderBottom: '1px solid #1f2937' }}>
        <div className="section-title">
          <span>Hyperparameters</span>
        </div>
        <div className="param-grid">
          {model.hyperparameters.map((p, i) => (
            <div key={i} className="param-cell">
              <div className="param-sym">{p.symbol}</div>
              <div className="param-val">{p.value}</div>
              <div className="param-name">{p.name}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Legend ── */}
      <div style={{ borderBottom: '1px solid #1f2937' }}>
        <div className="legend">
          {[
            ['sb sb-act',    'activations'],
            ['sb sb-wt',     'weights'],
            ['sb sb-out',    'outputs'],
            ['sb sb-cache',  'kv cache'],
            ['sb sb-scalar', 'scalars / const'],
            ['sb sb-gemv',   'GEMV token (decode)'],
          ].map(([cls, label]) => (
            <div key={label} className="legend-item">
              <span className={cls} style={{ fontSize: '0.58rem', padding: '0 5px' }}>[ ]</span>
              <span>{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Notes ── */}
      {model.notes && model.notes.length > 0 && (
        <div style={{ padding: '10px 0 2px', borderBottom: '1px solid #1f2937' }}>
          {model.notes.map((n, i) => (
            <div key={i} className={NOTE_CLASS[n.type] ?? 'note-box note-box-info'}>
              <span style={{ marginRight: '6px', fontSize: '0.8rem' }}>{NOTE_ICON[n.type]}</span>
              <span dangerouslySetInnerHTML={{ __html: n.html }} />
            </div>
          ))}
        </div>
      )}

      {/* ── Sections ── */}
      {model.sections.map((section, si) => (
        <div
          key={si}
          style={{ borderBottom: si < model.sections.length - 1 ? '1px solid #1f2937' : 'none' }}
        >
          {/* Section title */}
          <div className="section-title">
            <span>{section.title}</span>
            {section.repeat !== undefined && (
              <span
                className="repeat-badge"
                dangerouslySetInnerHTML={{
                  __html: section.repeatLabel ?? `Repeats <span>${section.repeat}×</span>`,
                }}
              />
            )}
          </div>

          {/* Optional section note */}
          {section.note && (
            <div className="section-note">{section.note}</div>
          )}

          {/* Tables */}
          {section.tables.map((table, ti) => (
            <div
              key={ti}
              style={{
                borderTop: ti > 0 ? '1px dashed #1f2937' : 'none',
                marginTop: ti > 0 ? '0' : '0',
              }}
            >
              <OpTableView table={table} tableIndex={ti} />
            </div>
          ))}
        </div>
      ))}

      {/* ── KV Cache ── */}
      {model.kvCache && (
        <div style={{ borderTop: '1px solid #1f2937' }}>
          <KVCacheSection spec={model.kvCache} />
        </div>
      )}

      {/* bottom pad */}
      <div style={{ height: '40px' }} />
    </div>
  );
};
