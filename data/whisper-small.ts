import { ModelSpec } from '../types';

const s = (shape: string, badge: 'act'|'wt'|'out'|'cache'|'scalar'|'gemv') =>
  ({ kind: 'shape' as const, shape, badge });
const op = (symbol: string) => ({ kind: 'op' as const, symbol });
const note = (text: string) => ({ kind: 'note' as const, text });

const A = op('→'), X = op('×'), P = op('+');

// Whisper Small: 244M params
// Encoder: 12 layers, H=768, n_h=12, d_h=64, F=3072
// Decoder: 12 layers, H=768, n_h=12, d_h=64, F=3072
// Mel features: 80 channels, T_audio frames = input_length/160
// After subsampling: T_enc = T_audio/2
// Encoder output: [1, T_enc, 768]
// Decoder: text tokens → 51865-class logits

export const whisperSmall: ModelSpec = {
  id: 'whisper-small',
  name: 'Whisper Small',
  family: 'Whisper',
  category: 'audio',
  paramCount: '244M',
  subtitle: 'Encoder-decoder ASR transformer. Conv1D frontend on mel spectrograms (80 mel bins), sinusoidal position embeddings, full attention in encoder, causal self-attn + cross-attn in decoder.',
  techniques: ['Conv Frontend', 'Sinusoidal Pos Embedding', 'Encoder-Decoder', 'Cross-Attention', 'MHA', 'LayerNorm', 'GELU'],
  hyperparameters: [
    { symbol: 'H', value: '768', name: 'hidden size' },
    { symbol: 'n_h', value: '12', name: 'attention heads' },
    { symbol: 'd_h', value: '64', name: 'head dim = H/n_h' },
    { symbol: 'F', value: '3072', name: 'FFN intermediate (4×H)' },
    { symbol: 'L_e', value: '12', name: 'encoder layers' },
    { symbol: 'L_d', value: '12', name: 'decoder layers' },
    { symbol: 'V', value: '51865', name: 'vocab size (multilingual)' },
    { symbol: 'M', value: '80', name: 'mel filter banks' },
    { symbol: 'T_e', value: '1500', name: 'encoder seq len (30s audio)' },
    { symbol: 'T_d', value: 'T_d', name: 'decoder generated length' },
  ],
  notes: [
    {
      type: 'info',
      html: '<strong>Conv frontend:</strong> Two Conv1D layers (with padding/stride) compress 80-dim mel spectrograms to 768-dim features at 2× lower frame rate: Conv1(80→768, k=3) → GELU → Conv1(768→768, k=3, s=2) → GELU. Sinusoidal positional embeddings are then added.',
    },
    {
      type: 'info',
      html: '<strong>Encoder-Decoder architecture:</strong> The encoder processes the full audio sequence (full attention, no causal mask) and produces 1500 context vectors. The decoder generates text tokens autoregressively using causal self-attention and cross-attention over the encoder output. The left column shows the encoder forward pass; the right column shows each decoder step.',
    },
  ],
  sections: [
    {
      title: '0 · Encoder — Conv1D Frontend',
      tables: [{
        col1Header: '▶ Encoder (full audio)',
        col1Class: 'phase-prefill',
        col2Header: '▶ Shapes explained',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Mel spectrogram', sub: '80 mel bins × T frames',
            col1: [s('[1, 80, T_audio]', 'act'), A, note('e.g. T=3000 for 30s')],
            col2: [note('hop=10ms → 100 frames/sec')],
          },
          {
            name: 'Conv1D block 1', sub: 'Conv(80→768, k=3, pad=1) + GELU',
            col1: [s('[1, 80, T_audio]', 'act'), X, s('[768, 80, 3]', 'wt'), A, s('[1, 768, T_audio]', 'out')],
            col2: [note('same-length output')],
          },
          {
            name: 'Conv1D block 2', sub: 'Conv(768→768, k=3, s=2, pad=1) + GELU',
            col1: [s('[1, 768, T_audio]', 'act'), X, s('[768, 768, 3]', 'wt'), A, s('[1, 768, T_audio/2]', 'out')],
            col2: [note('stride 2 halves length → T_enc=1500')],
          },
          {
            name: 'Transpose + pos embed', sub: '[C, T] → [T, C] + sinusoidal',
            col1: [s('[1, 768, 1500]', 'act'), A, s('[1, 1500, 768]', 'out'), P, s('[1500, 768]', 'scalar'), A, s('[1, 1500, 768]', 'out')],
            col2: [note('fixed sinusoidal, not learned')],
          },
        ],
      }],
    },
    {
      title: 'Encoder Block',
      repeat: 12,
      repeatLabel: 'Repeats <span>12 times</span>. Bidirectional (no causal mask). K/V from encoder only — no cross-attn.',
      tables: [
        {
          col1Header: '▶ Encoder',
          col1Class: 'phase-prefill',
          col2Header: '▶ Shapes',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm', sub: 'encoder_attn_layer_norm',
              col1: [s('[1, 1500, 768]', 'act'), X, s('[768]', 'wt'), A, s('[1, 1500, 768]', 'out')],
              col2: [],
            },
            {
              name: 'Q/K/V proj (MHA)', sub: '[768→768] each, 12 heads',
              col1: [s('[1, 1500, 768]', 'act'), X, s('[768, 768]', 'wt'), A, s('[1, 1500, 768]', 'out'), X, s('(×3)', 'scalar')],
              col2: [s('[1, 12, 1500, 64]', 'out'), X, s('(×3)', 'scalar')],
            },
            {
              name: 'Full self-attn + proj', sub: 'no causal mask, [768→768]',
              col1: [s('[1, 12, 1500, 1500]', 'act'), X, s('[1, 12, 1500, 64]', 'act'), A, s('[1, 1500, 768]', 'out')],
              col2: [note('1500² attn matrix')],
            },
            {
              name: 'Residual + LN + FFN', sub: 'LayerNorm → FC(768→3072) → GELU → FC(3072→768) → residual',
              col1: [s('[1, 1500, 768]', 'act'), X, s('[3072, 768]', 'wt'), A, s('[1, 1500, 3072]', 'out')],
              col2: [s('[1, 1500, 768]', 'out')],
            },
          ],
        },
      ],
    },
    {
      title: 'Encoder Output',
      tables: [{
        col1Header: '▶ Encoder final',
        col1Class: 'phase-prefill',
        col2Header: '▶ Cached for decoder',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Final LayerNorm', sub: 'encoder output norm',
            col1: [s('[1, 1500, 768]', 'act'), X, s('[768]', 'wt'), A, s('[1, 1500, 768]', 'out')],
            col2: [],
          },
          {
            name: 'Encoder K/V proj', sub: 'pre-computed for all decoder layers',
            col1: [s('[1, 1500, 768]', 'act'), X, s('[768, 768]', 'wt'), A, s('[1, 1500, 768]', 'cache'), X, s('(×2 per layer)', 'scalar')],
            col2: [note('K,V cached × 12 decoder layers')],
          },
        ],
      }],
    },
    {
      title: 'Decoder Step',
      repeat: 12,
      repeatLabel: 'Decoder: <span>12 layers</span>. Each step: causal self-attn over generated tokens, then cross-attn over encoder output.',
      tables: [
        {
          col1Header: '▶ Input token',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode step t',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'Token embed', sub: 'decoder embed_tokens',
              col1: [s('[1, T_d]', 'act'), X, s('[51865, 768]', 'wt'), A, s('[1, T_d, 768]', 'out')],
              col2: [s('[1, 1]', 'act'), X, s('[51865, 768]', 'wt'), A, s('[1, 1, 768]', 'out'), note('GEMV')],
            },
            {
              name: 'Pos embed', sub: 'learned [448, 768]',
              col1: [s('[1, T_d, 768]', 'act'), P, s('[T_d, 768]', 'scalar'), A, s('[1, T_d, 768]', 'out')],
              col2: [s('[1, 1, 768]', 'act'), P, s('[t, 768]', 'scalar'), A, s('[1, 1, 768]', 'out')],
            },
          ],
        },
        {
          col1Header: '▶ Decoder self-attn',
          col1Class: 'phase-prefill',
          col2Header: '▶ Step t (causal)',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm', sub: 'self_attn_layer_norm',
              col1: [s('[1, T_d, 768]', 'act'), X, s('[768]', 'wt'), A, s('[1, T_d, 768]', 'out')],
              col2: [s('[1, 1, 768]', 'act'), X, s('[768]', 'wt'), A, s('[1, 1, 768]', 'out')],
            },
            {
              name: 'Causal self-attn Q/K/V', sub: 'masked self-attn over generated tokens',
              col1: [s('[1, T_d, 768]', 'act'), X, s('[768, 768]', 'wt'), A, s('[1, T_d, 768]', 'out'), X, s('(×3)', 'scalar')],
              col2: [s('[1, 1, 768]', 'gemv'), X, s('[768, 768]', 'wt'), A, s('[1, 1, 768]', 'out'), note('GEMV')],
            },
            {
              name: 'Decoder KV cache', sub: 'self-attn cache grows',
              col1: [s('[1, 12, 0, 64]', 'cache'), P, s('[1, 12, T_d, 64]', 'act'), A, s('[1, 12, T_d, 64]', 'cache')],
              col2: [s('[1, 12, t, 64]', 'cache'), P, s('[1, 12, 1, 64]', 'act'), A, s('[1, 12, t+1, 64]', 'cache')],
            },
            {
              name: 'Causal attn + proj', sub: 'O proj [768→768]',
              col1: [s('[1, 12, T_d, T_d]', 'act'), X, s('[1, 12, T_d, 64]', 'cache'), A, s('[1, T_d, 768]', 'out')],
              col2: [s('[1, 12, 1, t]', 'gemv'), X, s('[1, 12, t, 64]', 'cache'), A, s('[1, 1, 768]', 'out')],
            },
          ],
        },
        {
          col1Header: '▶ Decoder cross-attn',
          col1Class: 'phase-prefill',
          col2Header: '▶ Step t (fixed encoder)',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm', sub: 'encoder_attn_layer_norm',
              col1: [s('[1, T_d, 768]', 'act'), X, s('[768]', 'wt'), A, s('[1, T_d, 768]', 'out')],
              col2: [s('[1, 1, 768]', 'act'), X, s('[768]', 'wt'), A, s('[1, 1, 768]', 'out')],
            },
            {
              name: 'Cross-attn Q proj', sub: 'Q from decoder [768→768]',
              col1: [s('[1, T_d, 768]', 'act'), X, s('[768, 768]', 'wt'), A, s('[1, T_d, 768]', 'out')],
              col2: [s('[1, 1, 768]', 'gemv'), X, s('[768, 768]', 'wt'), A, s('[1, 1, 768]', 'out'), note('GEMV')],
            },
            {
              name: 'Cross-attn K/V', sub: 'K,V from encoder cache (fixed)',
              col1: [s('[1, 12, 1500, 64]', 'cache'), A, note('encoder K/V — computed once')],
              col2: [s('[1, 12, 1500, 64]', 'cache'), A, note('reused every step')],
            },
            {
              name: 'Q · K\u1d40 (cross)', sub: 'decoder Q attends over 1500 enc tokens',
              col1: [s('[1, 12, T_d, 64]', 'act'), X, s('[1, 12, 64, 1500]', 'cache'), A, s('[1, 12, T_d, 1500]', 'out')],
              col2: [s('[1, 12, 1, 64]', 'gemv'), X, s('[1, 12, 64, 1500]', 'cache'), A, s('[1, 12, 1, 1500]', 'out')],
            },
            {
              name: 'Cross-attn + proj', sub: 'context → O proj [768→768]',
              col1: [s('[1, 12, T_d, 1500]', 'act'), X, s('[1, 12, 1500, 64]', 'cache'), A, s('[1, T_d, 768]', 'out')],
              col2: [s('[1, 12, 1, 1500]', 'gemv'), X, s('[1, 12, 1500, 64]', 'cache'), A, s('[1, 1, 768]', 'out')],
            },
            {
              name: 'Residual + LN + FFN', sub: 'LayerNorm → FC(768→3072) → GELU → FC(3072→768)',
              col1: [s('[1, T_d, 768]', 'act'), X, s('[3072, 768]', 'wt'), A, s('[1, T_d, 3072]', 'out')],
              col2: [s('[1, 1, 768]', 'gemv'), X, s('[3072, 768]', 'wt'), A, s('[1, 1, 3072]', 'out'), note('GEMV')],
            },
          ],
        },
      ],
    },
    {
      title: 'Decoder Output',
      tables: [{
        col1Header: '▶ Forward',
        col1Class: 'phase-prefill',
        col2Header: '▶ Step t',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Final LayerNorm', sub: 'decoder norm',
            col1: [s('[1, T_d, 768]', 'act'), X, s('[768]', 'wt'), A, s('[1, T_d, 768]', 'out')],
            col2: [s('[1, 1, 768]', 'act'), X, s('[768]', 'wt'), A, s('[1, 1, 768]', 'out')],
          },
          {
            name: 'proj_out', sub: 'linear to vocab [768→51865]',
            col1: [s('[1, T_d, 768]', 'act'), X, s('[51865, 768]', 'wt'), A, s('[1, T_d, 51865]', 'out')],
            col2: [s('[1, 1, 768]', 'gemv'), X, s('[51865, 768]', 'wt'), A, s('[1, 1, 51865]', 'out'), note('GEMV')],
          },
        ],
      }],
    },
  ],
  kvCache: {
    title: 'Decoder KV Cache Memory  (fp16, max T_d=448)',
    columns: [
      {
        title: 'Self-attn cache (grows)',
        lines: [
          { label: 'K cache shape', value: '[1, 12, T_d, 64]' },
          { label: 'V cache shape', value: '[1, 12, T_d, 64]' },
          { label: 'At T_d=448 @ fp16', value: '6.5 MB' },
        ],
      },
      {
        title: 'Cross-attn cache (fixed)',
        lines: [
          { label: 'Enc K cache (all layers)', value: '[12, 1, 12, 1500, 64]' },
          { label: 'Enc V cache (all layers)', value: '[12, 1, 12, 1500, 64]' },
          { label: 'Fixed at decode start', value: '27.6 MB (precomputed once)' },
        ],
      },
    ],
  },
};
