import { ModelSpec } from '../types';

const s = (shape: string, badge: 'act'|'wt'|'out'|'cache'|'scalar'|'gemv') =>
  ({ kind: 'shape' as const, shape, badge });
const op = (symbol: string) => ({ kind: 'op' as const, symbol });
const note = (text: string) => ({ kind: 'note' as const, text });

const A = op('→'), X = op('×'), P = op('+');

// Whisper Large (v3): 1.55B params
// Encoder: 32 layers, H=1280, n_h=20, d_h=64, F=5120
// Decoder: 32 layers, H=1280, n_h=20, d_h=64, F=5120
// Mel features: 128 channels (v3), T_audio frames
// After subsampling: T_enc = T_audio/2
// Encoder output: [1, T_enc, 1280]
// Decoder: text tokens → 51866-class logits (v3 vocab)

export const whisperLarge: ModelSpec = {
  id: 'whisper-large',
  name: 'Whisper Large (v3)',
  family: 'Whisper',
  category: 'audio',
  paramCount: '1.55B',
  subtitle: 'Encoder-decoder ASR transformer (Large variant). Conv1D frontend on 128 mel bins (v3), sinusoidal position embeddings, full attention in encoder, causal self-attn + cross-attn in decoder. 32 layers each side.',
  techniques: ['Conv Frontend', 'Sinusoidal Pos Embedding', 'Encoder-Decoder', 'Cross-Attention', 'MHA', 'LayerNorm', 'GELU'],
  hyperparameters: [
    { symbol: 'H', value: '1280', name: 'hidden size' },
    { symbol: 'n_h', value: '20', name: 'attention heads' },
    { symbol: 'd_h', value: '64', name: 'head dim = H/n_h' },
    { symbol: 'F', value: '5120', name: 'FFN intermediate (4×H)' },
    { symbol: 'L_e', value: '32', name: 'encoder layers' },
    { symbol: 'L_d', value: '32', name: 'decoder layers' },
    { symbol: 'V', value: '51866', name: 'vocab size (v3 multilingual)' },
    { symbol: 'M', value: '128', name: 'mel filter banks (v3)' },
    { symbol: 'T_e', value: '1500', name: 'encoder seq len (30s audio)' },
    { symbol: 'T_d', value: 'T_d', name: 'decoder generated length' },
  ],
  notes: [
    {
      type: 'info',
      html: '<strong>Conv frontend:</strong> Two Conv1D layers compress 128-dim mel spectrograms (v3 uses 128 instead of 80) to 1280-dim features at 2× lower frame rate: Conv1(128→1280, k=3) → GELU → Conv1(1280→1280, k=3, s=2) → GELU. Sinusoidal positional embeddings are then added.',
    },
    {
      type: 'info',
      html: '<strong>Encoder-Decoder architecture:</strong> The encoder processes the full audio sequence (full attention, no causal mask) and produces 1500 context vectors. The decoder generates text tokens autoregressively using causal self-attention and cross-attention over the encoder output.',
    },
    {
      type: 'info',
      html: '<strong>PIM relevance:</strong> At 1.55B parameters (~775 MB at FP16, ~388 MB at W4), Whisper Large fits in a single DDR5-32Gb chip. With 20 heads and 32 banks, bank_dim=heads requires padding to 32 heads (12 idle banks) or bank_dim=hidden_dim. The encoder is compute-bound (non-autoregressive); the decoder is memory-bound (autoregressive, analogous to LLM decode).',
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
            name: 'Mel spectrogram', sub: '128 mel bins × T frames (v3)',
            col1: [s('[1, 128, T_audio]', 'act'), A, note('e.g. T=3000 for 30s')],
            col2: [note('hop=10ms → 100 frames/sec')],
          },
          {
            name: 'Conv1D block 1', sub: 'Conv(128→1280, k=3, pad=1) + GELU',
            col1: [s('[1, 128, T_audio]', 'act'), X, s('[1280, 128, 3]', 'wt'), A, s('[1, 1280, T_audio]', 'out')],
            col2: [note('same-length output')],
          },
          {
            name: 'Conv1D block 2', sub: 'Conv(1280→1280, k=3, s=2, pad=1) + GELU',
            col1: [s('[1, 1280, T_audio]', 'act'), X, s('[1280, 1280, 3]', 'wt'), A, s('[1, 1280, T_audio/2]', 'out')],
            col2: [note('stride 2 halves length → T_enc=1500')],
          },
          {
            name: 'Transpose + pos embed', sub: '[C, T] → [T, C] + sinusoidal',
            col1: [s('[1, 1280, 1500]', 'act'), A, s('[1, 1500, 1280]', 'out'), P, s('[1500, 1280]', 'scalar'), A, s('[1, 1500, 1280]', 'out')],
            col2: [note('fixed sinusoidal, not learned')],
          },
        ],
      }],
    },
    {
      title: 'Encoder Block',
      repeat: 32,
      repeatLabel: 'Repeats <span>32 times</span>. Bidirectional (no causal mask). K/V from encoder only — no cross-attn.',
      tables: [
        {
          col1Header: '▶ Encoder',
          col1Class: 'phase-prefill',
          col2Header: '▶ Shapes',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm', sub: 'encoder_attn_layer_norm',
              col1: [s('[1, 1500, 1280]', 'act'), X, s('[1280]', 'wt'), A, s('[1, 1500, 1280]', 'out')],
              col2: [],
            },
            {
              name: 'Q/K/V proj (MHA)', sub: '[1280→1280] each, 20 heads',
              col1: [s('[1, 1500, 1280]', 'act'), X, s('[1280, 1280]', 'wt'), A, s('[1, 1500, 1280]', 'out'), X, s('(×3)', 'scalar')],
              col2: [s('[1, 20, 1500, 64]', 'out'), X, s('(×3)', 'scalar')],
            },
            {
              name: 'Full self-attn + proj', sub: 'no causal mask, [1280→1280]',
              col1: [s('[1, 20, 1500, 1500]', 'act'), X, s('[1, 20, 1500, 64]', 'act'), A, s('[1, 1500, 1280]', 'out')],
              col2: [note('1500² attn matrix')],
            },
            {
              name: 'Residual + LN + FFN', sub: 'LayerNorm → FC(1280→5120) → GELU → FC(5120→1280) → residual',
              col1: [s('[1, 1500, 1280]', 'act'), X, s('[5120, 1280]', 'wt'), A, s('[1, 1500, 5120]', 'out')],
              col2: [s('[1, 1500, 1280]', 'out')],
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
            col1: [s('[1, 1500, 1280]', 'act'), X, s('[1280]', 'wt'), A, s('[1, 1500, 1280]', 'out')],
            col2: [],
          },
          {
            name: 'Encoder K/V proj', sub: 'pre-computed for all 32 decoder layers',
            col1: [s('[1, 1500, 1280]', 'act'), X, s('[1280, 1280]', 'wt'), A, s('[1, 1500, 1280]', 'cache'), X, s('(×2 per layer)', 'scalar')],
            col2: [note('K,V cached × 32 decoder layers')],
          },
        ],
      }],
    },
    {
      title: 'Decoder Step',
      repeat: 32,
      repeatLabel: 'Decoder: <span>32 layers</span>. Each step: causal self-attn over generated tokens, then cross-attn over encoder output.',
      tables: [
        {
          col1Header: '▶ Input token',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode step t',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'Token embed', sub: 'decoder embed_tokens',
              col1: [s('[1, T_d]', 'act'), X, s('[51866, 1280]', 'wt'), A, s('[1, T_d, 1280]', 'out')],
              col2: [s('[1, 1]', 'act'), X, s('[51866, 1280]', 'wt'), A, s('[1, 1, 1280]', 'out'), note('GEMV')],
            },
            {
              name: 'Pos embed', sub: 'learned [448, 1280]',
              col1: [s('[1, T_d, 1280]', 'act'), P, s('[T_d, 1280]', 'scalar'), A, s('[1, T_d, 1280]', 'out')],
              col2: [s('[1, 1, 1280]', 'act'), P, s('[t, 1280]', 'scalar'), A, s('[1, 1, 1280]', 'out')],
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
              col1: [s('[1, T_d, 1280]', 'act'), X, s('[1280]', 'wt'), A, s('[1, T_d, 1280]', 'out')],
              col2: [s('[1, 1, 1280]', 'act'), X, s('[1280]', 'wt'), A, s('[1, 1, 1280]', 'out')],
            },
            {
              name: 'Causal self-attn Q/K/V', sub: 'masked self-attn over generated tokens',
              col1: [s('[1, T_d, 1280]', 'act'), X, s('[1280, 1280]', 'wt'), A, s('[1, T_d, 1280]', 'out'), X, s('(×3)', 'scalar')],
              col2: [s('[1, 1, 1280]', 'gemv'), X, s('[1280, 1280]', 'wt'), A, s('[1, 1, 1280]', 'out'), note('GEMV')],
            },
            {
              name: 'Decoder KV cache', sub: 'self-attn cache grows',
              col1: [s('[1, 20, 0, 64]', 'cache'), P, s('[1, 20, T_d, 64]', 'act'), A, s('[1, 20, T_d, 64]', 'cache')],
              col2: [s('[1, 20, t, 64]', 'cache'), P, s('[1, 20, 1, 64]', 'act'), A, s('[1, 20, t+1, 64]', 'cache')],
            },
            {
              name: 'Causal attn + proj', sub: 'O proj [1280→1280]',
              col1: [s('[1, 20, T_d, T_d]', 'act'), X, s('[1, 20, T_d, 64]', 'cache'), A, s('[1, T_d, 1280]', 'out')],
              col2: [s('[1, 20, 1, t]', 'gemv'), X, s('[1, 20, t, 64]', 'cache'), A, s('[1, 1, 1280]', 'out')],
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
              col1: [s('[1, T_d, 1280]', 'act'), X, s('[1280]', 'wt'), A, s('[1, T_d, 1280]', 'out')],
              col2: [s('[1, 1, 1280]', 'act'), X, s('[1280]', 'wt'), A, s('[1, 1, 1280]', 'out')],
            },
            {
              name: 'Cross-attn Q proj', sub: 'Q from decoder [1280→1280]',
              col1: [s('[1, T_d, 1280]', 'act'), X, s('[1280, 1280]', 'wt'), A, s('[1, T_d, 1280]', 'out')],
              col2: [s('[1, 1, 1280]', 'gemv'), X, s('[1280, 1280]', 'wt'), A, s('[1, 1, 1280]', 'out'), note('GEMV')],
            },
            {
              name: 'Cross-attn K/V', sub: 'K,V from encoder cache (fixed)',
              col1: [s('[1, 20, 1500, 64]', 'cache'), A, note('encoder K/V — computed once')],
              col2: [s('[1, 20, 1500, 64]', 'cache'), A, note('reused every step')],
            },
            {
              name: 'Q · K\u1d40 (cross)', sub: 'decoder Q attends over 1500 enc tokens',
              col1: [s('[1, 20, T_d, 64]', 'act'), X, s('[1, 20, 64, 1500]', 'cache'), A, s('[1, 20, T_d, 1500]', 'out')],
              col2: [s('[1, 20, 1, 64]', 'gemv'), X, s('[1, 20, 64, 1500]', 'cache'), A, s('[1, 20, 1, 1500]', 'out')],
            },
            {
              name: 'Cross-attn + proj', sub: 'context → O proj [1280→1280]',
              col1: [s('[1, 20, T_d, 1500]', 'act'), X, s('[1, 20, 1500, 64]', 'cache'), A, s('[1, T_d, 1280]', 'out')],
              col2: [s('[1, 20, 1, 1500]', 'gemv'), X, s('[1, 20, 1500, 64]', 'cache'), A, s('[1, 1, 1280]', 'out')],
            },
            {
              name: 'Residual + LN + FFN', sub: 'LayerNorm → FC(1280→5120) → GELU → FC(5120→1280)',
              col1: [s('[1, T_d, 1280]', 'act'), X, s('[5120, 1280]', 'wt'), A, s('[1, T_d, 5120]', 'out')],
              col2: [s('[1, 1, 1280]', 'gemv'), X, s('[5120, 1280]', 'wt'), A, s('[1, 1, 5120]', 'out'), note('GEMV')],
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
            col1: [s('[1, T_d, 1280]', 'act'), X, s('[1280]', 'wt'), A, s('[1, T_d, 1280]', 'out')],
            col2: [s('[1, 1, 1280]', 'act'), X, s('[1280]', 'wt'), A, s('[1, 1, 1280]', 'out')],
          },
          {
            name: 'proj_out', sub: 'linear to vocab [1280→51866]',
            col1: [s('[1, T_d, 1280]', 'act'), X, s('[51866, 1280]', 'wt'), A, s('[1, T_d, 51866]', 'out')],
            col2: [s('[1, 1, 1280]', 'gemv'), X, s('[51866, 1280]', 'wt'), A, s('[1, 1, 51866]', 'out'), note('GEMV')],
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
          { label: 'K cache shape', value: '[1, 20, T_d, 64]' },
          { label: 'V cache shape', value: '[1, 20, T_d, 64]' },
          { label: 'Per layer @ T_d=448 fp16', value: '2 × 20 × 448 × 64 × 2B = 2.3 MB' },
          { label: 'All 32 layers @ T_d=448', value: '73.0 MB' },
        ],
      },
      {
        title: 'Cross-attn cache (fixed)',
        lines: [
          { label: 'Enc K cache (all layers)', value: '[32, 1, 20, 1500, 64]' },
          { label: 'Enc V cache (all layers)', value: '[32, 1, 20, 1500, 64]' },
          { label: 'Fixed at decode start', value: '245.8 MB (precomputed once)' },
        ],
      },
    ],
  },
};
