import { ModelSpec } from '../types';

const s = (shape: string, badge: 'act'|'wt'|'out'|'cache'|'scalar'|'gemv') =>
  ({ kind: 'shape' as const, shape, badge });
const op = (symbol: string) => ({ kind: 'op' as const, symbol });
const note = (text: string) => ({ kind: 'note' as const, text });

const A = op('→'), X = op('×'), P = op('+');

// Gemma 2 2B (2.6B actual params)
// H = 2304, n_h = 8, n_kv = 4, d_h = 256, F = 9216
// L = 26 layers, V = 256000
// GQA ratio 2, alternating local (SWA W=4096) / global attention
// Logit soft-capping: attn=50, output=30
// Note: Q output dim = n_h × d_h = 8 × 256 = 2048 ≠ H = 2304

export const gemma2_2b: ModelSpec = {
  id: 'gemma2-2b',
  name: 'Gemma 2 2B',
  family: 'Gemma',
  category: 'llm',
  paramCount: '2.6B',
  subtitle: 'Dense decoder-only LLM (Gemma 2 family, 2B variant). GQA with 8 query / 4 KV heads. Alternating local sliding-window (W=4096) and global attention layers. Logit soft-capping on attn scores and output logits. GeGLU activation. RoPE positional encoding. Knowledge-distilled from larger models.',
  techniques: ['GQA', 'Sliding Window Attention', 'Logit Soft-Capping', 'GeGLU', 'RoPE', 'RMSNorm', 'Knowledge Distillation'],
  hyperparameters: [
    { symbol: 'H', value: '2304', name: 'hidden size (model dim)' },
    { symbol: 'n_h', value: '8', name: 'query attention heads' },
    { symbol: 'n_kv', value: '4', name: 'KV heads (GQA ratio 2)' },
    { symbol: 'd_h', value: '256', name: 'head dim' },
    { symbol: 'F', value: '9216', name: 'FFN intermediate (4×H)' },
    { symbol: 'L', value: '26', name: 'decoder layers' },
    { symbol: 'V', value: '256000', name: 'vocab size' },
    { symbol: 'W', value: '4096', name: 'sliding window size (local layers)' },
    { symbol: 'S_max', value: '8192', name: 'max context length' },
  ],
  notes: [
    {
      type: 'info',
      html: '<strong>Q/K/V projection dimensions:</strong> Q output = n_h × d_h = 8 × 256 = 2048, K/V output = n_kv × d_h = 4 × 256 = 1024. Note that Q output (2048) ≠ H (2304) — the projection reduces dimensionality.',
    },
    {
      type: 'info',
      html: '<strong>Alternating attention:</strong> Even layers use local sliding-window attention (W=4096 tokens). Odd layers use global full-sequence attention (up to 8192 tokens). This reduces compute cost while maintaining long-range context.',
    },
    {
      type: 'info',
      html: '<strong>Logit soft-capping:</strong> Attention scores are capped via <code>tanh(scores/50) × 50</code>; output logits via <code>tanh(logits/30) × 30</code>. Both are PIM-compatible (tanh via LUT + compile-time-constant scale).',
    },
  ],
  sections: [
    {
      title: '0 · Token Embedding',
      tables: [{
        col1Header: '▶ Prefill  (S tokens)',
        col1Class: 'phase-prefill',
        col2Header: '▶ Decode  (1 token)',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Token embed', sub: 'embed_tokens [V→H]',
            col1: [s('[1, S]', 'act'), X, s('[256000, 2304]', 'wt'), A, s('[1, S, 2304]', 'out')],
            col2: [s('[1, 1]', 'act'), X, s('[256000, 2304]', 'wt'), A, s('[1, 1, 2304]', 'out'), note('GEMV')],
          },
          {
            name: 'Norm embedding', sub: '× √H (Gemma scaling)',
            col1: [s('[1, S, 2304]', 'act'), X, s('√2304', 'scalar'), A, s('[1, S, 2304]', 'out')],
            col2: [s('[1, 1, 2304]', 'act'), X, s('√2304', 'scalar'), A, s('[1, 1, 2304]', 'out')],
          },
        ],
      }],
    },
    {
      title: 'Transformer Block',
      repeat: 26,
      repeatLabel: 'Repeats <span>26 times</span>. Even layers: local SWA (W=4096). Odd layers: global attention.',
      tables: [
        {
          col1Header: '▶ Prefill  (S tokens)',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode  (1 token)',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'RMSNorm', sub: 'input_layernorm',
              col1: [s('[1, S, 2304]', 'act'), X, s('[2304]', 'wt'), A, s('[1, S, 2304]', 'out')],
              col2: [s('[1, 1, 2304]', 'act'), X, s('[2304]', 'wt'), A, s('[1, 1, 2304]', 'out')],
            },
          ],
        },
        {
          col1Header: '▶ Prefill',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'Q projection', sub: 'q_proj [2304→2048], 8 heads',
              col1: [s('[1, S, 2304]', 'act'), X, s('[2048, 2304]', 'wt'), A, s('[1, S, 2048]', 'out')],
              col2: [s('[1, 1, 2304]', 'gemv'), X, s('[2048, 2304]', 'wt'), A, s('[1, 1, 2048]', 'out'), note('GEMV')],
            },
            {
              name: 'K projection', sub: 'k_proj [2304→1024], 4 KV heads (GQA)',
              col1: [s('[1, S, 2304]', 'act'), X, s('[1024, 2304]', 'wt'), A, s('[1, S, 1024]', 'out')],
              col2: [s('[1, 1, 2304]', 'gemv'), X, s('[1024, 2304]', 'wt'), A, s('[1, 1, 1024]', 'out'), note('GEMV')],
            },
            {
              name: 'V projection', sub: 'v_proj [2304→1024], 4 KV heads (GQA)',
              col1: [s('[1, S, 2304]', 'act'), X, s('[1024, 2304]', 'wt'), A, s('[1, S, 1024]', 'out')],
              col2: [s('[1, 1, 2304]', 'gemv'), X, s('[1024, 2304]', 'wt'), A, s('[1, 1, 1024]', 'out'), note('GEMV')],
            },
            {
              name: 'Reshape Q/K/V', sub: 'Q: 8 heads, K/V: 4 heads, d_h=256',
              col1: [s('[1, S, 2048]', 'act'), A, s('[1, 8, S, 256]', 'out'), note('K/V: [1, 4, S, 256]')],
              col2: [s('[1, 1, 2048]', 'act'), A, s('[1, 8, 1, 256]', 'out'), note('K/V: [1, 4, 1, 256]')],
            },
            {
              name: 'RoPE + GQA expand', sub: 'apply RoPE, broadcast KV 4→8',
              col1: [note('RoPE rotation applied to Q, K'), A, note('K/V repeated 2× for GQA')],
              col2: [note('RoPE rotation'), A, note('KV repeat interleave')],
            },
            {
              name: 'KV cache update', sub: 'append new K/V to cache',
              col1: [s('[1, 4, 0, 256]', 'cache'), P, s('[1, 4, S, 256]', 'act'), A, s('[1, 4, S, 256]', 'cache')],
              col2: [s('[1, 4, S_t, 256]', 'cache'), P, s('[1, 4, 1, 256]', 'act'), A, s('[1, 4, S_t+1, 256]', 'cache')],
            },
            {
              name: 'Q · K\u1d40 + soft-cap', sub: 'scores = tanh(Q@K\u1d40/50) × 50',
              col1: [s('[1, 8, S, 256]', 'act'), X, s('[1, 8, 256, S]', 'cache'), A, s('[1, 8, S, S]', 'out'), note('+ soft-cap 50')],
              col2: [s('[1, 8, 1, 256]', 'gemv'), X, s('[1, 8, 256, S_t]', 'cache'), A, s('[1, 8, 1, S_t]', 'out'), note('GEMV + soft-cap')],
            },
            {
              name: 'Softmax + Attn · V', sub: 'context vectors + causal mask',
              col1: [s('[1, 8, S, S]', 'act'), X, s('[1, 8, S, 256]', 'cache'), A, s('[1, 8, S, 256]', 'out')],
              col2: [s('[1, 8, 1, S_t]', 'act'), X, s('[1, 8, S_t, 256]', 'cache'), A, s('[1, 8, 1, 256]', 'out')],
            },
            {
              name: 'O projection', sub: 'o_proj [2048→2304]',
              col1: [s('[1, S, 2048]', 'act'), X, s('[2304, 2048]', 'wt'), A, s('[1, S, 2304]', 'out')],
              col2: [s('[1, 1, 2048]', 'gemv'), X, s('[2304, 2048]', 'wt'), A, s('[1, 1, 2304]', 'out'), note('GEMV')],
            },
            {
              name: 'Post-attn RMSNorm', sub: 'post_attention_layernorm',
              col1: [s('[1, S, 2304]', 'act'), X, s('[2304]', 'wt'), A, s('[1, S, 2304]', 'out')],
              col2: [s('[1, 1, 2304]', 'act'), X, s('[2304]', 'wt'), A, s('[1, 1, 2304]', 'out')],
            },
            {
              name: 'Residual add', sub: 'skip connection',
              col1: [s('[1, S, 2304]', 'act'), P, s('[1, S, 2304]', 'out'), A, s('[1, S, 2304]', 'out')],
              col2: [s('[1, 1, 2304]', 'act'), P, s('[1, 1, 2304]', 'out'), A, s('[1, 1, 2304]', 'out')],
            },
          ],
        },
        {
          col1Header: '▶ Prefill FFN',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode FFN',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'Pre-FFN RMSNorm', sub: 'pre_feedforward_layernorm',
              col1: [s('[1, S, 2304]', 'act'), X, s('[2304]', 'wt'), A, s('[1, S, 2304]', 'out')],
              col2: [s('[1, 1, 2304]', 'act'), X, s('[2304]', 'wt'), A, s('[1, 1, 2304]', 'out')],
            },
            {
              name: 'Gate proj', sub: 'gate_proj [2304→9216] (GeGLU gate)',
              col1: [s('[1, S, 2304]', 'act'), X, s('[9216, 2304]', 'wt'), A, s('[1, S, 9216]', 'out')],
              col2: [s('[1, 1, 2304]', 'gemv'), X, s('[9216, 2304]', 'wt'), A, s('[1, 1, 9216]', 'out'), note('GEMV')],
            },
            {
              name: 'Up proj', sub: 'up_proj [2304→9216]',
              col1: [s('[1, S, 2304]', 'act'), X, s('[9216, 2304]', 'wt'), A, s('[1, S, 9216]', 'out')],
              col2: [s('[1, 1, 2304]', 'gemv'), X, s('[9216, 2304]', 'wt'), A, s('[1, 1, 9216]', 'out'), note('GEMV')],
            },
            {
              name: 'GELU gate × up', sub: 'GeGLU: GELU(gate) ⊙ up',
              col1: [s('[1, S, 9216]', 'act'), X, s('[1, S, 9216]', 'act'), A, s('[1, S, 9216]', 'out')],
              col2: [s('[1, 1, 9216]', 'act'), X, s('[1, 1, 9216]', 'act'), A, s('[1, 1, 9216]', 'out')],
            },
            {
              name: 'Down proj', sub: 'down_proj [9216→2304]',
              col1: [s('[1, S, 9216]', 'act'), X, s('[2304, 9216]', 'wt'), A, s('[1, S, 2304]', 'out')],
              col2: [s('[1, 1, 9216]', 'gemv'), X, s('[2304, 9216]', 'wt'), A, s('[1, 1, 2304]', 'out'), note('GEMV')],
            },
            {
              name: 'Post-FFN RMSNorm', sub: 'post_feedforward_layernorm',
              col1: [s('[1, S, 2304]', 'act'), X, s('[2304]', 'wt'), A, s('[1, S, 2304]', 'out')],
              col2: [s('[1, 1, 2304]', 'act'), X, s('[2304]', 'wt'), A, s('[1, 1, 2304]', 'out')],
            },
            {
              name: 'Residual add', sub: 'skip connection',
              col1: [s('[1, S, 2304]', 'act'), P, s('[1, S, 2304]', 'out'), A, s('[1, S, 2304]', 'out')],
              col2: [s('[1, 1, 2304]', 'act'), P, s('[1, 1, 2304]', 'out'), A, s('[1, 1, 2304]', 'out')],
            },
          ],
        },
      ],
    },
    {
      title: 'LM Head',
      tables: [{
        col1Header: '▶ Prefill',
        col1Class: 'phase-prefill',
        col2Header: '▶ Decode',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Final RMSNorm', sub: 'norm',
            col1: [s('[1, S, 2304]', 'act'), X, s('[2304]', 'wt'), A, s('[1, S, 2304]', 'out')],
            col2: [s('[1, 1, 2304]', 'act'), X, s('[2304]', 'wt'), A, s('[1, 1, 2304]', 'out')],
          },
          {
            name: 'LM head', sub: 'tied embed → logits [2304→256000]',
            col1: [s('[1, S, 2304]', 'act'), X, s('[256000, 2304]', 'wt'), A, s('[1, S, 256000]', 'out')],
            col2: [s('[1, 1, 2304]', 'gemv'), X, s('[256000, 2304]', 'wt'), A, s('[1, 1, 256000]', 'out'), note('GEMV')],
          },
          {
            name: 'Soft-cap logits', sub: 'tanh(logits/30) × 30',
            col1: [s('[1, S, 256000]', 'act'), A, s('[1, S, 256000]', 'out'), note('soft-cap 30')],
            col2: [s('[1, 1, 256000]', 'act'), A, s('[1, 1, 256000]', 'out'), note('soft-cap 30')],
          },
        ],
      }],
    },
  ],
  kvCache: {
    title: 'KV Cache Memory  (fp16, S_t = 2048)',
    columns: [
      {
        title: 'Per layer',
        lines: [
          { label: 'K cache shape', value: '[1, 4, S_t, 256]' },
          { label: 'V cache shape', value: '[1, 4, S_t, 256]' },
          { label: 'Per layer at S=2048 fp16', value: '2 × 4 × 2048 × 256 × 2B = 8.4 MB' },
        ],
      },
      {
        title: 'All 26 layers',
        lines: [
          { label: 'Total KV cache', value: '26 × 8.4 MB = 218 MB' },
          { label: 'At INT4', value: '26 × 2.1 MB = 54 MB' },
        ],
      },
    ],
  },
};
