import { ModelSpec } from '../types';

const s = (shape: string, badge: 'act'|'wt'|'out'|'cache'|'scalar'|'gemv') =>
  ({ kind: 'shape' as const, shape, badge });
const op = (symbol: string) => ({ kind: 'op' as const, symbol });
const note = (text: string) => ({ kind: 'note' as const, text });

const A = op('→'), X = op('×'), P = op('+'), E = op('⊙');

export const gpt2: ModelSpec = {
  id: 'gpt2',
  name: 'GPT-2 (117M)',
  family: 'GPT',
  category: 'llm',
  paramCount: '117M',
  subtitle: 'Decoder-only, learned absolute positional embeddings, MHA, LayerNorm, GELU FFN. The original open autoregressive transformer baseline.',
  techniques: ['MHA', 'Absolute Pos Embedding', 'LayerNorm', 'GELU'],
  hyperparameters: [
    { symbol: 'H', value: '768', name: 'hidden size' },
    { symbol: 'n_h', value: '12', name: 'attention heads' },
    { symbol: 'd_h', value: '64', name: 'head dim = H/n_h' },
    { symbol: 'n_kv', value: '12', name: 'KV heads (MHA)' },
    { symbol: 'F', value: '3072', name: 'FFN intermediate' },
    { symbol: 'L', value: '12', name: 'decoder layers' },
    { symbol: 'V', value: '50257', name: 'vocab size' },
    { symbol: 'S', value: 'S_p / 1', name: 'prefill / decode' },
  ],
  notes: [
    {
      type: 'gemv',
      html: '<strong>Decode is memory-bandwidth bound.</strong> S = 1 collapses every projection into a <strong>vector × matrix (GEMV)</strong>: the full weight is streamed from DRAM to produce a single output vector. Arithmetic intensity ≈ 1 FLOP / byte. Prefill (large S_p) is compute-bound GEMM.',
    },
    {
      type: 'info',
      html: '<strong>Absolute positional embeddings:</strong> A learned [1024, 768] table is added to token embeddings. No rotary or ALiBi — position info is baked in at the input layer only.',
    },
  ],
  sections: [
    {
      title: '0 · Token + Position Embedding',
      tables: [{
        col1Header: '▶ Prefill  (S = S_p)',
        col1Class: 'phase-prefill',
        col2Header: '▶ Decode  (S = 1, step t)',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'wte', sub: 'token embedding lookup',
            col1: [s('[1, S_p]', 'act'), X, s('[50257, 768]', 'wt'), A, s('[1, S_p, 768]', 'out')],
            col2: [s('[1, 1]', 'gemv'), X, s('[50257, 768]', 'wt'), A, s('[1, 1, 768]', 'out')],
          },
          {
            name: 'wpe', sub: 'position embedding lookup',
            col1: [s('[S_p, 768]', 'scalar'), A, s('[S_p, 768]', 'out')],
            col2: [s('[1, 768]', 'scalar'), A, s('[1, 768]', 'out')],
          },
          {
            name: 'x = wte + wpe', sub: 'elementwise add',
            col1: [s('[1, S_p, 768]', 'act'), P, s('[1, S_p, 768]', 'scalar'), A, s('[1, S_p, 768]', 'out')],
            col2: [s('[1, 1, 768]', 'act'), P, s('[1, 1, 768]', 'scalar'), A, s('[1, 1, 768]', 'out')],
          },
        ],
      }],
    },
    {
      title: 'Decoder Layer',
      repeat: 12,
      repeatLabel: 'Repeats <span>12 times</span>. Weight shapes are identical in every layer.',
      tables: [
        {
          col1Header: '▶ Prefill',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm', sub: 'ln_1 (pre-attention)',
              col1: [s('[1, S_p, 768]', 'act'), X, s('[768]', 'wt'), A, s('[1, S_p, 768]', 'out')],
              col2: [s('[1, 1, 768]', 'gemv'), X, s('[768]', 'wt'), A, s('[1, 1, 768]', 'out')],
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
              name: 'Q projection', sub: 'c_attn Q slice · GEMM',
              col1: [s('[1, S_p, 768]', 'act'), X, s('[768, 768]', 'wt'), A, s('[1, S_p, 768]', 'out')],
              col2: [s('[1, 1, 768]', 'gemv'), X, s('[768, 768]', 'wt'), A, s('[1, 1, 768]', 'out'), note('GEMV')],
            },
            {
              name: 'K projection', sub: 'c_attn K slice · GEMM',
              col1: [s('[1, S_p, 768]', 'act'), X, s('[768, 768]', 'wt'), A, s('[1, S_p, 768]', 'out')],
              col2: [s('[1, 1, 768]', 'gemv'), X, s('[768, 768]', 'wt'), A, s('[1, 1, 768]', 'out'), note('GEMV')],
            },
            {
              name: 'V projection', sub: 'c_attn V slice · GEMM',
              col1: [s('[1, S_p, 768]', 'act'), X, s('[768, 768]', 'wt'), A, s('[1, S_p, 768]', 'out')],
              col2: [s('[1, 1, 768]', 'gemv'), X, s('[768, 768]', 'wt'), A, s('[1, 1, 768]', 'out'), note('GEMV')],
            },
            {
              name: 'Reshape Q', sub: 'view: split heads',
              col1: [s('[1, S_p, 768]', 'act'), A, s('[1, 12, S_p, 64]', 'out')],
              col2: [s('[1, 1, 768]', 'act'), A, s('[1, 12, 1, 64]', 'out')],
            },
            {
              name: 'Reshape K', sub: 'view: split heads',
              col1: [s('[1, S_p, 768]', 'act'), A, s('[1, 12, S_p, 64]', 'out')],
              col2: [s('[1, 1, 768]', 'act'), A, s('[1, 12, 1, 64]', 'out')],
            },
            {
              name: 'Reshape V', sub: 'view: split heads',
              col1: [s('[1, S_p, 768]', 'act'), A, s('[1, 12, S_p, 64]', 'out')],
              col2: [s('[1, 1, 768]', 'act'), A, s('[1, 12, 1, 64]', 'out')],
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
              name: 'KV cache concat K', sub: 'cat(cache, K, dim=2)',
              col1: [s('[1, 12, 0, 64]', 'cache'), P, s('[1, 12, S_p, 64]', 'act'), A, s('[1, 12, S_p, 64]', 'cache')],
              col2: [s('[1, 12, S_t, 64]', 'cache'), P, s('[1, 12, 1, 64]', 'act'), A, s('[1, 12, S_t+1, 64]', 'cache')],
            },
            {
              name: 'KV cache concat V', sub: 'cat(cache, V, dim=2)',
              col1: [s('[1, 12, 0, 64]', 'cache'), P, s('[1, 12, S_p, 64]', 'act'), A, s('[1, 12, S_p, 64]', 'cache')],
              col2: [s('[1, 12, S_t, 64]', 'cache'), P, s('[1, 12, 1, 64]', 'act'), A, s('[1, 12, S_t+1, 64]', 'cache')],
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
              name: 'Q · K\u1d40', sub: 'attn scores',
              col1: [s('[1, 12, S_p, 64]', 'act'), X, s('[1, 12, 64, S_p]', 'cache'), A, s('[1, 12, S_p, S_p]', 'out')],
              col2: [s('[1, 12, 1, 64]', 'gemv'), X, s('[1, 12, 64, S_t]', 'cache'), A, s('[1, 12, 1, S_t]', 'out'), note('dot per head')],
            },
            {
              name: 'Scale ÷ √64', sub: 'scalar = 0.125',
              col1: [s('[1, 12, S_p, S_p]', 'act'), X, s('0.125', 'scalar'), A, s('[1, 12, S_p, S_p]', 'out')],
              col2: [s('[1, 12, 1, S_t]', 'act'), X, s('0.125', 'scalar'), A, s('[1, 12, 1, S_t]', 'out')],
            },
            {
              name: 'Causal mask', sub: 'additive (0 or −∞)',
              col1: [s('[1, 12, S_p, S_p]', 'act'), P, s('[1, 1, S_p, S_p]', 'scalar'), A, s('[1, 12, S_p, S_p]', 'out')],
              col2: [s('[1, 12, 1, S_t]', 'act'), P, s('[1, 1, 1, S_t]', 'scalar'), A, s('[1, 12, 1, S_t]', 'out'), note('all zeros')],
            },
            {
              name: 'Softmax', sub: 'dim = −1',
              col1: [s('[1, 12, S_p, S_p]', 'act'), A, s('[1, 12, S_p, S_p]', 'out')],
              col2: [s('[1, 12, 1, S_t]', 'act'), A, s('[1, 12, 1, S_t]', 'out')],
            },
            {
              name: 'Attn · V', sub: 'context vectors',
              col1: [s('[1, 12, S_p, S_p]', 'act'), X, s('[1, 12, S_p, 64]', 'cache'), A, s('[1, 12, S_p, 64]', 'out')],
              col2: [s('[1, 12, 1, S_t]', 'gemv'), X, s('[1, 12, S_t, 64]', 'cache'), A, s('[1, 12, 1, 64]', 'out'), note('weighted sum')],
            },
            {
              name: 'Reshape', sub: 'merge heads',
              col1: [s('[1, 12, S_p, 64]', 'act'), A, s('[1, S_p, 768]', 'out')],
              col2: [s('[1, 12, 1, 64]', 'act'), A, s('[1, 1, 768]', 'out')],
            },
            {
              name: 'O projection', sub: 'c_proj · GEMM',
              col1: [s('[1, S_p, 768]', 'act'), X, s('[768, 768]', 'wt'), A, s('[1, S_p, 768]', 'out')],
              col2: [s('[1, 1, 768]', 'gemv'), X, s('[768, 768]', 'wt'), A, s('[1, 1, 768]', 'out'), note('GEMV')],
            },
            {
              name: 'Residual add', sub: 'skip connection',
              col1: [s('[1, S_p, 768]', 'act'), P, s('[1, S_p, 768]', 'out'), A, s('[1, S_p, 768]', 'out')],
              col2: [s('[1, 1, 768]', 'act'), P, s('[1, 1, 768]', 'out'), A, s('[1, 1, 768]', 'out')],
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
              name: 'LayerNorm', sub: 'ln_2 (pre-FFN)',
              col1: [s('[1, S_p, 768]', 'act'), X, s('[768]', 'wt'), A, s('[1, S_p, 768]', 'out')],
              col2: [s('[1, 1, 768]', 'gemv'), X, s('[768]', 'wt'), A, s('[1, 1, 768]', 'out')],
            },
            {
              name: 'FC up', sub: 'c_fc · GEMM',
              col1: [s('[1, S_p, 768]', 'act'), X, s('[3072, 768]', 'wt'), A, s('[1, S_p, 3072]', 'out')],
              col2: [s('[1, 1, 768]', 'gemv'), X, s('[3072, 768]', 'wt'), A, s('[1, 1, 3072]', 'out'), note('GEMV')],
            },
            {
              name: 'GELU', sub: 'elementwise activation',
              col1: [s('[1, S_p, 3072]', 'act'), A, s('[1, S_p, 3072]', 'out')],
              col2: [s('[1, 1, 3072]', 'act'), A, s('[1, 1, 3072]', 'out')],
            },
            {
              name: 'FC down', sub: 'c_proj · GEMM',
              col1: [s('[1, S_p, 3072]', 'act'), X, s('[768, 3072]', 'wt'), A, s('[1, S_p, 768]', 'out')],
              col2: [s('[1, 1, 3072]', 'gemv'), X, s('[768, 3072]', 'wt'), A, s('[1, 1, 768]', 'out'), note('GEMV')],
            },
            {
              name: 'Residual add', sub: 'skip connection',
              col1: [s('[1, S_p, 768]', 'act'), P, s('[1, S_p, 768]', 'out'), A, s('[1, S_p, 768]', 'out')],
              col2: [s('[1, 1, 768]', 'act'), P, s('[1, 1, 768]', 'out'), A, s('[1, 1, 768]', 'out')],
            },
          ],
        },
      ],
    },
    {
      title: '2 · Final LayerNorm',
      tables: [{
        col1Header: '▶ Prefill',
        col1Class: 'phase-prefill',
        col2Header: '▶ Decode',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'LayerNorm', sub: 'ln_f',
            col1: [s('[1, S_p, 768]', 'act'), X, s('[768]', 'wt'), A, s('[1, S_p, 768]', 'out')],
            col2: [s('[1, 1, 768]', 'gemv'), X, s('[768]', 'wt'), A, s('[1, 1, 768]', 'out')],
          },
        ],
      }],
    },
    {
      title: '3 · LM Head (tied weights)',
      tables: [{
        col1Header: '▶ Prefill',
        col1Class: 'phase-prefill',
        col2Header: '▶ Decode',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'lm_head', sub: 'linear, tied to wte',
            col1: [s('[1, S_p, 768]', 'act'), X, s('[50257, 768]', 'wt'), A, s('[1, S_p, 50257]', 'out')],
            col2: [s('[1, 1, 768]', 'gemv'), X, s('[50257, 768]', 'wt'), A, s('[1, 1, 50257]', 'out'), note('GEMV')],
          },
          {
            name: 'Next-token slice', sub: 'logits[:, −1, :]',
            col1: [s('[1, S_p, 50257]', 'act'), A, s('[1, 50257]', 'out')],
            col2: [s('[1, 1, 50257]', 'act'), A, s('[1, 50257]', 'out')],
          },
        ],
      }],
    },
  ],
  kvCache: {
    title: 'KV Cache Memory  (fp16, full context S_total = 1024)',
    columns: [
      {
        title: 'Per layer',
        lines: [
          { label: 'K cache shape', value: '[1, 12, 1024, 64]' },
          { label: 'V cache shape', value: '[1, 12, 1024, 64]' },
          { label: 'Elements (K + V)', value: '1,572,864' },
          { label: 'Memory @ fp16', value: '3 MB' },
        ],
      },
      {
        title: 'All 12 layers',
        lines: [
          { label: 'K cache shape', value: '[12, 1, 12, 1024, 64]' },
          { label: 'V cache shape', value: '[12, 1, 12, 1024, 64]' },
          { label: 'Total elements', value: '18,874,368' },
          { label: 'Total @ fp16', value: '36 MB' },
        ],
      },
    ],
  },
};
