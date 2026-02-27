import { ModelSpec } from '../types';

const s = (shape: string, badge: 'act'|'wt'|'out'|'cache'|'scalar'|'gemv') =>
  ({ kind: 'shape' as const, shape, badge });
const op = (symbol: string) => ({ kind: 'op' as const, symbol });
const note = (text: string) => ({ kind: 'note' as const, text });

const A = op('→'), X = op('×'), P = op('+'), E = op('⊙');

export const llama2_7b: ModelSpec = {
  id: 'llama2-7b',
  name: 'LLaMA 2 (7B)',
  family: 'LLaMA',
  category: 'llm',
  paramCount: '6.7B',
  subtitle: 'Decoder-only, RoPE, RMSNorm, SwiGLU, MHA (32 KV heads). The foundational open-weight LLM architecture.',
  techniques: ['MHA', 'RoPE', 'RMSNorm', 'SwiGLU'],
  hyperparameters: [
    { symbol: 'H', value: '4096', name: 'hidden size' },
    { symbol: 'n_h', value: '32', name: 'attention heads' },
    { symbol: 'd_h', value: '128', name: 'head dim = H/n_h' },
    { symbol: 'n_kv', value: '32', name: 'KV heads (MHA)' },
    { symbol: 'F', value: '11008', name: 'FFN intermediate' },
    { symbol: 'L', value: '32', name: 'decoder layers' },
    { symbol: 'V', value: '32000', name: 'vocab size' },
    { symbol: 'S', value: 'S_p / 1', name: 'prefill / decode' },
  ],
  notes: [
    {
      type: 'gemv',
      html: '<strong>Decode is memory-bandwidth bound.</strong> S = 1 collapses every projection into a <strong>vector × matrix (GEMV)</strong>: the full weight is streamed from DRAM to produce a single output vector. Arithmetic intensity ≈ 1 FLOP / byte. Prefill (large S_p) is compute-bound GEMM.',
    },
  ],
  sections: [
    {
      title: '0 · Token Embedding',
      tables: [{
        col1Header: '▶ Prefill  (S = S_p)',
        col1Class: 'phase-prefill',
        col2Header: '▶ Decode  (S = 1, step t)',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'embed_tokens', sub: 'table lookup',
            col1: [s('[1, S_p]', 'act'), X, s('[32000, 4096]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
            col2: [s('[1, 1]', 'gemv'), X, s('[32000, 4096]', 'wt'), A, s('[1, 1, 4096]', 'out')],
          },
        ],
      }],
    },
    {
      title: 'Decoder Layer',
      repeat: 32,
      repeatLabel: 'Repeats <span>32 times</span>. Weight shapes are identical in every layer.',
      tables: [
        {
          col1Header: '▶ Prefill  (S = S_p)',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode  (S = 1)',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'RMSNorm', sub: 'input_layernorm',
              col1: [s('[1, S_p, 4096]', 'act'), X, s('[4096]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
              col2: [s('[1, 1, 4096]', 'gemv'), X, s('[4096]', 'wt'), A, s('[1, 1, 4096]', 'out')],
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
              name: 'Q projection', sub: 'q_proj · GEMM',
              col1: [s('[1, S_p, 4096]', 'act'), X, s('[4096, 4096]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
              col2: [s('[1, 1, 4096]', 'gemv'), X, s('[4096, 4096]', 'wt'), A, s('[1, 1, 4096]', 'out'), note('GEMV')],
            },
            {
              name: 'K projection', sub: 'k_proj · GEMM',
              col1: [s('[1, S_p, 4096]', 'act'), X, s('[4096, 4096]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
              col2: [s('[1, 1, 4096]', 'gemv'), X, s('[4096, 4096]', 'wt'), A, s('[1, 1, 4096]', 'out'), note('GEMV')],
            },
            {
              name: 'V projection', sub: 'v_proj · GEMM',
              col1: [s('[1, S_p, 4096]', 'act'), X, s('[4096, 4096]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
              col2: [s('[1, 1, 4096]', 'gemv'), X, s('[4096, 4096]', 'wt'), A, s('[1, 1, 4096]', 'out'), note('GEMV')],
            },
            {
              name: 'Reshape Q', sub: 'view: split heads',
              col1: [s('[1, S_p, 4096]', 'act'), A, s('[1, 32, S_p, 128]', 'out')],
              col2: [s('[1, 1, 4096]', 'act'), A, s('[1, 32, 1, 128]', 'out')],
            },
            {
              name: 'Reshape K', sub: 'view: split heads',
              col1: [s('[1, S_p, 4096]', 'act'), A, s('[1, 32, S_p, 128]', 'out')],
              col2: [s('[1, 1, 4096]', 'act'), A, s('[1, 32, 1, 128]', 'out')],
            },
            {
              name: 'Reshape V', sub: 'view: split heads',
              col1: [s('[1, S_p, 4096]', 'act'), A, s('[1, 32, S_p, 128]', 'out')],
              col2: [s('[1, 1, 4096]', 'act'), A, s('[1, 32, 1, 128]', 'out')],
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
              name: 'RoPE → Q', sub: 'rotate_half · elementwise',
              col1: [s('[1, 32, S_p, 128]', 'act'), X, s('[S_p, 64]', 'scalar'), A, s('[1, 32, S_p, 128]', 'out')],
              col2: [s('[1, 32, 1, 128]', 'act'), X, s('[1, 64]', 'scalar'), A, s('[1, 32, 1, 128]', 'out')],
            },
            {
              name: 'RoPE → K', sub: 'rotate_half · elementwise',
              col1: [s('[1, 32, S_p, 128]', 'act'), X, s('[S_p, 64]', 'scalar'), A, s('[1, 32, S_p, 128]', 'out')],
              col2: [s('[1, 32, 1, 128]', 'act'), X, s('[1, 64]', 'scalar'), A, s('[1, 32, 1, 128]', 'out')],
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
              col1: [s('[1, 32, 0, 128]', 'cache'), P, s('[1, 32, S_p, 128]', 'act'), A, s('[1, 32, S_p, 128]', 'cache')],
              col2: [s('[1, 32, S_t, 128]', 'cache'), P, s('[1, 32, 1, 128]', 'act'), A, s('[1, 32, S_t+1, 128]', 'cache')],
            },
            {
              name: 'KV cache concat V', sub: 'cat(cache, V, dim=2)',
              col1: [s('[1, 32, 0, 128]', 'cache'), P, s('[1, 32, S_p, 128]', 'act'), A, s('[1, 32, S_p, 128]', 'cache')],
              col2: [s('[1, 32, S_t, 128]', 'cache'), P, s('[1, 32, 1, 128]', 'act'), A, s('[1, 32, S_t+1, 128]', 'cache')],
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
              col1: [s('[1, 32, S_p, 128]', 'act'), X, s('[1, 32, 128, S_p]', 'cache'), A, s('[1, 32, S_p, S_p]', 'out')],
              col2: [s('[1, 32, 1, 128]', 'gemv'), X, s('[1, 32, 128, S_t]', 'cache'), A, s('[1, 32, 1, S_t]', 'out'), note('dot per head')],
            },
            {
              name: 'Scale ÷ √128', sub: 'scalar = 0.0884',
              col1: [s('[1, 32, S_p, S_p]', 'act'), X, s('0.0884', 'scalar'), A, s('[1, 32, S_p, S_p]', 'out')],
              col2: [s('[1, 32, 1, S_t]', 'act'), X, s('0.0884', 'scalar'), A, s('[1, 32, 1, S_t]', 'out')],
            },
            {
              name: 'Causal mask', sub: 'additive (0 or −∞)',
              col1: [s('[1, 32, S_p, S_p]', 'act'), P, s('[1, 1, S_p, S_p]', 'scalar'), A, s('[1, 32, S_p, S_p]', 'out')],
              col2: [s('[1, 32, 1, S_t]', 'act'), P, s('[1, 1, 1, S_t]', 'scalar'), A, s('[1, 32, 1, S_t]', 'out'), note('all zeros')],
            },
            {
              name: 'Softmax', sub: 'dim = −1',
              col1: [s('[1, 32, S_p, S_p]', 'act'), A, s('[1, 32, S_p, S_p]', 'out')],
              col2: [s('[1, 32, 1, S_t]', 'act'), A, s('[1, 32, 1, S_t]', 'out')],
            },
            {
              name: 'Attn · V', sub: 'context vectors',
              col1: [s('[1, 32, S_p, S_p]', 'act'), X, s('[1, 32, S_p, 128]', 'cache'), A, s('[1, 32, S_p, 128]', 'out')],
              col2: [s('[1, 32, 1, S_t]', 'gemv'), X, s('[1, 32, S_t, 128]', 'cache'), A, s('[1, 32, 1, 128]', 'out'), note('weighted sum')],
            },
            {
              name: 'Reshape', sub: 'merge heads',
              col1: [s('[1, 32, S_p, 128]', 'act'), A, s('[1, S_p, 4096]', 'out')],
              col2: [s('[1, 32, 1, 128]', 'act'), A, s('[1, 1, 4096]', 'out')],
            },
            {
              name: 'O projection', sub: 'o_proj · GEMM',
              col1: [s('[1, S_p, 4096]', 'act'), X, s('[4096, 4096]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
              col2: [s('[1, 1, 4096]', 'gemv'), X, s('[4096, 4096]', 'wt'), A, s('[1, 1, 4096]', 'out'), note('GEMV')],
            },
            {
              name: 'Residual add', sub: 'skip connection',
              col1: [s('[1, S_p, 4096]', 'act'), P, s('[1, S_p, 4096]', 'out'), A, s('[1, S_p, 4096]', 'out')],
              col2: [s('[1, 1, 4096]', 'act'), P, s('[1, 1, 4096]', 'out'), A, s('[1, 1, 4096]', 'out')],
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
              name: 'RMSNorm', sub: 'post_attention_layernorm',
              col1: [s('[1, S_p, 4096]', 'act'), X, s('[4096]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
              col2: [s('[1, 1, 4096]', 'gemv'), X, s('[4096]', 'wt'), A, s('[1, 1, 4096]', 'out')],
            },
            {
              name: 'Gate projection', sub: 'gate_proj · GEMM',
              col1: [s('[1, S_p, 4096]', 'act'), X, s('[11008, 4096]', 'wt'), A, s('[1, S_p, 11008]', 'out')],
              col2: [s('[1, 1, 4096]', 'gemv'), X, s('[11008, 4096]', 'wt'), A, s('[1, 1, 11008]', 'out'), note('GEMV')],
            },
            {
              name: 'Up projection', sub: 'up_proj · GEMM',
              col1: [s('[1, S_p, 4096]', 'act'), X, s('[11008, 4096]', 'wt'), A, s('[1, S_p, 11008]', 'out')],
              col2: [s('[1, 1, 4096]', 'gemv'), X, s('[11008, 4096]', 'wt'), A, s('[1, 1, 11008]', 'out'), note('GEMV')],
            },
            {
              name: 'SiLU(gate)', sub: 'elementwise x·σ(x)',
              col1: [s('[1, S_p, 11008]', 'act'), A, s('[1, S_p, 11008]', 'out')],
              col2: [s('[1, 1, 11008]', 'act'), A, s('[1, 1, 11008]', 'out')],
            },
            {
              name: 'SwiGLU ×', sub: 'SiLU(gate) ⊙ up',
              col1: [s('[1, S_p, 11008]', 'act'), E, s('[1, S_p, 11008]', 'act'), A, s('[1, S_p, 11008]', 'out')],
              col2: [s('[1, 1, 11008]', 'act'), E, s('[1, 1, 11008]', 'act'), A, s('[1, 1, 11008]', 'out')],
            },
            {
              name: 'Down projection', sub: 'down_proj · GEMM',
              col1: [s('[1, S_p, 11008]', 'act'), X, s('[4096, 11008]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
              col2: [s('[1, 1, 11008]', 'gemv'), X, s('[4096, 11008]', 'wt'), A, s('[1, 1, 4096]', 'out'), note('GEMV')],
            },
            {
              name: 'Residual add', sub: 'skip connection',
              col1: [s('[1, S_p, 4096]', 'act'), P, s('[1, S_p, 4096]', 'out'), A, s('[1, S_p, 4096]', 'out')],
              col2: [s('[1, 1, 4096]', 'act'), P, s('[1, 1, 4096]', 'out'), A, s('[1, 1, 4096]', 'out')],
            },
          ],
        },
      ],
    },
    {
      title: '2 · Final RMSNorm',
      tables: [{
        col1Header: '▶ Prefill',
        col1Class: 'phase-prefill',
        col2Header: '▶ Decode',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'RMSNorm', sub: 'model.norm',
            col1: [s('[1, S_p, 4096]', 'act'), X, s('[4096]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
            col2: [s('[1, 1, 4096]', 'gemv'), X, s('[4096]', 'wt'), A, s('[1, 1, 4096]', 'out')],
          },
        ],
      }],
    },
    {
      title: '3 · LM Head',
      tables: [{
        col1Header: '▶ Prefill',
        col1Class: 'phase-prefill',
        col2Header: '▶ Decode',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'lm_head', sub: 'linear, no bias',
            col1: [s('[1, S_p, 4096]', 'act'), X, s('[32000, 4096]', 'wt'), A, s('[1, S_p, 32000]', 'out')],
            col2: [s('[1, 1, 4096]', 'gemv'), X, s('[32000, 4096]', 'wt'), A, s('[1, 1, 32000]', 'out'), note('GEMV')],
          },
          {
            name: 'Next-token slice', sub: 'logits[:, −1, :]',
            col1: [s('[1, S_p, 32000]', 'act'), A, s('[1, 32000]', 'out')],
            col2: [s('[1, 1, 32000]', 'act'), A, s('[1, 32000]', 'out')],
          },
        ],
      }],
    },
  ],
  kvCache: {
    title: 'KV Cache Memory  (fp16, full context S_total = 4096)',
    columns: [
      {
        title: 'Per layer',
        lines: [
          { label: 'K cache shape', value: '[1, 32, 4096, 128]' },
          { label: 'V cache shape', value: '[1, 32, 4096, 128]' },
          { label: 'Elements (K + V)', value: '33,554,432' },
          { label: 'Memory @ fp16', value: '64 MB' },
        ],
      },
      {
        title: 'All 32 layers',
        lines: [
          { label: 'K cache shape', value: '[32, 1, 32, 4096, 128]' },
          { label: 'V cache shape', value: '[32, 1, 32, 4096, 128]' },
          { label: 'Total elements', value: '1,073,741,824' },
          { label: 'Total @ fp16', value: '2 GB' },
        ],
      },
    ],
  },
};
