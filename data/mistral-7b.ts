import { ModelSpec } from '../types';

const s = (shape: string, badge: 'act'|'wt'|'out'|'cache'|'scalar'|'gemv') =>
  ({ kind: 'shape' as const, shape, badge });
const op = (symbol: string) => ({ kind: 'op' as const, symbol });
const note = (text: string) => ({ kind: 'note' as const, text });

const A = op('→'), X = op('×'), P = op('+'), E = op('⊙');

export const mistral7b: ModelSpec = {
  id: 'mistral-7b',
  name: 'Mistral 7B',
  family: 'Mistral',
  category: 'llm',
  paramCount: '7.3B',
  subtitle: 'Decoder-only with Grouped Query Attention (GQA, 8 KV heads) and Sliding Window Attention (SWA, W=4096). RoPE, RMSNorm, SwiGLU.',
  techniques: ['GQA', 'Sliding Window Attention', 'RoPE', 'RMSNorm', 'SwiGLU'],
  hyperparameters: [
    { symbol: 'H', value: '4096', name: 'hidden size' },
    { symbol: 'n_h', value: '32', name: 'query heads' },
    { symbol: 'n_kv', value: '8', name: 'KV heads (GQA, ratio 4:1)' },
    { symbol: 'd_h', value: '128', name: 'head dim = H/n_h' },
    { symbol: 'F', value: '14336', name: 'FFN intermediate' },
    { symbol: 'W', value: '4096', name: 'sliding window size' },
    { symbol: 'L', value: '32', name: 'decoder layers' },
    { symbol: 'V', value: '32000', name: 'vocab size' },
  ],
  notes: [
    {
      type: 'gemv',
      html: '<strong>GQA shrinks KV projections 4×.</strong> K and V projections are [4096→1024] (8 heads × 128) instead of [4096→4096]. Each of the 8 KV heads is shared across 4 query heads via repeat_interleave. This reduces KV cache memory 4× vs MHA.',
    },
    {
      type: 'info',
      html: '<strong>Sliding Window Attention (SWA):</strong> Each token attends only to the W=4096 most recent tokens. The KV cache per layer holds at most W=4096 tokens regardless of sequence length, making memory constant beyond W.',
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
      repeatLabel: 'Repeats <span>32 times</span>. All layers identical; SWA applies uniformly.',
      tables: [
        {
          col1Header: '▶ Prefill',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode',
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
              name: 'Q projection', sub: 'q_proj [4096→4096], 32 heads',
              col1: [s('[1, S_p, 4096]', 'act'), X, s('[4096, 4096]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
              col2: [s('[1, 1, 4096]', 'gemv'), X, s('[4096, 4096]', 'wt'), A, s('[1, 1, 4096]', 'out'), note('GEMV')],
            },
            {
              name: 'K projection', sub: 'k_proj [4096→1024], 8 KV heads',
              col1: [s('[1, S_p, 4096]', 'act'), X, s('[1024, 4096]', 'wt'), A, s('[1, S_p, 1024]', 'out')],
              col2: [s('[1, 1, 4096]', 'gemv'), X, s('[1024, 4096]', 'wt'), A, s('[1, 1, 1024]', 'out'), note('GEMV · 4× smaller')],
            },
            {
              name: 'V projection', sub: 'v_proj [4096→1024], 8 KV heads',
              col1: [s('[1, S_p, 4096]', 'act'), X, s('[1024, 4096]', 'wt'), A, s('[1, S_p, 1024]', 'out')],
              col2: [s('[1, 1, 4096]', 'gemv'), X, s('[1024, 4096]', 'wt'), A, s('[1, 1, 1024]', 'out'), note('GEMV · 4× smaller')],
            },
            {
              name: 'Reshape Q', sub: 'view + RoPE heads',
              col1: [s('[1, S_p, 4096]', 'act'), A, s('[1, 32, S_p, 128]', 'out')],
              col2: [s('[1, 1, 4096]', 'act'), A, s('[1, 32, 1, 128]', 'out')],
            },
            {
              name: 'Reshape K', sub: 'view: 8 KV heads',
              col1: [s('[1, S_p, 1024]', 'act'), A, s('[1, 8, S_p, 128]', 'out')],
              col2: [s('[1, 1, 1024]', 'act'), A, s('[1, 8, 1, 128]', 'out')],
            },
            {
              name: 'Reshape V', sub: 'view: 8 KV heads',
              col1: [s('[1, S_p, 1024]', 'act'), A, s('[1, 8, S_p, 128]', 'out')],
              col2: [s('[1, 1, 1024]', 'act'), A, s('[1, 8, 1, 128]', 'out')],
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
              name: 'RoPE → Q', sub: 'rotate_half on Q',
              col1: [s('[1, 32, S_p, 128]', 'act'), X, s('[S_p, 64]', 'scalar'), A, s('[1, 32, S_p, 128]', 'out')],
              col2: [s('[1, 32, 1, 128]', 'act'), X, s('[1, 64]', 'scalar'), A, s('[1, 32, 1, 128]', 'out')],
            },
            {
              name: 'RoPE → K', sub: 'rotate_half on K',
              col1: [s('[1, 8, S_p, 128]', 'act'), X, s('[S_p, 64]', 'scalar'), A, s('[1, 8, S_p, 128]', 'out')],
              col2: [s('[1, 8, 1, 128]', 'act'), X, s('[1, 64]', 'scalar'), A, s('[1, 8, 1, 128]', 'out')],
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
              name: 'SWA KV cache K', sub: 'rolling window W=4096',
              col1: [s('[1, 8, 0, 128]', 'cache'), P, s('[1, 8, S_p, 128]', 'act'), A, s('[1, 8, min(S_p,W), 128]', 'cache')],
              col2: [s('[1, 8, S_w, 128]', 'cache'), P, s('[1, 8, 1, 128]', 'act'), A, s('[1, 8, min(S_w+1,W), 128]', 'cache')],
            },
            {
              name: 'SWA KV cache V', sub: 'rolling window W=4096',
              col1: [s('[1, 8, 0, 128]', 'cache'), P, s('[1, 8, S_p, 128]', 'act'), A, s('[1, 8, min(S_p,W), 128]', 'cache')],
              col2: [s('[1, 8, S_w, 128]', 'cache'), P, s('[1, 8, 1, 128]', 'act'), A, s('[1, 8, min(S_w+1,W), 128]', 'cache')],
            },
            {
              name: 'repeat_interleave K', sub: '8 KV → 32 Q heads (×4)',
              col1: [s('[1, 8, S_p, 128]', 'cache'), A, s('[1, 32, S_p, 128]', 'out'), note('no copy, stride trick')],
              col2: [s('[1, 8, S_w, 128]', 'cache'), A, s('[1, 32, S_w, 128]', 'out'), note('no copy, stride trick')],
            },
            {
              name: 'repeat_interleave V', sub: '8 KV → 32 Q heads (×4)',
              col1: [s('[1, 8, S_p, 128]', 'cache'), A, s('[1, 32, S_p, 128]', 'out')],
              col2: [s('[1, 8, S_w, 128]', 'cache'), A, s('[1, 32, S_w, 128]', 'out')],
            },
          ],
        },
        {
          col1Header: '▶ Prefill (SWA)',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'Q · K\u1d40', sub: 'sliding-window attn scores',
              col1: [s('[1, 32, S_p, 128]', 'act'), X, s('[1, 32, 128, S_p]', 'cache'), A, s('[1, 32, S_p, S_p]', 'out'), note('SWA block-sparse')],
              col2: [s('[1, 32, 1, 128]', 'gemv'), X, s('[1, 32, 128, S_w]', 'cache'), A, s('[1, 32, 1, S_w]', 'out')],
            },
            {
              name: 'Scale + SWA mask', sub: '÷√128, mask beyond window',
              col1: [s('[1, 32, S_p, S_p]', 'act'), X, s('0.0884', 'scalar'), A, s('[1, 32, S_p, S_p]', 'out')],
              col2: [s('[1, 32, 1, S_w]', 'act'), X, s('0.0884', 'scalar'), A, s('[1, 32, 1, S_w]', 'out')],
            },
            {
              name: 'Softmax', sub: 'dim = −1',
              col1: [s('[1, 32, S_p, S_p]', 'act'), A, s('[1, 32, S_p, S_p]', 'out')],
              col2: [s('[1, 32, 1, S_w]', 'act'), A, s('[1, 32, 1, S_w]', 'out')],
            },
            {
              name: 'Attn · V', sub: 'context vectors',
              col1: [s('[1, 32, S_p, S_p]', 'act'), X, s('[1, 32, S_p, 128]', 'cache'), A, s('[1, 32, S_p, 128]', 'out')],
              col2: [s('[1, 32, 1, S_w]', 'gemv'), X, s('[1, 32, S_w, 128]', 'cache'), A, s('[1, 32, 1, 128]', 'out')],
            },
            {
              name: 'Reshape + O proj', sub: 'merge heads → o_proj',
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
              name: 'Gate + Up proj', sub: 'SwiGLU (14336 intermediate)',
              col1: [s('[1, S_p, 4096]', 'act'), X, s('[14336, 4096]', 'wt'), A, s('[1, S_p, 14336]', 'out'), note('×2 for gate/up')],
              col2: [s('[1, 1, 4096]', 'gemv'), X, s('[14336, 4096]', 'wt'), A, s('[1, 1, 14336]', 'out'), note('GEMV')],
            },
            {
              name: 'SiLU(gate) ⊙ up', sub: 'elementwise gated activation',
              col1: [s('[1, S_p, 14336]', 'act'), E, s('[1, S_p, 14336]', 'act'), A, s('[1, S_p, 14336]', 'out')],
              col2: [s('[1, 1, 14336]', 'act'), E, s('[1, 1, 14336]', 'act'), A, s('[1, 1, 14336]', 'out')],
            },
            {
              name: 'Down projection', sub: 'down_proj · GEMM',
              col1: [s('[1, S_p, 14336]', 'act'), X, s('[4096, 14336]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
              col2: [s('[1, 1, 14336]', 'gemv'), X, s('[4096, 14336]', 'wt'), A, s('[1, 1, 4096]', 'out'), note('GEMV')],
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
      title: '2 · Final RMSNorm + LM Head',
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
          {
            name: 'lm_head', sub: 'linear, no bias',
            col1: [s('[1, S_p, 4096]', 'act'), X, s('[32000, 4096]', 'wt'), A, s('[1, S_p, 32000]', 'out')],
            col2: [s('[1, 1, 4096]', 'gemv'), X, s('[32000, 4096]', 'wt'), A, s('[1, 1, 32000]', 'out'), note('GEMV')],
          },
        ],
      }],
    },
  ],
  kvCache: {
    title: 'KV Cache Memory  (fp16, SWA window W=4096)',
    columns: [
      {
        title: 'Per layer (GQA+SWA)',
        lines: [
          { label: 'K cache shape', value: '[1, 8, 4096, 128]' },
          { label: 'V cache shape', value: '[1, 8, 4096, 128]' },
          { label: 'Elements (K + V)', value: '8,388,608' },
          { label: 'Memory @ fp16', value: '16 MB' },
        ],
      },
      {
        title: 'All 32 layers',
        lines: [
          { label: 'Total K+V elements', value: '268,435,456' },
          { label: 'Total @ fp16', value: '512 MB' },
          { label: 'vs MHA equivalent', value: '−75% (4× GQA)' },
          { label: 'Bounded by', value: 'W=4096 (SWA)' },
        ],
      },
    ],
  },
};
