import { ModelSpec } from '../types';

const s = (shape: string, badge: 'act'|'wt'|'out'|'cache'|'scalar'|'gemv') =>
  ({ kind: 'shape' as const, shape, badge });
const op = (symbol: string) => ({ kind: 'op' as const, symbol });
const note = (text: string) => ({ kind: 'note' as const, text });

const A = op('→'), X = op('×'), P = op('+'), E = op('⊙');

export const falcon7b: ModelSpec = {
  id: 'falcon-7b',
  name: 'Falcon 7B',
  family: 'Falcon',
  category: 'llm',
  paramCount: '7B',
  subtitle: 'Multi-Query Attention (MQA, 1 KV head), parallel attention+MLP, RoPE, LayerNorm. GELU FFN. Fused QKV projection.',
  techniques: ['MQA', 'Parallel Attention+MLP', 'RoPE', 'LayerNorm', 'GELU'],
  hyperparameters: [
    { symbol: 'H', value: '4544', name: 'hidden size' },
    { symbol: 'n_h', value: '71', name: 'query heads' },
    { symbol: 'n_kv', value: '1', name: 'KV heads (MQA — extreme!)' },
    { symbol: 'd_h', value: '64', name: 'head dim = 4544/71' },
    { symbol: 'F', value: '18176', name: 'FFN intermediate (4×H)' },
    { symbol: 'L', value: '32', name: 'decoder layers' },
    { symbol: 'V', value: '65024', name: 'vocab size' },
  ],
  notes: [
    {
      type: 'gemv',
      html: '<strong>MQA extreme case:</strong> Only <strong>1 KV head</strong> (d=64) is shared across all 71 query heads. K and V projections are tiny: [4544→64]. The KV cache is 71× smaller per layer than a comparable MHA model. All 71 query heads attend over the same single K and V sequence.',
    },
    {
      type: 'info',
      html: '<strong>Parallel attention+MLP:</strong> The MLP and attention sub-layers run <em>in parallel</em> from the same LayerNorm output, then their results are added: <code>h = x + Attn(LN(x)) + MLP(LN(x))</code>. A single shared LayerNorm replaces two separate norms. This improves throughput on tensor-parallel hardware.',
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
            name: 'word_embeddings', sub: 'table lookup',
            col1: [s('[1, S_p]', 'act'), X, s('[65024, 4544]', 'wt'), A, s('[1, S_p, 4544]', 'out')],
            col2: [s('[1, 1]', 'gemv'), X, s('[65024, 4544]', 'wt'), A, s('[1, 1, 4544]', 'out')],
          },
        ],
      }],
    },
    {
      title: 'Decoder Layer (parallel attn+MLP)',
      repeat: 32,
      repeatLabel: 'Repeats <span>32 times</span>. Attention and MLP run in parallel from shared norm.',
      tables: [
        {
          col1Header: '▶ Prefill',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'Shared LayerNorm', sub: 'input_layernorm (used by both attn & MLP)',
              col1: [s('[1, S_p, 4544]', 'act'), X, s('[4544]', 'wt'), A, s('[1, S_p, 4544]', 'out')],
              col2: [s('[1, 1, 4544]', 'gemv'), X, s('[4544]', 'wt'), A, s('[1, 1, 4544]', 'out')],
            },
          ],
        },
        {
          col1Header: '▶ Prefill  [ATTENTION BRANCH]',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'Fused QKV proj', sub: 'query_key_value [4544→(4544+64+64)]',
              col1: [s('[1, S_p, 4544]', 'act'), X, s('[4672, 4544]', 'wt'), A, s('[1, S_p, 4672]', 'out'), note('fused · MQA')],
              col2: [s('[1, 1, 4544]', 'gemv'), X, s('[4672, 4544]', 'wt'), A, s('[1, 1, 4672]', 'out'), note('GEMV')],
            },
            {
              name: 'Split Q / K / V', sub: 'Q:[·,4544], K:[·,64], V:[·,64]',
              col1: [s('[1, S_p, 4672]', 'act'), A, s('[1, S_p, 4544]', 'out'), P, s('[1, S_p, 64]', 'act'), P, s('[1, S_p, 64]', 'act')],
              col2: [s('[1, 1, 4672]', 'act'), A, s('[1, 1, 4544]', 'out'), P, s('[1, 1, 64]', 'act'), P, s('[1, 1, 64]', 'act')],
            },
            {
              name: 'Reshape Q', sub: '71 heads of dim 64',
              col1: [s('[1, S_p, 4544]', 'act'), A, s('[1, 71, S_p, 64]', 'out')],
              col2: [s('[1, 1, 4544]', 'act'), A, s('[1, 71, 1, 64]', 'out')],
            },
            {
              name: 'Reshape K (1 head)', sub: 'single KV head',
              col1: [s('[1, S_p, 64]', 'act'), A, s('[1, 1, S_p, 64]', 'out')],
              col2: [s('[1, 1, 64]', 'act'), A, s('[1, 1, 1, 64]', 'out')],
            },
            {
              name: 'Reshape V (1 head)', sub: 'single KV head',
              col1: [s('[1, S_p, 64]', 'act'), A, s('[1, 1, S_p, 64]', 'out')],
              col2: [s('[1, 1, 64]', 'act'), A, s('[1, 1, 1, 64]', 'out')],
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
              name: 'RoPE → Q', sub: 'rotate_half, per head',
              col1: [s('[1, 71, S_p, 64]', 'act'), X, s('[S_p, 32]', 'scalar'), A, s('[1, 71, S_p, 64]', 'out')],
              col2: [s('[1, 71, 1, 64]', 'act'), X, s('[1, 32]', 'scalar'), A, s('[1, 71, 1, 64]', 'out')],
            },
            {
              name: 'RoPE → K', sub: 'rotate_half, single head',
              col1: [s('[1, 1, S_p, 64]', 'act'), X, s('[S_p, 32]', 'scalar'), A, s('[1, 1, S_p, 64]', 'out')],
              col2: [s('[1, 1, 1, 64]', 'act'), X, s('[1, 32]', 'scalar'), A, s('[1, 1, 1, 64]', 'out')],
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
              name: 'MQA KV cache K', sub: 'single head cached',
              col1: [s('[1, 1, 0, 64]', 'cache'), P, s('[1, 1, S_p, 64]', 'act'), A, s('[1, 1, S_p, 64]', 'cache')],
              col2: [s('[1, 1, S_t, 64]', 'cache'), P, s('[1, 1, 1, 64]', 'act'), A, s('[1, 1, S_t+1, 64]', 'cache')],
            },
            {
              name: 'MQA KV cache V', sub: 'single head cached',
              col1: [s('[1, 1, 0, 64]', 'cache'), P, s('[1, 1, S_p, 64]', 'act'), A, s('[1, 1, S_p, 64]', 'cache')],
              col2: [s('[1, 1, S_t, 64]', 'cache'), P, s('[1, 1, 1, 64]', 'act'), A, s('[1, 1, S_t+1, 64]', 'cache')],
            },
            {
              name: 'broadcast K → 71 heads', sub: 'expand: [1,1,S,64]→[1,71,S,64]',
              col1: [s('[1, 1, S_p, 64]', 'cache'), A, s('[1, 71, S_p, 64]', 'out'), note('zero-copy broadcast')],
              col2: [s('[1, 1, S_t, 64]', 'cache'), A, s('[1, 71, S_t, 64]', 'out'), note('zero-copy broadcast')],
            },
            {
              name: 'Q · K\u1d40', sub: 'attn scores, all 71 heads',
              col1: [s('[1, 71, S_p, 64]', 'act'), X, s('[1, 71, 64, S_p]', 'cache'), A, s('[1, 71, S_p, S_p]', 'out')],
              col2: [s('[1, 71, 1, 64]', 'gemv'), X, s('[1, 71, 64, S_t]', 'cache'), A, s('[1, 71, 1, S_t]', 'out')],
            },
            {
              name: 'Scale ÷ √64 + mask', sub: '× 0.125, causal mask',
              col1: [s('[1, 71, S_p, S_p]', 'act'), A, s('[1, 71, S_p, S_p]', 'out')],
              col2: [s('[1, 71, 1, S_t]', 'act'), A, s('[1, 71, 1, S_t]', 'out')],
            },
            {
              name: 'Softmax + Attn · V', sub: 'context vectors, broadcast V',
              col1: [s('[1, 71, S_p, S_p]', 'act'), X, s('[1, 71, S_p, 64]', 'cache'), A, s('[1, 71, S_p, 64]', 'out')],
              col2: [s('[1, 71, 1, S_t]', 'gemv'), X, s('[1, 71, S_t, 64]', 'cache'), A, s('[1, 71, 1, 64]', 'out')],
            },
            {
              name: 'Reshape + dense proj', sub: 'merge → dense [4544→4544]',
              col1: [s('[1, S_p, 4544]', 'act'), X, s('[4544, 4544]', 'wt'), A, s('[1, S_p, 4544]', 'out')],
              col2: [s('[1, 1, 4544]', 'gemv'), X, s('[4544, 4544]', 'wt'), A, s('[1, 1, 4544]', 'out'), note('GEMV')],
            },
          ],
        },
        {
          col1Header: '▶ Prefill  [MLP BRANCH — parallel]',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'dense_h_to_4h', sub: 'FC up [4544→18176]',
              col1: [s('[1, S_p, 4544]', 'act'), X, s('[18176, 4544]', 'wt'), A, s('[1, S_p, 18176]', 'out')],
              col2: [s('[1, 1, 4544]', 'gemv'), X, s('[18176, 4544]', 'wt'), A, s('[1, 1, 18176]', 'out'), note('GEMV')],
            },
            {
              name: 'GELU', sub: 'elementwise activation',
              col1: [s('[1, S_p, 18176]', 'act'), A, s('[1, S_p, 18176]', 'out')],
              col2: [s('[1, 1, 18176]', 'act'), A, s('[1, 1, 18176]', 'out')],
            },
            {
              name: 'dense_4h_to_h', sub: 'FC down [18176→4544]',
              col1: [s('[1, S_p, 18176]', 'act'), X, s('[4544, 18176]', 'wt'), A, s('[1, S_p, 4544]', 'out')],
              col2: [s('[1, 1, 18176]', 'gemv'), X, s('[4544, 18176]', 'wt'), A, s('[1, 1, 4544]', 'out'), note('GEMV')],
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
              name: 'Parallel merge', sub: 'h = x + attn_out + mlp_out',
              col1: [s('[1, S_p, 4544]', 'act'), P, s('[1, S_p, 4544]', 'out'), P, s('[1, S_p, 4544]', 'out'), A, s('[1, S_p, 4544]', 'out')],
              col2: [s('[1, 1, 4544]', 'act'), P, s('[1, 1, 4544]', 'out'), P, s('[1, 1, 4544]', 'out'), A, s('[1, 1, 4544]', 'out')],
            },
          ],
        },
      ],
    },
    {
      title: '2 · Final LayerNorm + LM Head',
      tables: [{
        col1Header: '▶ Prefill',
        col1Class: 'phase-prefill',
        col2Header: '▶ Decode',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'LayerNorm', sub: 'ln_f',
            col1: [s('[1, S_p, 4544]', 'act'), X, s('[4544]', 'wt'), A, s('[1, S_p, 4544]', 'out')],
            col2: [s('[1, 1, 4544]', 'gemv'), X, s('[4544]', 'wt'), A, s('[1, 1, 4544]', 'out')],
          },
          {
            name: 'lm_head', sub: 'linear, no tie',
            col1: [s('[1, S_p, 4544]', 'act'), X, s('[65024, 4544]', 'wt'), A, s('[1, S_p, 65024]', 'out')],
            col2: [s('[1, 1, 4544]', 'gemv'), X, s('[65024, 4544]', 'wt'), A, s('[1, 1, 65024]', 'out'), note('GEMV')],
          },
        ],
      }],
    },
  ],
  kvCache: {
    title: 'KV Cache Memory  (fp16, MQA, 1 KV head)',
    columns: [
      {
        title: 'Per layer (MQA)',
        lines: [
          { label: 'K cache shape', value: '[1, 1, S, 64]' },
          { label: 'V cache shape', value: '[1, 1, S, 64]' },
          { label: 'At S=2048 (fp16)', value: '0.5 MB/layer' },
          { label: 'vs MHA (71 heads)', value: '−98.6% (71× smaller!)' },
        ],
      },
      {
        title: 'All 32 layers (S=2048)',
        lines: [
          { label: 'K cache', value: '[32, 1, 1, 2048, 64]' },
          { label: 'V cache', value: '[32, 1, 1, 2048, 64]' },
          { label: 'Total @ fp16', value: '16 MB  (trivial)' },
        ],
      },
    ],
  },
};
