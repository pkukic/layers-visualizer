import { ModelSpec } from '../types';

const s = (shape: string, badge: 'act'|'wt'|'out'|'cache'|'scalar'|'gemv') =>
  ({ kind: 'shape' as const, shape, badge });
const op = (symbol: string) => ({ kind: 'op' as const, symbol });
const note = (text: string) => ({ kind: 'note' as const, text });

const A = op('→'), X = op('×'), P = op('+'), E = op('⊙');

export const gemma2_9b: ModelSpec = {
  id: 'gemma2-9b',
  name: 'Gemma 2 (9B)',
  family: 'Gemma',
  category: 'llm',
  paramCount: '9.2B',
  subtitle: 'Alternating local (W=4096) / global (full) attention layers, pre+post RMSNorm, logit soft-capping (tanh), GQA (8 KV heads), GeGLU, RoPE.',
  techniques: ['GQA', 'Alternating Local/Global Attention', 'Logit Soft-Capping', 'GeGLU', 'Pre+Post RMSNorm', 'RoPE'],
  hyperparameters: [
    { symbol: 'H', value: '3584', name: 'hidden size' },
    { symbol: 'n_h', value: '16', name: 'query heads' },
    { symbol: 'n_kv', value: '8', name: 'KV heads (GQA, ratio 2:1)' },
    { symbol: 'd_h', value: '256', name: 'head dim (large!)' },
    { symbol: 'F', value: '14336', name: 'FFN intermediate' },
    { symbol: 'W', value: '4096', name: 'local attn window (even layers)' },
    { symbol: 'L', value: '42', name: 'decoder layers' },
    { symbol: 'V', value: '256128', name: 'vocab size' },
    { symbol: 'cap_a', value: '50.0', name: 'attention logit soft-cap' },
    { symbol: 'cap_f', value: '30.0', name: 'final logit soft-cap' },
  ],
  notes: [
    {
      type: 'info',
      html: '<strong>Alternating attention:</strong> Even layers (0,2,4,…) use <em>local sliding-window attention</em> (W=4096). Odd layers (1,3,5,…) use <em>global (full causal) attention</em>. This halves the local KV cache footprint while preserving long-range dependencies.',
    },
    {
      type: 'info',
      html: '<strong>Logit soft-capping:</strong> Attention scores are scaled by <code>tanh(x / 50.0) × 50.0</code> before softmax, bounding them to (−50, +50). Final logits are capped at ±30.0 to prevent training instability.',
    },
    {
      type: 'info',
      html: '<strong>Pre + Post normalization:</strong> Unlike standard pre-norm, Gemma 2 applies RMSNorm both <em>before</em> (pre-norm) and <em>after</em> (post-norm) each sub-layer. This improves training stability at scale.',
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
            name: 'embed_tokens', sub: 'table lookup × √H',
            col1: [s('[1, S_p]', 'act'), X, s('[256128, 3584]', 'wt'), A, s('[1, S_p, 3584]', 'out')],
            col2: [s('[1, 1]', 'gemv'), X, s('[256128, 3584]', 'wt'), A, s('[1, 1, 3584]', 'out')],
          },
          {
            name: 'Scale embeddings', sub: '× √3584 = 59.87',
            col1: [s('[1, S_p, 3584]', 'act'), X, s('59.87', 'scalar'), A, s('[1, S_p, 3584]', 'out')],
            col2: [s('[1, 1, 3584]', 'act'), X, s('59.87', 'scalar'), A, s('[1, 1, 3584]', 'out')],
          },
        ],
      }],
    },
    {
      title: 'Even Layer (local attention, W=4096)',
      repeat: 21,
      repeatLabel: 'Layers <span>0, 2, 4, … 40</span>  (21 local layers). Attends only to last W=4096 tokens.',
      tables: [
        {
          col1Header: '▶ Prefill',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'Pre-RMSNorm', sub: 'input_layernorm (before attn)',
              col1: [s('[1, S_p, 3584]', 'act'), X, s('[3584]', 'wt'), A, s('[1, S_p, 3584]', 'out')],
              col2: [s('[1, 1, 3584]', 'gemv'), X, s('[3584]', 'wt'), A, s('[1, 1, 3584]', 'out')],
            },
            {
              name: 'Q projection', sub: 'q_proj [3584→4096], 16 heads × 256',
              col1: [s('[1, S_p, 3584]', 'act'), X, s('[4096, 3584]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
              col2: [s('[1, 1, 3584]', 'gemv'), X, s('[4096, 3584]', 'wt'), A, s('[1, 1, 4096]', 'out'), note('GEMV')],
            },
            {
              name: 'K projection', sub: 'k_proj [3584→2048], 8 KV heads × 256',
              col1: [s('[1, S_p, 3584]', 'act'), X, s('[2048, 3584]', 'wt'), A, s('[1, S_p, 2048]', 'out')],
              col2: [s('[1, 1, 3584]', 'gemv'), X, s('[2048, 3584]', 'wt'), A, s('[1, 1, 2048]', 'out'), note('GEMV')],
            },
            {
              name: 'V projection', sub: 'v_proj [3584→2048], 8 KV heads × 256',
              col1: [s('[1, S_p, 3584]', 'act'), X, s('[2048, 3584]', 'wt'), A, s('[1, S_p, 2048]', 'out')],
              col2: [s('[1, 1, 3584]', 'gemv'), X, s('[2048, 3584]', 'wt'), A, s('[1, 1, 2048]', 'out'), note('GEMV')],
            },
            {
              name: 'Reshape + RoPE Q', sub: '[1, 16, S, 256] + rotate_half',
              col1: [s('[1, S_p, 4096]', 'act'), A, s('[1, 16, S_p, 256]', 'out')],
              col2: [s('[1, 1, 4096]', 'act'), A, s('[1, 16, 1, 256]', 'out')],
            },
            {
              name: 'Reshape + RoPE K', sub: '[1, 8, S, 256] + rotate_half',
              col1: [s('[1, S_p, 2048]', 'act'), A, s('[1, 8, S_p, 256]', 'out')],
              col2: [s('[1, 1, 2048]', 'act'), A, s('[1, 8, 1, 256]', 'out')],
            },
          ],
        },
        {
          col1Header: '▶ Prefill (local cache)',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode (local cache)',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'Local KV cache K', sub: 'rolling window W=4096',
              col1: [s('[1, 8, 0, 256]', 'cache'), P, s('[1, 8, S_p, 256]', 'act'), A, s('[1, 8, min(S_p,W), 256]', 'cache')],
              col2: [s('[1, 8, S_w, 256]', 'cache'), P, s('[1, 8, 1, 256]', 'act'), A, s('[1, 8, min(S_w+1,W), 256]', 'cache')],
            },
            {
              name: 'Local KV cache V', sub: 'rolling window W=4096',
              col1: [s('[1, 8, 0, 256]', 'cache'), P, s('[1, 8, S_p, 256]', 'act'), A, s('[1, 8, min(S_p,W), 256]', 'cache')],
              col2: [s('[1, 8, S_w, 256]', 'cache'), P, s('[1, 8, 1, 256]', 'act'), A, s('[1, 8, min(S_w+1,W), 256]', 'cache')],
            },
            {
              name: 'repeat_interleave K', sub: '8 KV → 16 Q heads (×2)',
              col1: [s('[1, 8, S_w, 256]', 'cache'), A, s('[1, 16, S_w, 256]', 'out')],
              col2: [s('[1, 8, S_w, 256]', 'cache'), A, s('[1, 16, S_w, 256]', 'out')],
            },
            {
              name: 'Q · K\u1d40', sub: 'local attn scores',
              col1: [s('[1, 16, S_p, 256]', 'act'), X, s('[1, 16, 256, S_w]', 'cache'), A, s('[1, 16, S_p, S_w]', 'out')],
              col2: [s('[1, 16, 1, 256]', 'gemv'), X, s('[1, 16, 256, S_w]', 'cache'), A, s('[1, 16, 1, S_w]', 'out')],
            },
            {
              name: 'Soft-cap ÷ √256', sub: 'tanh(x/50) × 50, ÷ √256',
              col1: [s('[1, 16, S_p, S_w]', 'act'), A, s('[1, 16, S_p, S_w]', 'out'), note('tanh cap')],
              col2: [s('[1, 16, 1, S_w]', 'act'), A, s('[1, 16, 1, S_w]', 'out'), note('tanh cap')],
            },
            {
              name: 'Softmax + Attn · V', sub: 'merge heads → o_proj',
              col1: [s('[1, 16, S_p, S_w]', 'act'), X, s('[1, 16, S_w, 256]', 'cache'), A, s('[1, 16, S_p, 256]', 'out')],
              col2: [s('[1, 16, 1, S_w]', 'gemv'), X, s('[1, 16, S_w, 256]', 'cache'), A, s('[1, 16, 1, 256]', 'out')],
            },
            {
              name: 'O projection', sub: 'o_proj [4096→3584]',
              col1: [s('[1, S_p, 4096]', 'act'), X, s('[3584, 4096]', 'wt'), A, s('[1, S_p, 3584]', 'out')],
              col2: [s('[1, 1, 4096]', 'gemv'), X, s('[3584, 4096]', 'wt'), A, s('[1, 1, 3584]', 'out'), note('GEMV')],
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
              name: 'Post-RMSNorm (attn)', sub: 'post_attention_layernorm',
              col1: [s('[1, S_p, 3584]', 'act'), X, s('[3584]', 'wt'), A, s('[1, S_p, 3584]', 'out')],
              col2: [s('[1, 1, 3584]', 'gemv'), X, s('[3584]', 'wt'), A, s('[1, 1, 3584]', 'out')],
            },
            {
              name: 'Residual add', sub: 'h = x + attn_out',
              col1: [s('[1, S_p, 3584]', 'act'), P, s('[1, S_p, 3584]', 'out'), A, s('[1, S_p, 3584]', 'out')],
              col2: [s('[1, 1, 3584]', 'act'), P, s('[1, 1, 3584]', 'out'), A, s('[1, 1, 3584]', 'out')],
            },
            {
              name: 'Pre-RMSNorm (FFN)', sub: 'pre_feedforward_layernorm',
              col1: [s('[1, S_p, 3584]', 'act'), X, s('[3584]', 'wt'), A, s('[1, S_p, 3584]', 'out')],
              col2: [s('[1, 1, 3584]', 'gemv'), X, s('[3584]', 'wt'), A, s('[1, 1, 3584]', 'out')],
            },
            {
              name: 'Gate proj', sub: 'gate_proj [3584→14336]',
              col1: [s('[1, S_p, 3584]', 'act'), X, s('[14336, 3584]', 'wt'), A, s('[1, S_p, 14336]', 'out')],
              col2: [s('[1, 1, 3584]', 'gemv'), X, s('[14336, 3584]', 'wt'), A, s('[1, 1, 14336]', 'out'), note('GEMV')],
            },
            {
              name: 'Up proj', sub: 'up_proj [3584→14336]',
              col1: [s('[1, S_p, 3584]', 'act'), X, s('[14336, 3584]', 'wt'), A, s('[1, S_p, 14336]', 'out')],
              col2: [s('[1, 1, 3584]', 'gemv'), X, s('[14336, 3584]', 'wt'), A, s('[1, 1, 14336]', 'out'), note('GEMV')],
            },
            {
              name: 'GeGLU', sub: 'GELU(gate) ⊙ up',
              col1: [s('[1, S_p, 14336]', 'act'), E, s('[1, S_p, 14336]', 'act'), A, s('[1, S_p, 14336]', 'out')],
              col2: [s('[1, 1, 14336]', 'act'), E, s('[1, 1, 14336]', 'act'), A, s('[1, 1, 14336]', 'out')],
            },
            {
              name: 'Down proj', sub: 'down_proj [14336→3584]',
              col1: [s('[1, S_p, 14336]', 'act'), X, s('[3584, 14336]', 'wt'), A, s('[1, S_p, 3584]', 'out')],
              col2: [s('[1, 1, 14336]', 'gemv'), X, s('[3584, 14336]', 'wt'), A, s('[1, 1, 3584]', 'out'), note('GEMV')],
            },
            {
              name: 'Post-RMSNorm (FFN)', sub: 'post_feedforward_layernorm',
              col1: [s('[1, S_p, 3584]', 'act'), X, s('[3584]', 'wt'), A, s('[1, S_p, 3584]', 'out')],
              col2: [s('[1, 1, 3584]', 'gemv'), X, s('[3584]', 'wt'), A, s('[1, 1, 3584]', 'out')],
            },
            {
              name: 'Residual add', sub: 'h = h + ffn_out',
              col1: [s('[1, S_p, 3584]', 'act'), P, s('[1, S_p, 3584]', 'out'), A, s('[1, S_p, 3584]', 'out')],
              col2: [s('[1, 1, 3584]', 'act'), P, s('[1, 1, 3584]', 'out'), A, s('[1, 1, 3584]', 'out')],
            },
          ],
        },
      ],
    },
    {
      title: 'Odd Layer (global / full attention)',
      repeat: 21,
      repeatLabel: 'Layers <span>1, 3, 5, … 41</span>  (21 global layers). Attends to full context. Same as even layer but KV cache is unbounded.',
      note: 'Identical structure to even layers except: (1) no sliding-window limit on KV cache — the full context [1, 8, S_total, 256] is attended over; (2) same soft-capping and GQA apply.',
      tables: [
        {
          col1Header: '▶ Prefill',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'Q/K/V projections', sub: 'same as even layer',
              col1: [s('[1, S_p, 3584]', 'act'), X, s('[4096/2048, 3584]', 'wt'), A, s('[1, S_p, ·]', 'out')],
              col2: [s('[1, 1, 3584]', 'gemv'), X, s('[4096/2048, 3584]', 'wt'), A, s('[1, 1, ·]', 'out'), note('GEMV')],
            },
            {
              name: 'Global KV cache K', sub: 'full context (no window limit)',
              col1: [s('[1, 8, 0, 256]', 'cache'), P, s('[1, 8, S_p, 256]', 'act'), A, s('[1, 8, S_p, 256]', 'cache')],
              col2: [s('[1, 8, S_t, 256]', 'cache'), P, s('[1, 8, 1, 256]', 'act'), A, s('[1, 8, S_t+1, 256]', 'cache')],
            },
            {
              name: 'Q · K\u1d40 (full)', sub: 'full causal attn + soft-cap',
              col1: [s('[1, 16, S_p, 256]', 'act'), X, s('[1, 16, 256, S_p]', 'cache'), A, s('[1, 16, S_p, S_p]', 'out'), note('tanh cap')],
              col2: [s('[1, 16, 1, 256]', 'gemv'), X, s('[1, 16, 256, S_t]', 'cache'), A, s('[1, 16, 1, S_t]', 'out'), note('tanh cap')],
            },
            {
              name: 'GeGLU FFN + norms', sub: 'identical to even layer',
              col1: [s('[1, S_p, 3584]', 'act'), A, s('[1, S_p, 3584]', 'out'), note('pre+post norm')],
              col2: [s('[1, 1, 3584]', 'act'), A, s('[1, 1, 3584]', 'out'), note('pre+post norm')],
            },
          ],
        },
      ],
    },
    {
      title: '3 · Final RMSNorm + LM Head',
      tables: [{
        col1Header: '▶ Prefill',
        col1Class: 'phase-prefill',
        col2Header: '▶ Decode',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Final RMSNorm', sub: 'model.norm',
            col1: [s('[1, S_p, 3584]', 'act'), X, s('[3584]', 'wt'), A, s('[1, S_p, 3584]', 'out')],
            col2: [s('[1, 1, 3584]', 'gemv'), X, s('[3584]', 'wt'), A, s('[1, 1, 3584]', 'out')],
          },
          {
            name: 'lm_head', sub: 'tied to embed_tokens',
            col1: [s('[1, S_p, 3584]', 'act'), X, s('[256128, 3584]', 'wt'), A, s('[1, S_p, 256128]', 'out')],
            col2: [s('[1, 1, 3584]', 'gemv'), X, s('[256128, 3584]', 'wt'), A, s('[1, 1, 256128]', 'out'), note('GEMV · huge V')],
          },
          {
            name: 'Final logit soft-cap', sub: 'tanh(x / 30.0) × 30.0',
            col1: [s('[1, S_p, 256128]', 'act'), A, s('[1, S_p, 256128]', 'out'), note('tanh cap ±30')],
            col2: [s('[1, 1, 256128]', 'act'), A, s('[1, 1, 256128]', 'out'), note('tanh cap ±30')],
          },
        ],
      }],
    },
  ],
  kvCache: {
    title: 'KV Cache Memory  (fp16, S_total up to context limit)',
    columns: [
      {
        title: 'Local layer (per layer)',
        lines: [
          { label: 'K cache shape', value: '[1, 8, 4096, 256]' },
          { label: 'V cache shape', value: '[1, 8, 4096, 256]' },
          { label: 'Memory @ fp16', value: '32 MB' },
        ],
      },
      {
        title: 'Global layer (S=8192)',
        lines: [
          { label: 'K cache shape', value: '[1, 8, 8192, 256]' },
          { label: 'V cache shape', value: '[1, 8, 8192, 256]' },
          { label: 'Memory @ fp16', value: '64 MB' },
        ],
      },
      {
        title: 'All 42 layers',
        lines: [
          { label: 'Local (21 layers)', value: '672 MB' },
          { label: 'Global (21 layers)', value: '1,344 MB' },
          { label: 'Total @ fp16 (S=8192)', value: '~2 GB' },
        ],
      },
    ],
  },
};
