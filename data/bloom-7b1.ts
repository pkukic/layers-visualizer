import { ModelSpec } from '../types';

const s = (shape: string, badge: 'act'|'wt'|'out'|'cache'|'scalar'|'gemv') =>
  ({ kind: 'shape' as const, shape, badge });
const op = (symbol: string) => ({ kind: 'op' as const, symbol });
const note = (text: string) => ({ kind: 'note' as const, text });

const A = op('→'), X = op('×'), P = op('+');

export const bloom7b1: ModelSpec = {
  id: 'bloom-7b1',
  name: 'BLOOM 7B1',
  family: 'BLOOM',
  category: 'llm',
  paramCount: '7.1B',
  subtitle: 'ALiBi positional encoding (no position embeddings), MHA, LayerNorm with embedding norm, GELU FFN. Multilingual training (46 languages).',
  techniques: ['ALiBi', 'MHA', 'LayerNorm', 'GELU', 'Embedding Norm'],
  hyperparameters: [
    { symbol: 'H', value: '4096', name: 'hidden size' },
    { symbol: 'n_h', value: '32', name: 'attention heads' },
    { symbol: 'd_h', value: '128', name: 'head dim = H/n_h' },
    { symbol: 'n_kv', value: '32', name: 'KV heads (MHA)' },
    { symbol: 'F', value: '16384', name: 'FFN intermediate (4×H)' },
    { symbol: 'L', value: '30', name: 'decoder layers' },
    { symbol: 'V', value: '250880', name: 'vocab size (multilingual)' },
  ],
  notes: [
    {
      type: 'info',
      html: '<strong>ALiBi (Attention with Linear Biases):</strong> No position embeddings are added to token embeddings. Instead, a per-head linear bias <code>m · (i−j)</code> is subtracted from attention scores, where m is a head-specific slope and (i−j) is the query-key distance. This allows extrapolation to longer sequences than trained on.',
    },
    {
      type: 'info',
      html: '<strong>Embedding LayerNorm:</strong> Unlike most LLMs, BLOOM applies LayerNorm immediately after the token embedding (before the decoder stack). This stabilises training on the large multilingual vocabulary.',
    },
  ],
  sections: [
    {
      title: '0 · Token Embedding + Embedding Norm',
      tables: [{
        col1Header: '▶ Prefill  (S = S_p)',
        col1Class: 'phase-prefill',
        col2Header: '▶ Decode  (S = 1, step t)',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'word_embeddings', sub: 'table lookup (no pos embedding)',
            col1: [s('[1, S_p]', 'act'), X, s('[250880, 4096]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
            col2: [s('[1, 1]', 'gemv'), X, s('[250880, 4096]', 'wt'), A, s('[1, 1, 4096]', 'out')],
          },
          {
            name: 'word_embeddings_layernorm', sub: 'LayerNorm on embeddings (unique!)',
            col1: [s('[1, S_p, 4096]', 'act'), X, s('[4096]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
            col2: [s('[1, 1, 4096]', 'gemv'), X, s('[4096]', 'wt'), A, s('[1, 1, 4096]', 'out')],
          },
        ],
      }],
    },
    {
      title: 'Decoder Layer',
      repeat: 30,
      repeatLabel: 'Repeats <span>30 times</span>. ALiBi bias is unique per head but fixed — computed once and reused.',
      tables: [
        {
          col1Header: '▶ Prefill',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm', sub: 'input_layernorm (pre-attn)',
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
              name: 'Fused QKV', sub: 'query_key_value [4096→12288]',
              col1: [s('[1, S_p, 4096]', 'act'), X, s('[12288, 4096]', 'wt'), A, s('[1, S_p, 12288]', 'out'), note('fused Q,K,V')],
              col2: [s('[1, 1, 4096]', 'gemv'), X, s('[12288, 4096]', 'wt'), A, s('[1, 1, 12288]', 'out'), note('GEMV')],
            },
            {
              name: 'Reshape Q/K/V', sub: 'split 32 heads each',
              col1: [s('[1, S_p, 12288]', 'act'), A, s('[1, 32, S_p, 128]', 'out'), X, s('(×3)', 'scalar')],
              col2: [s('[1, 1, 12288]', 'act'), A, s('[1, 32, 1, 128]', 'out'), X, s('(×3)', 'scalar')],
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
              name: 'KV cache K', sub: 'concat along seq dim',
              col1: [s('[1, 32, 0, 128]', 'cache'), P, s('[1, 32, S_p, 128]', 'act'), A, s('[1, 32, S_p, 128]', 'cache')],
              col2: [s('[1, 32, S_t, 128]', 'cache'), P, s('[1, 32, 1, 128]', 'act'), A, s('[1, 32, S_t+1, 128]', 'cache')],
            },
            {
              name: 'KV cache V', sub: 'concat along seq dim',
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
              col2: [s('[1, 32, 1, 128]', 'gemv'), X, s('[1, 32, 128, S_t]', 'cache'), A, s('[1, 32, 1, S_t]', 'out')],
            },
            {
              name: 'Scale ÷ √128', sub: '× 0.0884',
              col1: [s('[1, 32, S_p, S_p]', 'act'), X, s('0.0884', 'scalar'), A, s('[1, 32, S_p, S_p]', 'out')],
              col2: [s('[1, 32, 1, S_t]', 'act'), X, s('0.0884', 'scalar'), A, s('[1, 32, 1, S_t]', 'out')],
            },
            {
              name: 'ALiBi bias', sub: '− m_h · |i−j| (no pos embedding!)',
              col1: [s('[1, 32, S_p, S_p]', 'act'), P, s('[1, 32, S_p, S_p]', 'scalar'), A, s('[1, 32, S_p, S_p]', 'out'), note('linear bias')],
              col2: [s('[1, 32, 1, S_t]', 'act'), P, s('[1, 32, 1, S_t]', 'scalar'), A, s('[1, 32, 1, S_t]', 'out'), note('linear bias')],
            },
            {
              name: 'Causal mask', sub: 'additive (0 or −∞)',
              col1: [s('[1, 32, S_p, S_p]', 'act'), P, s('[1, 1, S_p, S_p]', 'scalar'), A, s('[1, 32, S_p, S_p]', 'out')],
              col2: [s('[1, 32, 1, S_t]', 'act'), P, s('[1, 1, 1, S_t]', 'scalar'), A, s('[1, 32, 1, S_t]', 'out')],
            },
            {
              name: 'Softmax + Attn · V', sub: 'context vectors',
              col1: [s('[1, 32, S_p, S_p]', 'act'), X, s('[1, 32, S_p, 128]', 'cache'), A, s('[1, 32, S_p, 128]', 'out')],
              col2: [s('[1, 32, 1, S_t]', 'gemv'), X, s('[1, 32, S_t, 128]', 'cache'), A, s('[1, 32, 1, 128]', 'out')],
            },
            {
              name: 'Reshape + dense', sub: 'merge → dense [4096→4096]',
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
              name: 'LayerNorm', sub: 'post_attention_layernorm',
              col1: [s('[1, S_p, 4096]', 'act'), X, s('[4096]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
              col2: [s('[1, 1, 4096]', 'gemv'), X, s('[4096]', 'wt'), A, s('[1, 1, 4096]', 'out')],
            },
            {
              name: 'dense_h_to_4h', sub: 'FFN up [4096→16384]',
              col1: [s('[1, S_p, 4096]', 'act'), X, s('[16384, 4096]', 'wt'), A, s('[1, S_p, 16384]', 'out')],
              col2: [s('[1, 1, 4096]', 'gemv'), X, s('[16384, 4096]', 'wt'), A, s('[1, 1, 16384]', 'out'), note('GEMV')],
            },
            {
              name: 'GELU', sub: 'elementwise activation',
              col1: [s('[1, S_p, 16384]', 'act'), A, s('[1, S_p, 16384]', 'out')],
              col2: [s('[1, 1, 16384]', 'act'), A, s('[1, 1, 16384]', 'out')],
            },
            {
              name: 'dense_4h_to_h', sub: 'FFN down [16384→4096]',
              col1: [s('[1, S_p, 16384]', 'act'), X, s('[4096, 16384]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
              col2: [s('[1, 1, 16384]', 'gemv'), X, s('[4096, 16384]', 'wt'), A, s('[1, 1, 4096]', 'out'), note('GEMV')],
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
      title: '2 · Final LayerNorm + LM Head',
      tables: [{
        col1Header: '▶ Prefill',
        col1Class: 'phase-prefill',
        col2Header: '▶ Decode',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'LayerNorm', sub: 'ln_f',
            col1: [s('[1, S_p, 4096]', 'act'), X, s('[4096]', 'wt'), A, s('[1, S_p, 4096]', 'out')],
            col2: [s('[1, 1, 4096]', 'gemv'), X, s('[4096]', 'wt'), A, s('[1, 1, 4096]', 'out')],
          },
          {
            name: 'lm_head', sub: 'tied to word_embeddings',
            col1: [s('[1, S_p, 4096]', 'act'), X, s('[250880, 4096]', 'wt'), A, s('[1, S_p, 250880]', 'out')],
            col2: [s('[1, 1, 4096]', 'gemv'), X, s('[250880, 4096]', 'wt'), A, s('[1, 1, 250880]', 'out'), note('GEMV · large V')],
          },
        ],
      }],
    },
  ],
  kvCache: {
    title: 'KV Cache Memory  (fp16, S_total = 2048)',
    columns: [
      {
        title: 'Per layer',
        lines: [
          { label: 'K cache shape', value: '[1, 32, 2048, 128]' },
          { label: 'V cache shape', value: '[1, 32, 2048, 128]' },
          { label: 'Memory @ fp16', value: '32 MB' },
        ],
      },
      {
        title: 'All 30 layers',
        lines: [
          { label: 'Total @ fp16', value: '960 MB' },
          { label: 'ALiBi slopes (fixed)', value: '[32] floats — negligible' },
          { label: 'ALiBi bias matrix', value: '[1, 32, S, S] — 2048²×32 fp16 = 256 MB' },
        ],
      },
    ],
  },
};
