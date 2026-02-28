import { ModelSpec } from '../types';

const s = (shape: string, badge: 'act'|'wt'|'out'|'cache'|'scalar'|'gemv') =>
  ({ kind: 'shape' as const, shape, badge });
const op = (symbol: string) => ({ kind: 'op' as const, symbol });
const note = (text: string) => ({ kind: 'note' as const, text });

const A = op('→'), X = op('×'), P = op('+');

// PaliGemma 3B: Vision-Language Model
// Vision encoder: SigLIP-So400m/14 (400M params)
//   H_v = 1152, F_v = 4304, L_v = 27, n_h_v = 16, patch = 14
//   At 224px: 16×16 = 256 image tokens
// Linear projection: 1152 → 2048
// LLM decoder: Gemma 1 2B (~2.0B params)
//   H = 2048, n_h = 8, n_kv = 1 (MQA), d_h = 256, F = 16384, L = 18
//   V = 257152 (256000 base + 1024 location + 128 segmentation)
//   GeGLU activation, RoPE
// Total: ~3B params

export const paligemma3b: ModelSpec = {
  id: 'paligemma-3b',
  name: 'PaliGemma 3B',
  family: 'PaliGemma',
  category: 'vlm',
  paramCount: '3B',
  subtitle: 'Vision-language model (VLM). SigLIP-So400m/14 image encoder (400M params) feeds into Gemma 2B decoder (2.0B params) via linear projection. Image tokens concatenated with text tokens. Full attention over image+prompt, causal attention for generation. MQA decoder (8 query / 1 KV head).',
  techniques: ['SigLIP Vision Encoder', 'Linear Projection', 'MQA', 'GeGLU', 'RoPE', 'RMSNorm', 'Vision-Language Fusion'],
  hyperparameters: [
    { symbol: 'H_v', value: '1152', name: 'vision encoder hidden size' },
    { symbol: 'F_v', value: '4304', name: 'vision FFN intermediate' },
    { symbol: 'L_v', value: '27', name: 'vision encoder layers' },
    { symbol: 'n_h_v', value: '16', name: 'vision attention heads' },
    { symbol: 'P', value: '14', name: 'patch size' },
    { symbol: 'N_img', value: '256', name: 'image tokens (224/14)²' },
    { symbol: 'H', value: '2048', name: 'LLM hidden size' },
    { symbol: 'n_h', value: '8', name: 'LLM query heads' },
    { symbol: 'n_kv', value: '1', name: 'LLM KV heads (MQA)' },
    { symbol: 'd_h', value: '256', name: 'head dim' },
    { symbol: 'F', value: '16384', name: 'LLM FFN intermediate (8×H)' },
    { symbol: 'L', value: '18', name: 'LLM decoder layers' },
    { symbol: 'V', value: '257152', name: 'vocab (256K + loc + seg tokens)' },
  ],
  notes: [
    {
      type: 'info',
      html: '<strong>SigLIP-So400m vision encoder:</strong> "Shape-Optimized" ViT (400M params) trained with sigmoid contrastive loss. Processes 224×224 images into 256 patch tokens of 1152 dimensions. A linear projection maps these to the Gemma decoder\'s 2048-dim space.',
    },
    {
      type: 'info',
      html: '<strong>Vision-language fusion:</strong> Image tokens are concatenated with text tokens. During prefill, full bidirectional attention is used over image+prompt tokens. During generation, standard causal attention is used. No cross-attention — the decoder sees image tokens as part of its input sequence.',
    },
    {
      type: 'info',
      html: '<strong>Gemma 2B decoder (MQA):</strong> Uses Multi-Query Attention (1 KV head shared across 8 query heads). The KV cache is 8× smaller than MHA, and the KV projection weights are 8× smaller. GeGLU activation with 8× expansion (F=16384).',
    },
  ],
  sections: [
    {
      title: '0 · SigLIP Vision Encoder',
      tables: [{
        col1Header: '▶ Encoder (image)',
        col1Class: 'phase-prefill',
        col2Header: '▶ Shapes',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Input image', sub: '224×224 RGB',
            col1: [s('[1, 3, 224, 224]', 'act'), A, note('single image')],
            col2: [],
          },
          {
            name: 'Patch embed', sub: 'Conv2D(3→1152, k=14, s=14)',
            col1: [s('[1, 3, 224, 224]', 'act'), X, s('[1152, 3, 14, 14]', 'wt'), A, s('[1, 1152, 16, 16]', 'out')],
            col2: [note('16×16 = 256 patches')],
          },
          {
            name: 'Flatten + pos embed', sub: 'reshape → 256 tokens + learned pos',
            col1: [s('[1, 256, 1152]', 'act'), P, s('[256, 1152]', 'scalar'), A, s('[1, 256, 1152]', 'out')],
            col2: [],
          },
        ],
      }],
    },
    {
      title: 'SigLIP Encoder Block',
      repeat: 27,
      repeatLabel: 'Repeats <span>27 times</span>. Full bidirectional attention (no causal mask). 16 heads, H=1152.',
      tables: [
        {
          col1Header: '▶ Encoder',
          col1Class: 'phase-prefill',
          col2Header: '▶ Shapes',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm', sub: 'pre-attn norm',
              col1: [s('[1, 256, 1152]', 'act'), X, s('[1152]', 'wt'), A, s('[1, 256, 1152]', 'out')],
              col2: [],
            },
            {
              name: 'Q/K/V proj (MHA)', sub: '[1152→1152] each, 16 heads',
              col1: [s('[1, 256, 1152]', 'act'), X, s('[1152, 1152]', 'wt'), A, s('[1, 256, 1152]', 'out'), X, s('(×3)', 'scalar')],
              col2: [s('[1, 16, 256, 72]', 'out'), X, s('(×3)', 'scalar')],
            },
            {
              name: 'Full self-attn + O proj', sub: 'bidirectional, [1152→1152]',
              col1: [s('[1, 16, 256, 256]', 'act'), X, s('[1, 16, 256, 72]', 'act'), A, s('[1, 256, 1152]', 'out')],
              col2: [note('256² attn matrix')],
            },
            {
              name: 'Residual + LN + FFN', sub: 'FC(1152→4304) → GELU → FC(4304→1152) → residual',
              col1: [s('[1, 256, 1152]', 'act'), X, s('[4304, 1152]', 'wt'), A, s('[1, 256, 4304]', 'out')],
              col2: [s('[1, 256, 1152]', 'out')],
            },
          ],
        },
      ],
    },
    {
      title: '1 · Linear Projection (Vision → Language)',
      tables: [{
        col1Header: '▶ Project image tokens',
        col1Class: 'phase-prefill',
        col2Header: '▶ Shape',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Final vision LN', sub: 'encoder output norm',
            col1: [s('[1, 256, 1152]', 'act'), X, s('[1152]', 'wt'), A, s('[1, 256, 1152]', 'out')],
            col2: [],
          },
          {
            name: 'Linear projection', sub: 'proj [1152→2048]',
            col1: [s('[1, 256, 1152]', 'act'), X, s('[2048, 1152]', 'wt'), A, s('[1, 256, 2048]', 'out')],
            col2: [note('matches Gemma 2B input dim')],
          },
          {
            name: 'Concatenate with text', sub: '[img_tokens, BOS, text_tokens]',
            col1: [s('[1, 256, 2048]', 'act'), P, s('[1, T_text, 2048]', 'act'), A, s('[1, 256+T_text, 2048]', 'out')],
            col2: [note('total seq = 256 + text tokens')],
          },
        ],
      }],
    },
    {
      title: 'Gemma 2B Decoder Block',
      repeat: 18,
      repeatLabel: 'Decoder: <span>18 layers</span>. MQA (8 query, 1 KV head). GeGLU activation (8× FFN expansion).',
      tables: [
        {
          col1Header: '▶ Prefill (img + prompt)',
          col1Class: 'phase-prefill',
          col2Header: '▶ Decode (1 token)',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'RMSNorm', sub: 'input_layernorm',
              col1: [s('[1, S, 2048]', 'act'), X, s('[2048]', 'wt'), A, s('[1, S, 2048]', 'out')],
              col2: [s('[1, 1, 2048]', 'act'), X, s('[2048]', 'wt'), A, s('[1, 1, 2048]', 'out')],
            },
            {
              name: 'Q projection', sub: 'q_proj [2048→2048], 8 heads',
              col1: [s('[1, S, 2048]', 'act'), X, s('[2048, 2048]', 'wt'), A, s('[1, S, 2048]', 'out')],
              col2: [s('[1, 1, 2048]', 'gemv'), X, s('[2048, 2048]', 'wt'), A, s('[1, 1, 2048]', 'out'), note('GEMV')],
            },
            {
              name: 'K projection', sub: 'k_proj [2048→256], 1 KV head (MQA)',
              col1: [s('[1, S, 2048]', 'act'), X, s('[256, 2048]', 'wt'), A, s('[1, S, 256]', 'out')],
              col2: [s('[1, 1, 2048]', 'gemv'), X, s('[256, 2048]', 'wt'), A, s('[1, 1, 256]', 'out'), note('GEMV')],
            },
            {
              name: 'V projection', sub: 'v_proj [2048→256], 1 KV head (MQA)',
              col1: [s('[1, S, 2048]', 'act'), X, s('[256, 2048]', 'wt'), A, s('[1, S, 256]', 'out')],
              col2: [s('[1, 1, 2048]', 'gemv'), X, s('[256, 2048]', 'wt'), A, s('[1, 1, 256]', 'out'), note('GEMV')],
            },
            {
              name: 'KV cache + MQA broadcast', sub: 'K/V: 1 head → broadcast to 8',
              col1: [s('[1, 1, S, 256]', 'cache'), A, s('[1, 8, S, 256]', 'out'), note('repeat 8×')],
              col2: [s('[1, 1, S_t, 256]', 'cache'), A, s('[1, 8, S_t, 256]', 'out')],
            },
            {
              name: 'Q · K\u1d40 + softmax + Attn · V', sub: 'full attn (img+prefix) / causal (gen)',
              col1: [s('[1, 8, S, 256]', 'act'), X, s('[1, 8, 256, S]', 'cache'), A, s('[1, 8, S, 256]', 'out')],
              col2: [s('[1, 8, 1, 256]', 'gemv'), X, s('[1, 8, 256, S_t]', 'cache'), A, s('[1, 8, 1, 256]', 'out')],
            },
            {
              name: 'O projection', sub: 'o_proj [2048→2048]',
              col1: [s('[1, S, 2048]', 'act'), X, s('[2048, 2048]', 'wt'), A, s('[1, S, 2048]', 'out')],
              col2: [s('[1, 1, 2048]', 'gemv'), X, s('[2048, 2048]', 'wt'), A, s('[1, 1, 2048]', 'out'), note('GEMV')],
            },
            {
              name: 'Residual add', sub: 'skip connection',
              col1: [s('[1, S, 2048]', 'act'), P, s('[1, S, 2048]', 'out'), A, s('[1, S, 2048]', 'out')],
              col2: [s('[1, 1, 2048]', 'act'), P, s('[1, 1, 2048]', 'out'), A, s('[1, 1, 2048]', 'out')],
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
              name: 'RMSNorm', sub: 'pre-FFN norm',
              col1: [s('[1, S, 2048]', 'act'), X, s('[2048]', 'wt'), A, s('[1, S, 2048]', 'out')],
              col2: [s('[1, 1, 2048]', 'act'), X, s('[2048]', 'wt'), A, s('[1, 1, 2048]', 'out')],
            },
            {
              name: 'Gate + Up proj', sub: 'GeGLU: gate [2048→16384], up [2048→16384]',
              col1: [s('[1, S, 2048]', 'act'), X, s('[16384, 2048]', 'wt'), A, s('[1, S, 16384]', 'out'), X, s('(×2)', 'scalar')],
              col2: [s('[1, 1, 2048]', 'gemv'), X, s('[16384, 2048]', 'wt'), A, s('[1, 1, 16384]', 'out'), note('GEMV ×2')],
            },
            {
              name: 'GELU gate × up', sub: 'GeGLU: GELU(gate) ⊙ up',
              col1: [s('[1, S, 16384]', 'act'), X, s('[1, S, 16384]', 'act'), A, s('[1, S, 16384]', 'out')],
              col2: [s('[1, 1, 16384]', 'act'), X, s('[1, 1, 16384]', 'act'), A, s('[1, 1, 16384]', 'out')],
            },
            {
              name: 'Down proj', sub: 'down_proj [16384→2048]',
              col1: [s('[1, S, 16384]', 'act'), X, s('[2048, 16384]', 'wt'), A, s('[1, S, 2048]', 'out')],
              col2: [s('[1, 1, 16384]', 'gemv'), X, s('[2048, 16384]', 'wt'), A, s('[1, 1, 2048]', 'out'), note('GEMV')],
            },
            {
              name: 'Residual add', sub: 'skip connection',
              col1: [s('[1, S, 2048]', 'act'), P, s('[1, S, 2048]', 'out'), A, s('[1, S, 2048]', 'out')],
              col2: [s('[1, 1, 2048]', 'act'), P, s('[1, 1, 2048]', 'out'), A, s('[1, 1, 2048]', 'out')],
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
            col1: [s('[1, S, 2048]', 'act'), X, s('[2048]', 'wt'), A, s('[1, S, 2048]', 'out')],
            col2: [s('[1, 1, 2048]', 'act'), X, s('[2048]', 'wt'), A, s('[1, 1, 2048]', 'out')],
          },
          {
            name: 'LM head', sub: 'tied embed → logits [2048→257152]',
            col1: [s('[1, S, 2048]', 'act'), X, s('[257152, 2048]', 'wt'), A, s('[1, S, 257152]', 'out')],
            col2: [s('[1, 1, 2048]', 'gemv'), X, s('[257152, 2048]', 'wt'), A, s('[1, 1, 257152]', 'out'), note('GEMV')],
          },
        ],
      }],
    },
  ],
  kvCache: {
    title: 'KV Cache Memory  (fp16)',
    columns: [
      {
        title: 'Per layer (MQA — 1 KV head)',
        lines: [
          { label: 'K cache shape', value: '[1, 1, S_t, 256]' },
          { label: 'V cache shape', value: '[1, 1, S_t, 256]' },
          { label: 'Per layer at S=512 fp16', value: '2 × 1 × 512 × 256 × 2B = 0.5 MB' },
        ],
      },
      {
        title: 'All 18 layers',
        lines: [
          { label: 'Total at S=512 fp16', value: '18 × 0.5 MB = 9.2 MB' },
          { label: 'Total at S=512 INT4', value: '18 × 0.13 MB = 2.3 MB' },
          { label: 'MQA advantage', value: '8× smaller than MHA cache' },
        ],
      },
    ],
  },
};
