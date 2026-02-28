import { ModelSpec } from '../types';

const s = (shape: string, badge: 'act'|'wt'|'out'|'cache'|'scalar'|'gemv') =>
  ({ kind: 'shape' as const, shape, badge });
const op = (symbol: string) => ({ kind: 'op' as const, symbol });
const note = (text: string) => ({ kind: 'note' as const, text });

const A = op('→'), X = op('×'), P = op('+');

// ViT-Huge/14: 632M params
// H = 1280, n_h = 16, d_h = 80, F = 5120 (4×H)
// L = 32 encoder layers
// Patch size 14 → 16×16 = 256 patches → T = 257 with CLS

export const vitHuge: ModelSpec = {
  id: 'vit-huge',
  name: 'ViT-Huge/14',
  family: 'ViT',
  category: 'vision',
  paramCount: '632M',
  subtitle: 'Encoder-only vision transformer (Huge variant). Image split into 14×14 patches → sequence of 256+1 tokens. Isotropic depth (32 layers), MHA with 16 heads (d_h=80), LayerNorm, GELU FFN, learned 1D position embeddings. No KV cache.',
  techniques: ['Patch Embedding', 'Learned 1D Pos Embedding', 'MHA', 'LayerNorm', 'GELU', 'CLS Token'],
  hyperparameters: [
    { symbol: 'H', value: '1280', name: 'hidden size' },
    { symbol: 'n_h', value: '16', name: 'attention heads' },
    { symbol: 'd_h', value: '80', name: 'head dim = H/n_h' },
    { symbol: 'F', value: '5120', name: 'FFN intermediate (4×H)' },
    { symbol: 'L', value: '32', name: 'encoder layers' },
    { symbol: 'P', value: '14', name: 'patch size (px)' },
    { symbol: 'N', value: '256', name: 'patches = (224/14)²' },
    { symbol: 'T', value: '257', name: 'sequence len = N+1 (CLS)' },
    { symbol: 'C_in', value: '3', name: 'input channels (RGB)' },
  ],
  notes: [
    {
      type: 'info',
      html: '<strong>Patch embedding:</strong> A 2D Conv (stride=patch_size) embeds non-overlapping 14×14 patches into 1280-dim vectors: <code>Conv2D(3, 1280, kernel=14, stride=14)</code>. The 224×224 image produces 16×16=256 patch tokens.',
    },
    {
      type: 'info',
      html: '<strong>Non-autoregressive — no prefill/decode split.</strong> ViT processes the full image in a single forward pass. Left column shows batch B=N (analogous to prefill); right column shows single image B=1 (analogous to decode).',
    },
    {
      type: 'info',
      html: '<strong>PIM relevance:</strong> At 632M parameters (~316 MB at FP16, ~158 MB at W8), ViT-Huge fits comfortably in a single DDR5 chip. With 16 heads and 32 banks, bank_dim=heads maps 2 banks per head. The non-standard d_h=80 requires padding to 96 or 128 for SIMD alignment (1536b ÷ 4b = 384 lanes; 80 does not divide 384).',
    },
  ],
  sections: [
    {
      title: '0 · Patch Embedding',
      tables: [{
        col1Header: '▶ Forward  (B = N images)',
        col1Class: 'phase-prefill',
        col2Header: '▶ Forward  (B = 1)',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Input image', sub: 'raw pixels',
            col1: [s('[B, 3, 224, 224]', 'act'), A, s('(B images)', 'scalar')],
            col2: [s('[1, 3, 224, 224]', 'act'), A, s('(1 image)', 'scalar')],
          },
          {
            name: 'Conv2D patch proj', sub: 'Conv(3→1280, k=14, s=14) → 16×16 grid',
            col1: [s('[B, 3, 224, 224]', 'act'), X, s('[1280, 3, 14, 14]', 'wt'), A, s('[B, 1280, 16, 16]', 'out')],
            col2: [s('[1, 3, 224, 224]', 'act'), X, s('[1280, 3, 14, 14]', 'wt'), A, s('[1, 1280, 16, 16]', 'out')],
          },
          {
            name: 'Flatten patches', sub: 'reshape: (16×16) → 256 tokens',
            col1: [s('[B, 1280, 16, 16]', 'act'), A, s('[B, 256, 1280]', 'out')],
            col2: [s('[1, 1280, 16, 16]', 'act'), A, s('[1, 256, 1280]', 'out')],
          },
          {
            name: 'Prepend CLS token', sub: 'cat([cls], patches, dim=1)',
            col1: [s('[B, 1, 1280]', 'scalar'), P, s('[B, 256, 1280]', 'act'), A, s('[B, 257, 1280]', 'out')],
            col2: [s('[1, 1, 1280]', 'scalar'), P, s('[1, 256, 1280]', 'act'), A, s('[1, 257, 1280]', 'out')],
          },
          {
            name: 'Add position embed', sub: 'learned [257, 1280] + broadcast',
            col1: [s('[B, 257, 1280]', 'act'), P, s('[1, 257, 1280]', 'scalar'), A, s('[B, 257, 1280]', 'out')],
            col2: [s('[1, 257, 1280]', 'act'), P, s('[1, 257, 1280]', 'scalar'), A, s('[1, 257, 1280]', 'out')],
          },
        ],
      }],
    },
    {
      title: 'Encoder Block',
      repeat: 32,
      repeatLabel: 'Repeats <span>32 times</span>. Full (bidirectional) attention — no causal mask.',
      tables: [
        {
          col1Header: '▶ Forward  (B = N)',
          col1Class: 'phase-prefill',
          col2Header: '▶ Forward  (B = 1)',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm', sub: 'norm1 (pre-attn)',
              col1: [s('[B, 257, 1280]', 'act'), X, s('[1280]', 'wt'), A, s('[B, 257, 1280]', 'out')],
              col2: [s('[1, 257, 1280]', 'act'), X, s('[1280]', 'wt'), A, s('[1, 257, 1280]', 'out')],
            },
          ],
        },
        {
          col1Header: '▶ Forward  (B = N)',
          col1Class: 'phase-prefill',
          col2Header: '▶ Forward  (B = 1)',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'Q projection', sub: 'q_proj [1280→1280]',
              col1: [s('[B, 257, 1280]', 'act'), X, s('[1280, 1280]', 'wt'), A, s('[B, 257, 1280]', 'out')],
              col2: [s('[1, 257, 1280]', 'act'), X, s('[1280, 1280]', 'wt'), A, s('[1, 257, 1280]', 'out')],
            },
            {
              name: 'K projection', sub: 'k_proj [1280→1280]',
              col1: [s('[B, 257, 1280]', 'act'), X, s('[1280, 1280]', 'wt'), A, s('[B, 257, 1280]', 'out')],
              col2: [s('[1, 257, 1280]', 'act'), X, s('[1280, 1280]', 'wt'), A, s('[1, 257, 1280]', 'out')],
            },
            {
              name: 'V projection', sub: 'v_proj [1280→1280]',
              col1: [s('[B, 257, 1280]', 'act'), X, s('[1280, 1280]', 'wt'), A, s('[B, 257, 1280]', 'out')],
              col2: [s('[1, 257, 1280]', 'act'), X, s('[1280, 1280]', 'wt'), A, s('[1, 257, 1280]', 'out')],
            },
            {
              name: 'Reshape Q/K/V', sub: 'split 16 heads (d_h=80)',
              col1: [s('[B, 257, 1280]', 'act'), A, s('[B, 16, 257, 80]', 'out'), X, s('(×3)', 'scalar')],
              col2: [s('[1, 257, 1280]', 'act'), A, s('[1, 16, 257, 80]', 'out'), X, s('(×3)', 'scalar')],
            },
            {
              name: 'Q · K\u1d40', sub: 'full (bidirectional) attn scores',
              col1: [s('[B, 16, 257, 80]', 'act'), X, s('[B, 16, 80, 257]', 'act'), A, s('[B, 16, 257, 257]', 'out')],
              col2: [s('[1, 16, 257, 80]', 'act'), X, s('[1, 16, 80, 257]', 'act'), A, s('[1, 16, 257, 257]', 'out')],
            },
            {
              name: 'Scale ÷ √80', sub: '× 0.1118  (NO causal mask)',
              col1: [s('[B, 16, 257, 257]', 'act'), X, s('0.1118', 'scalar'), A, s('[B, 16, 257, 257]', 'out'), note('no mask')],
              col2: [s('[1, 16, 257, 257]', 'act'), X, s('0.1118', 'scalar'), A, s('[1, 16, 257, 257]', 'out'), note('no mask')],
            },
            {
              name: 'Softmax + Attn · V', sub: 'context vectors',
              col1: [s('[B, 16, 257, 257]', 'act'), X, s('[B, 16, 257, 80]', 'act'), A, s('[B, 16, 257, 80]', 'out')],
              col2: [s('[1, 16, 257, 257]', 'act'), X, s('[1, 16, 257, 80]', 'act'), A, s('[1, 16, 257, 80]', 'out')],
            },
            {
              name: 'Reshape + O proj', sub: 'merge → proj [1280→1280]',
              col1: [s('[B, 257, 1280]', 'act'), X, s('[1280, 1280]', 'wt'), A, s('[B, 257, 1280]', 'out')],
              col2: [s('[1, 257, 1280]', 'act'), X, s('[1280, 1280]', 'wt'), A, s('[1, 257, 1280]', 'out')],
            },
            {
              name: 'Residual add', sub: 'skip connection',
              col1: [s('[B, 257, 1280]', 'act'), P, s('[B, 257, 1280]', 'out'), A, s('[B, 257, 1280]', 'out')],
              col2: [s('[1, 257, 1280]', 'act'), P, s('[1, 257, 1280]', 'out'), A, s('[1, 257, 1280]', 'out')],
            },
          ],
        },
        {
          col1Header: '▶ Forward  (B = N)',
          col1Class: 'phase-prefill',
          col2Header: '▶ Forward  (B = 1)',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm', sub: 'norm2 (pre-FFN)',
              col1: [s('[B, 257, 1280]', 'act'), X, s('[1280]', 'wt'), A, s('[B, 257, 1280]', 'out')],
              col2: [s('[1, 257, 1280]', 'act'), X, s('[1280]', 'wt'), A, s('[1, 257, 1280]', 'out')],
            },
            {
              name: 'FC up', sub: 'mlp.fc1 [1280→5120]',
              col1: [s('[B, 257, 1280]', 'act'), X, s('[5120, 1280]', 'wt'), A, s('[B, 257, 5120]', 'out')],
              col2: [s('[1, 257, 1280]', 'act'), X, s('[5120, 1280]', 'wt'), A, s('[1, 257, 5120]', 'out')],
            },
            {
              name: 'GELU', sub: 'elementwise activation',
              col1: [s('[B, 257, 5120]', 'act'), A, s('[B, 257, 5120]', 'out')],
              col2: [s('[1, 257, 5120]', 'act'), A, s('[1, 257, 5120]', 'out')],
            },
            {
              name: 'FC down', sub: 'mlp.fc2 [5120→1280]',
              col1: [s('[B, 257, 5120]', 'act'), X, s('[1280, 5120]', 'wt'), A, s('[B, 257, 1280]', 'out')],
              col2: [s('[1, 257, 5120]', 'act'), X, s('[1280, 5120]', 'wt'), A, s('[1, 257, 1280]', 'out')],
            },
            {
              name: 'Residual add', sub: 'skip connection',
              col1: [s('[B, 257, 1280]', 'act'), P, s('[B, 257, 1280]', 'out'), A, s('[B, 257, 1280]', 'out')],
              col2: [s('[1, 257, 1280]', 'act'), P, s('[1, 257, 1280]', 'out'), A, s('[1, 257, 1280]', 'out')],
            },
          ],
        },
      ],
    },
    {
      title: '2 · Classification Head',
      tables: [{
        col1Header: '▶ Forward  (B = N)',
        col1Class: 'phase-prefill',
        col2Header: '▶ Forward  (B = 1)',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'LayerNorm', sub: 'final norm',
            col1: [s('[B, 257, 1280]', 'act'), X, s('[1280]', 'wt'), A, s('[B, 257, 1280]', 'out')],
            col2: [s('[1, 257, 1280]', 'act'), X, s('[1280]', 'wt'), A, s('[1, 257, 1280]', 'out')],
          },
          {
            name: 'CLS extract', sub: 'token 0 only',
            col1: [s('[B, 257, 1280]', 'act'), A, s('[B, 1280]', 'out')],
            col2: [s('[1, 257, 1280]', 'act'), A, s('[1, 1280]', 'out')],
          },
          {
            name: 'Linear head', sub: 'classification [1280→num_classes]',
            col1: [s('[B, 1280]', 'act'), X, s('[1000, 1280]', 'wt'), A, s('[B, 1000]', 'out')],
            col2: [s('[1, 1280]', 'act'), X, s('[1000, 1280]', 'wt'), A, s('[1, 1000]', 'out')],
          },
        ],
      }],
    },
  ],
};
