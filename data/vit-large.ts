import { ModelSpec } from '../types';

const s = (shape: string, badge: 'act'|'wt'|'out'|'cache'|'scalar'|'gemv') =>
  ({ kind: 'shape' as const, shape, badge });
const op = (symbol: string) => ({ kind: 'op' as const, symbol });
const note = (text: string) => ({ kind: 'note' as const, text });

const A = op('→'), X = op('×'), P = op('+');

// ViT-Large/16: 304M params
// H = 1024, n_h = 16, d_h = 64, F = 4096 (4×H)
// L = 24 encoder layers
// Patch size 16 → 14×14 = 196 patches → T = 197 with CLS

export const vitLarge: ModelSpec = {
  id: 'vit-large',
  name: 'ViT-Large/16',
  family: 'ViT',
  category: 'vision',
  paramCount: '304M',
  subtitle: 'Encoder-only vision transformer (Large variant). Image split into 16×16 patches → sequence of 196+1 tokens. Isotropic depth (24 layers), MHA with 16 heads, LayerNorm, GELU FFN, learned 1D position embeddings. No KV cache.',
  techniques: ['Patch Embedding', 'Learned 1D Pos Embedding', 'MHA', 'LayerNorm', 'GELU', 'CLS Token'],
  hyperparameters: [
    { symbol: 'H', value: '1024', name: 'hidden size' },
    { symbol: 'n_h', value: '16', name: 'attention heads' },
    { symbol: 'd_h', value: '64', name: 'head dim = H/n_h' },
    { symbol: 'F', value: '4096', name: 'FFN intermediate (4×H)' },
    { symbol: 'L', value: '24', name: 'encoder layers' },
    { symbol: 'P', value: '16', name: 'patch size (px)' },
    { symbol: 'N', value: '196', name: 'patches = (224/16)²' },
    { symbol: 'T', value: '197', name: 'sequence len = N+1 (CLS)' },
    { symbol: 'C_in', value: '3', name: 'input channels (RGB)' },
  ],
  notes: [
    {
      type: 'info',
      html: '<strong>Patch embedding:</strong> A 2D Conv (stride=patch_size) embeds non-overlapping 16×16 patches into 1024-dim vectors: <code>Conv2D(3, 1024, kernel=16, stride=16)</code>. The 224×224 image produces 14×14=196 patch tokens.',
    },
    {
      type: 'info',
      html: '<strong>Non-autoregressive — no prefill/decode split.</strong> ViT processes the full image in a single forward pass. Left column shows batch B=N (analogous to prefill); right column shows single image B=1 (analogous to decode).',
    },
    {
      type: 'info',
      html: '<strong>PIM relevance:</strong> At 304M parameters (~152 MB at FP16, ~76 MB at W8), ViT-Large fits trivially in a single DDR5 chip. With 16 heads and 32 banks, bank_dim=heads requires 2 banks/head or padding to 32 heads. The primary bottleneck is the 4096-wide FFN.',
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
            name: 'Conv2D patch proj', sub: 'Conv(3→1024, k=16, s=16) → 14×14 grid',
            col1: [s('[B, 3, 224, 224]', 'act'), X, s('[1024, 3, 16, 16]', 'wt'), A, s('[B, 1024, 14, 14]', 'out')],
            col2: [s('[1, 3, 224, 224]', 'act'), X, s('[1024, 3, 16, 16]', 'wt'), A, s('[1, 1024, 14, 14]', 'out')],
          },
          {
            name: 'Flatten patches', sub: 'reshape: (14×14) → 196 tokens',
            col1: [s('[B, 1024, 14, 14]', 'act'), A, s('[B, 196, 1024]', 'out')],
            col2: [s('[1, 1024, 14, 14]', 'act'), A, s('[1, 196, 1024]', 'out')],
          },
          {
            name: 'Prepend CLS token', sub: 'cat([cls], patches, dim=1)',
            col1: [s('[B, 1, 1024]', 'scalar'), P, s('[B, 196, 1024]', 'act'), A, s('[B, 197, 1024]', 'out')],
            col2: [s('[1, 1, 1024]', 'scalar'), P, s('[1, 196, 1024]', 'act'), A, s('[1, 197, 1024]', 'out')],
          },
          {
            name: 'Add position embed', sub: 'learned [197, 1024] + broadcast',
            col1: [s('[B, 197, 1024]', 'act'), P, s('[1, 197, 1024]', 'scalar'), A, s('[B, 197, 1024]', 'out')],
            col2: [s('[1, 197, 1024]', 'act'), P, s('[1, 197, 1024]', 'scalar'), A, s('[1, 197, 1024]', 'out')],
          },
        ],
      }],
    },
    {
      title: 'Encoder Block',
      repeat: 24,
      repeatLabel: 'Repeats <span>24 times</span>. Full (bidirectional) attention — no causal mask.',
      tables: [
        {
          col1Header: '▶ Forward  (B = N)',
          col1Class: 'phase-prefill',
          col2Header: '▶ Forward  (B = 1)',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm', sub: 'norm1 (pre-attn)',
              col1: [s('[B, 197, 1024]', 'act'), X, s('[1024]', 'wt'), A, s('[B, 197, 1024]', 'out')],
              col2: [s('[1, 197, 1024]', 'act'), X, s('[1024]', 'wt'), A, s('[1, 197, 1024]', 'out')],
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
              name: 'Q projection', sub: 'q_proj [1024→1024]',
              col1: [s('[B, 197, 1024]', 'act'), X, s('[1024, 1024]', 'wt'), A, s('[B, 197, 1024]', 'out')],
              col2: [s('[1, 197, 1024]', 'act'), X, s('[1024, 1024]', 'wt'), A, s('[1, 197, 1024]', 'out')],
            },
            {
              name: 'K projection', sub: 'k_proj [1024→1024]',
              col1: [s('[B, 197, 1024]', 'act'), X, s('[1024, 1024]', 'wt'), A, s('[B, 197, 1024]', 'out')],
              col2: [s('[1, 197, 1024]', 'act'), X, s('[1024, 1024]', 'wt'), A, s('[1, 197, 1024]', 'out')],
            },
            {
              name: 'V projection', sub: 'v_proj [1024→1024]',
              col1: [s('[B, 197, 1024]', 'act'), X, s('[1024, 1024]', 'wt'), A, s('[B, 197, 1024]', 'out')],
              col2: [s('[1, 197, 1024]', 'act'), X, s('[1024, 1024]', 'wt'), A, s('[1, 197, 1024]', 'out')],
            },
            {
              name: 'Reshape Q/K/V', sub: 'split 16 heads',
              col1: [s('[B, 197, 1024]', 'act'), A, s('[B, 16, 197, 64]', 'out'), X, s('(×3)', 'scalar')],
              col2: [s('[1, 197, 1024]', 'act'), A, s('[1, 16, 197, 64]', 'out'), X, s('(×3)', 'scalar')],
            },
            {
              name: 'Q · K\u1d40', sub: 'full (bidirectional) attn scores',
              col1: [s('[B, 16, 197, 64]', 'act'), X, s('[B, 16, 64, 197]', 'act'), A, s('[B, 16, 197, 197]', 'out')],
              col2: [s('[1, 16, 197, 64]', 'act'), X, s('[1, 16, 64, 197]', 'act'), A, s('[1, 16, 197, 197]', 'out')],
            },
            {
              name: 'Scale ÷ √64', sub: '× 0.125  (NO causal mask)',
              col1: [s('[B, 16, 197, 197]', 'act'), X, s('0.125', 'scalar'), A, s('[B, 16, 197, 197]', 'out'), note('no mask')],
              col2: [s('[1, 16, 197, 197]', 'act'), X, s('0.125', 'scalar'), A, s('[1, 16, 197, 197]', 'out'), note('no mask')],
            },
            {
              name: 'Softmax + Attn · V', sub: 'context vectors',
              col1: [s('[B, 16, 197, 197]', 'act'), X, s('[B, 16, 197, 64]', 'act'), A, s('[B, 16, 197, 64]', 'out')],
              col2: [s('[1, 16, 197, 197]', 'act'), X, s('[1, 16, 197, 64]', 'act'), A, s('[1, 16, 197, 64]', 'out')],
            },
            {
              name: 'Reshape + O proj', sub: 'merge → proj [1024→1024]',
              col1: [s('[B, 197, 1024]', 'act'), X, s('[1024, 1024]', 'wt'), A, s('[B, 197, 1024]', 'out')],
              col2: [s('[1, 197, 1024]', 'act'), X, s('[1024, 1024]', 'wt'), A, s('[1, 197, 1024]', 'out')],
            },
            {
              name: 'Residual add', sub: 'skip connection',
              col1: [s('[B, 197, 1024]', 'act'), P, s('[B, 197, 1024]', 'out'), A, s('[B, 197, 1024]', 'out')],
              col2: [s('[1, 197, 1024]', 'act'), P, s('[1, 197, 1024]', 'out'), A, s('[1, 197, 1024]', 'out')],
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
              col1: [s('[B, 197, 1024]', 'act'), X, s('[1024]', 'wt'), A, s('[B, 197, 1024]', 'out')],
              col2: [s('[1, 197, 1024]', 'act'), X, s('[1024]', 'wt'), A, s('[1, 197, 1024]', 'out')],
            },
            {
              name: 'FC up', sub: 'mlp.fc1 [1024→4096]',
              col1: [s('[B, 197, 1024]', 'act'), X, s('[4096, 1024]', 'wt'), A, s('[B, 197, 4096]', 'out')],
              col2: [s('[1, 197, 1024]', 'act'), X, s('[4096, 1024]', 'wt'), A, s('[1, 197, 4096]', 'out')],
            },
            {
              name: 'GELU', sub: 'elementwise activation',
              col1: [s('[B, 197, 4096]', 'act'), A, s('[B, 197, 4096]', 'out')],
              col2: [s('[1, 197, 4096]', 'act'), A, s('[1, 197, 4096]', 'out')],
            },
            {
              name: 'FC down', sub: 'mlp.fc2 [4096→1024]',
              col1: [s('[B, 197, 4096]', 'act'), X, s('[1024, 4096]', 'wt'), A, s('[B, 197, 1024]', 'out')],
              col2: [s('[1, 197, 4096]', 'act'), X, s('[1024, 4096]', 'wt'), A, s('[1, 197, 1024]', 'out')],
            },
            {
              name: 'Residual add', sub: 'skip connection',
              col1: [s('[B, 197, 1024]', 'act'), P, s('[B, 197, 1024]', 'out'), A, s('[B, 197, 1024]', 'out')],
              col2: [s('[1, 197, 1024]', 'act'), P, s('[1, 197, 1024]', 'out'), A, s('[1, 197, 1024]', 'out')],
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
            col1: [s('[B, 197, 1024]', 'act'), X, s('[1024]', 'wt'), A, s('[B, 197, 1024]', 'out')],
            col2: [s('[1, 197, 1024]', 'act'), X, s('[1024]', 'wt'), A, s('[1, 197, 1024]', 'out')],
          },
          {
            name: 'CLS extract', sub: 'token 0 only',
            col1: [s('[B, 197, 1024]', 'act'), A, s('[B, 1024]', 'out')],
            col2: [s('[1, 197, 1024]', 'act'), A, s('[1, 1024]', 'out')],
          },
          {
            name: 'Linear head', sub: 'classification [1024→num_classes]',
            col1: [s('[B, 1024]', 'act'), X, s('[1000, 1024]', 'wt'), A, s('[B, 1000]', 'out')],
            col2: [s('[1, 1024]', 'act'), X, s('[1000, 1024]', 'wt'), A, s('[1, 1000]', 'out')],
          },
        ],
      }],
    },
  ],
};
