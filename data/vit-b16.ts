import { ModelSpec } from '../types';

const s = (shape: string, badge: 'act'|'wt'|'out'|'cache'|'scalar'|'gemv') =>
  ({ kind: 'shape' as const, shape, badge });
const op = (symbol: string) => ({ kind: 'op' as const, symbol });
const note = (text: string) => ({ kind: 'note' as const, text });

const A = op('→'), X = op('×'), P = op('+');

export const vitB16: ModelSpec = {
  id: 'vit-b16',
  name: 'ViT-B/16',
  family: 'ViT',
  category: 'vision',
  paramCount: '86M',
  subtitle: 'Encoder-only vision transformer. Image split into 16×16 patches → sequence of 196+1 tokens. Isotropic depth, MHA, LayerNorm, GELU FFN, learned 1D position embeddings. No KV cache.',
  techniques: ['Patch Embedding', 'Learned 1D Pos Embedding', 'MHA', 'LayerNorm', 'GELU', 'CLS Token'],
  hyperparameters: [
    { symbol: 'H', value: '768', name: 'hidden size' },
    { symbol: 'n_h', value: '12', name: 'attention heads' },
    { symbol: 'd_h', value: '64', name: 'head dim = H/n_h' },
    { symbol: 'F', value: '3072', name: 'FFN intermediate (4×H)' },
    { symbol: 'L', value: '12', name: 'encoder layers' },
    { symbol: 'P', value: '16', name: 'patch size (px)' },
    { symbol: 'N', value: '196', name: 'patches = (224/16)²' },
    { symbol: 'T', value: '197', name: 'sequence len = N+1 (CLS)' },
    { symbol: 'C_in', value: '3', name: 'input channels (RGB)' },
  ],
  notes: [
    {
      type: 'info',
      html: '<strong>Patch embedding:</strong> A 2D Conv (stride=patch_size) embeds non-overlapping 16×16 patches into vectors: <code>Conv2D(3, 768, kernel=16, stride=16)</code>. The 224×224 image produces 14×14=196 patch tokens.',
    },
    {
      type: 'info',
      html: '<strong>Non-autoregressive — no prefill/decode split.</strong> ViT processes the full image in a single forward pass. Left column shows batch B=N (analogous to prefill); right column shows single image B=1 (analogous to decode).',
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
            name: 'Conv2D patch proj', sub: 'Conv(3→768, k=16, s=16) → 14×14 grid',
            col1: [s('[B, 3, 224, 224]', 'act'), X, s('[768, 3, 16, 16]', 'wt'), A, s('[B, 768, 14, 14]', 'out')],
            col2: [s('[1, 3, 224, 224]', 'act'), X, s('[768, 3, 16, 16]', 'wt'), A, s('[1, 768, 14, 14]', 'out')],
          },
          {
            name: 'Flatten patches', sub: 'reshape: (14×14) → 196 tokens',
            col1: [s('[B, 768, 14, 14]', 'act'), A, s('[B, 196, 768]', 'out')],
            col2: [s('[1, 768, 14, 14]', 'act'), A, s('[1, 196, 768]', 'out')],
          },
          {
            name: 'Prepend CLS token', sub: 'cat([cls], patches, dim=1)',
            col1: [s('[B, 1, 768]', 'scalar'), P, s('[B, 196, 768]', 'act'), A, s('[B, 197, 768]', 'out')],
            col2: [s('[1, 1, 768]', 'scalar'), P, s('[1, 196, 768]', 'act'), A, s('[1, 197, 768]', 'out')],
          },
          {
            name: 'Add position embed', sub: 'learned [197, 768] + broadcast',
            col1: [s('[B, 197, 768]', 'act'), P, s('[1, 197, 768]', 'scalar'), A, s('[B, 197, 768]', 'out')],
            col2: [s('[1, 197, 768]', 'act'), P, s('[1, 197, 768]', 'scalar'), A, s('[1, 197, 768]', 'out')],
          },
        ],
      }],
    },
    {
      title: 'Encoder Block',
      repeat: 12,
      repeatLabel: 'Repeats <span>12 times</span>. Full (bidirectional) attention — no causal mask.',
      tables: [
        {
          col1Header: '▶ Forward  (B = N)',
          col1Class: 'phase-prefill',
          col2Header: '▶ Forward  (B = 1)',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm', sub: 'norm1 (pre-attn)',
              col1: [s('[B, 197, 768]', 'act'), X, s('[768]', 'wt'), A, s('[B, 197, 768]', 'out')],
              col2: [s('[1, 197, 768]', 'act'), X, s('[768]', 'wt'), A, s('[1, 197, 768]', 'out')],
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
              name: 'Q projection', sub: 'q_proj [768→768]',
              col1: [s('[B, 197, 768]', 'act'), X, s('[768, 768]', 'wt'), A, s('[B, 197, 768]', 'out')],
              col2: [s('[1, 197, 768]', 'act'), X, s('[768, 768]', 'wt'), A, s('[1, 197, 768]', 'out')],
            },
            {
              name: 'K projection', sub: 'k_proj [768→768]',
              col1: [s('[B, 197, 768]', 'act'), X, s('[768, 768]', 'wt'), A, s('[B, 197, 768]', 'out')],
              col2: [s('[1, 197, 768]', 'act'), X, s('[768, 768]', 'wt'), A, s('[1, 197, 768]', 'out')],
            },
            {
              name: 'V projection', sub: 'v_proj [768→768]',
              col1: [s('[B, 197, 768]', 'act'), X, s('[768, 768]', 'wt'), A, s('[B, 197, 768]', 'out')],
              col2: [s('[1, 197, 768]', 'act'), X, s('[768, 768]', 'wt'), A, s('[1, 197, 768]', 'out')],
            },
            {
              name: 'Reshape Q/K/V', sub: 'split 12 heads',
              col1: [s('[B, 197, 768]', 'act'), A, s('[B, 12, 197, 64]', 'out'), X, s('(×3)', 'scalar')],
              col2: [s('[1, 197, 768]', 'act'), A, s('[1, 12, 197, 64]', 'out'), X, s('(×3)', 'scalar')],
            },
            {
              name: 'Q · K\u1d40', sub: 'full (bidirectional) attn scores',
              col1: [s('[B, 12, 197, 64]', 'act'), X, s('[B, 12, 64, 197]', 'act'), A, s('[B, 12, 197, 197]', 'out')],
              col2: [s('[1, 12, 197, 64]', 'act'), X, s('[1, 12, 64, 197]', 'act'), A, s('[1, 12, 197, 197]', 'out')],
            },
            {
              name: 'Scale ÷ √64', sub: '× 0.125  (NO causal mask)',
              col1: [s('[B, 12, 197, 197]', 'act'), X, s('0.125', 'scalar'), A, s('[B, 12, 197, 197]', 'out'), note('no mask')],
              col2: [s('[1, 12, 197, 197]', 'act'), X, s('0.125', 'scalar'), A, s('[1, 12, 197, 197]', 'out'), note('no mask')],
            },
            {
              name: 'Softmax + Attn · V', sub: 'context vectors',
              col1: [s('[B, 12, 197, 197]', 'act'), X, s('[B, 12, 197, 64]', 'act'), A, s('[B, 12, 197, 64]', 'out')],
              col2: [s('[1, 12, 197, 197]', 'act'), X, s('[1, 12, 197, 64]', 'act'), A, s('[1, 12, 197, 64]', 'out')],
            },
            {
              name: 'Reshape + O proj', sub: 'merge → proj [768→768]',
              col1: [s('[B, 197, 768]', 'act'), X, s('[768, 768]', 'wt'), A, s('[B, 197, 768]', 'out')],
              col2: [s('[1, 197, 768]', 'act'), X, s('[768, 768]', 'wt'), A, s('[1, 197, 768]', 'out')],
            },
            {
              name: 'Residual add', sub: 'skip connection',
              col1: [s('[B, 197, 768]', 'act'), P, s('[B, 197, 768]', 'out'), A, s('[B, 197, 768]', 'out')],
              col2: [s('[1, 197, 768]', 'act'), P, s('[1, 197, 768]', 'out'), A, s('[1, 197, 768]', 'out')],
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
              col1: [s('[B, 197, 768]', 'act'), X, s('[768]', 'wt'), A, s('[B, 197, 768]', 'out')],
              col2: [s('[1, 197, 768]', 'act'), X, s('[768]', 'wt'), A, s('[1, 197, 768]', 'out')],
            },
            {
              name: 'FC up', sub: 'mlp.fc1 [768→3072]',
              col1: [s('[B, 197, 768]', 'act'), X, s('[3072, 768]', 'wt'), A, s('[B, 197, 3072]', 'out')],
              col2: [s('[1, 197, 768]', 'act'), X, s('[3072, 768]', 'wt'), A, s('[1, 197, 3072]', 'out')],
            },
            {
              name: 'GELU', sub: 'elementwise activation',
              col1: [s('[B, 197, 3072]', 'act'), A, s('[B, 197, 3072]', 'out')],
              col2: [s('[1, 197, 3072]', 'act'), A, s('[1, 197, 3072]', 'out')],
            },
            {
              name: 'FC down', sub: 'mlp.fc2 [3072→768]',
              col1: [s('[B, 197, 3072]', 'act'), X, s('[768, 3072]', 'wt'), A, s('[B, 197, 768]', 'out')],
              col2: [s('[1, 197, 3072]', 'act'), X, s('[768, 3072]', 'wt'), A, s('[1, 197, 768]', 'out')],
            },
            {
              name: 'Residual add', sub: 'skip connection',
              col1: [s('[B, 197, 768]', 'act'), P, s('[B, 197, 768]', 'out'), A, s('[B, 197, 768]', 'out')],
              col2: [s('[1, 197, 768]', 'act'), P, s('[1, 197, 768]', 'out'), A, s('[1, 197, 768]', 'out')],
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
            col1: [s('[B, 197, 768]', 'act'), X, s('[768]', 'wt'), A, s('[B, 197, 768]', 'out')],
            col2: [s('[1, 197, 768]', 'act'), X, s('[768]', 'wt'), A, s('[1, 197, 768]', 'out')],
          },
          {
            name: 'CLS extract', sub: 'token 0 only',
            col1: [s('[B, 197, 768]', 'act'), A, s('[B, 768]', 'out')],
            col2: [s('[1, 197, 768]', 'act'), A, s('[1, 768]', 'out')],
          },
          {
            name: 'Linear head', sub: 'classification [768→num_classes]',
            col1: [s('[B, 768]', 'act'), X, s('[1000, 768]', 'wt'), A, s('[B, 1000]', 'out')],
            col2: [s('[1, 768]', 'act'), X, s('[1000, 768]', 'wt'), A, s('[1, 1000]', 'out')],
          },
        ],
      }],
    },
  ],
};
