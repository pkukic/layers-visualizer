import { ModelSpec } from '../types';

const s = (shape: string, badge: 'act' | 'wt' | 'out' | 'cache' | 'scalar' | 'gemv') =>
  ({ kind: 'shape' as const, shape, badge });
const op = (symbol: string) => ({ kind: 'op' as const, symbol });
const note = (text: string) => ({ kind: 'note' as const, text });

const A = op('→'), X = op('×'), P = op('+'), E = op('⊙');

// Stage dims (Tiny): [56×56×96, 28×28×192, 14×14×384, 7×7×768]
// Window size: 7×7=49 tokens. n_heads: [3,6,12,24]. FFN ratio: 4×.

export const swinTiny: ModelSpec = {
  id: 'swin-tiny',
  name: 'Swin-T',
  family: 'Swin',
  category: 'vision',
  paramCount: '28M',
  subtitle: 'Hierarchical vision transformer with Shifted Window Attention (SW-MSA), relative position bias, GELU FFN, and 4-stage feature pyramid.',
  techniques: ['Shifted Window Attention', 'Relative Position Bias', 'Hierarchical Stages', 'Patch Merging', 'GELU'],
  hyperparameters: [
    { symbol: 'P', value: '4', name: 'initial patch size (px)' },
    { symbol: 'W', value: '7', name: 'window size (tokens)' },
    { symbol: 'C', value: '96', name: 'stage-1 channels' },
    { symbol: 'n_h', value: '[3,6,12,24]', name: 'heads per stage' },
    { symbol: 'F', value: '4×C', name: 'FFN ratio (4× per stage)' },
    { symbol: 'L', value: '[2,2,6,2]', name: 'blocks per stage' },
    { symbol: 'Stg', value: '4', name: 'stages (hierarchical)' },
  ],
  notes: [
    {
      type: 'info',
      html: '<strong>Shifted Window Attention:</strong> Alternating blocks use (1) regular W-MSA (each window attends within itself) and (2) SW-MSA (windows shifted by (⌊W/2⌋, ⌊W/2⌋) pixels, enabling cross-window information flow). This gives linear compute in image size.',
    },
    {
      type: 'info',
      html: '<strong>Relative position bias:</strong> A learned bias table B is added to attention logits: <code>Attn = softmax(QK\u1d40/√d + B) · V</code>. B has shape [(2W−1)², (2W−1)²] = [169, 169] for W=7, indexed by relative row/column offsets.',
    },
    {
      type: 'info',
      html: '<strong>Patch Merging</strong> (downsampling): Between stages, 2×2 neighbouring patches are concatenated and projected, halving spatial resolution and doubling channels: [H/2, W/2, 2C] → [H/4, W/4, 2C] → linear [2C→C].',
    },
  ],
  sections: [
    {
      title: '0 · Patch Partition & Embedding',
      tables: [{
        col1Header: '▶ Forward  (B = 1)',
        col1Class: 'phase-prefill',
        col2Header: '▶ Shapes at each stage',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Input image', sub: 'RGB 224×224',
            col1: [s('[1, 3, 224, 224]', 'act')],
            col2: [note('H=W=224, C_in=3')],
          },
          {
            name: 'Patch embed', sub: 'Conv2D(3→96, k=4, s=4) + LayerNorm',
            col1: [s('[1, 3, 224, 224]', 'act'), X, s('[96, 3, 4, 4]', 'wt'), A, s('[1, 96, 56, 56]', 'out')],
            col2: [s('[1, 56×56, 96]', 'out'), note('3136 tokens, C=96')],
          },
        ],
      }],
    },
    {
      title: 'Stage 1 — W-MSA / SW-MSA, C=96, n_h=3',
      repeat: 2,
      repeatLabel: '<span>2 blocks</span>: block 0 uses W-MSA (no shift), block 1 uses SW-MSA (shift=3). Input: [1, 56×56, 96] = [1, 3136, 96]. Window partition → [3136/49=64 windows, 49, 96].',
      tables: [
        {
          col1Header: '▶ W-MSA block (block 0)',
          col1Class: 'phase-prefill',
          col2Header: '▶ SW-MSA block (block 1)',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm', sub: 'norm1',
              col1: [s('[1, 3136, 96]', 'act'), X, s('[96]', 'wt'), A, s('[1, 3136, 96]', 'out')],
              col2: [s('[1, 3136, 96]', 'act'), X, s('[96]', 'wt'), A, s('[1, 3136, 96]', 'out')],
            },
            {
              name: 'Window partition', sub: '[H×W, C] → [nW, W²×W², C]',
              col1: [s('[1, 3136, 96]', 'act'), A, s('[64, 49, 96]', 'out')],
              col2: [s('[1, 3136, 96]', 'act'), A, s('[64, 49, 96]', 'out'), note('cyclic shift first')],
            },
            {
              name: 'W-MSA / SW-MSA', sub: 'Q,K,V proj + window attn + rel pos bias',
              col1: [s('[64, 49, 96]', 'act'), X, s('[288, 96]', 'wt'), A, s('[64, 49, 288]', 'out'), note('3× for QKV')],
              col2: [s('[64, 49, 96]', 'act'), X, s('[288, 96]', 'wt'), A, s('[64, 49, 288]', 'out'), note('masked for shift')],
            },
            {
              name: 'Reshape Q/K/V', sub: '3 heads of 32 each',
              col1: [s('[64, 49, 288]', 'act'), A, s('[64, 3, 49, 32]', 'out'), X, s('(×3)', 'scalar')],
              col2: [s('[64, 49, 288]', 'act'), A, s('[64, 3, 49, 32]', 'out'), X, s('(×3)', 'scalar')],
            },
            {
              name: 'Q · K\u1d40', sub: 'within-window attn scores',
              col1: [s('[64, 3, 49, 32]', 'act'), X, s('[64, 3, 32, 49]', 'act'), A, s('[64, 3, 49, 49]', 'out')],
              col2: [s('[64, 3, 49, 32]', 'act'), X, s('[64, 3, 32, 49]', 'act'), A, s('[64, 3, 49, 49]', 'out')],
            },
            {
              name: 'Rel pos bias', sub: '+ B[169,169] indexed by (Δrow, Δcol)',
              col1: [s('[64, 3, 49, 49]', 'act'), P, s('[3, 49, 49]', 'scalar'), A, s('[64, 3, 49, 49]', 'out')],
              col2: [s('[64, 3, 49, 49]', 'act'), P, s('[3, 49, 49]', 'scalar'), P, s('attn_mask', 'scalar'), A, s('[64, 3, 49, 49]', 'out')],
            },
            {
              name: 'Softmax + Attn·V + proj', sub: 'context → proj [96→96]',
              col1: [s('[64, 3, 49, 49]', 'act'), X, s('[64, 3, 49, 32]', 'act'), A, s('[64, 49, 96]', 'out')],
              col2: [s('[64, 3, 49, 49]', 'act'), X, s('[64, 3, 49, 32]', 'act'), A, s('[64, 49, 96]', 'out')],
            },
            {
              name: 'Window merge + residual', sub: '[nW, 49, 96] → [1, 3136, 96]',
              col1: [s('[64, 49, 96]', 'act'), A, s('[1, 3136, 96]', 'out'), P, s('[1, 3136, 96]', 'act'), A, s('[1, 3136, 96]', 'out')],
              col2: [s('[64, 49, 96]', 'act'), A, s('[1, 3136, 96]', 'out'), P, s('[1, 3136, 96]', 'act'), A, s('[1, 3136, 96]', 'out'), note('reverse shift')],
            },
          ],
        },
        {
          col1Header: '▶ W-MSA block (block 0)',
          col1Class: 'phase-prefill',
          col2Header: '▶ SW-MSA block (block 1)',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm', sub: 'norm2',
              col1: [s('[1, 3136, 96]', 'act'), X, s('[96]', 'wt'), A, s('[1, 3136, 96]', 'out')],
              col2: [s('[1, 3136, 96]', 'act'), X, s('[96]', 'wt'), A, s('[1, 3136, 96]', 'out')],
            },
            {
              name: 'MLP FC up', sub: '[96→384]',
              col1: [s('[1, 3136, 96]', 'act'), X, s('[384, 96]', 'wt'), A, s('[1, 3136, 384]', 'out')],
              col2: [s('[1, 3136, 96]', 'act'), X, s('[384, 96]', 'wt'), A, s('[1, 3136, 384]', 'out')],
            },
            {
              name: 'GELU + MLP FC down', sub: '[384→96]',
              col1: [s('[1, 3136, 384]', 'act'), X, s('[96, 384]', 'wt'), A, s('[1, 3136, 96]', 'out')],
              col2: [s('[1, 3136, 384]', 'act'), X, s('[96, 384]', 'wt'), A, s('[1, 3136, 96]', 'out')],
            },
            {
              name: 'Residual add', sub: 'skip connection',
              col1: [s('[1, 3136, 96]', 'act'), P, s('[1, 3136, 96]', 'out'), A, s('[1, 3136, 96]', 'out')],
              col2: [s('[1, 3136, 96]', 'act'), P, s('[1, 3136, 96]', 'out'), A, s('[1, 3136, 96]', 'out')],
            },
          ],
        },
      ],
    },
    {
      title: 'Patch Merging 1→2  (2× downsample)',
      tables: [{
        col1Header: '▶ Forward',
        col1Class: 'phase-prefill',
        col2Header: '▶ Output shape',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Concat 2×2 neighbours', sub: '[H×W, C] → [H/2×W/2, 4C]',
            col1: [s('[1, 3136, 96]', 'act'), A, s('[1, 784, 384]', 'out')],
            col2: [note('56×56 → 28×28')],
          },
          {
            name: 'LayerNorm + Linear', sub: '[4C→2C] = [384→192]',
            col1: [s('[1, 784, 384]', 'act'), X, s('[192, 384]', 'wt'), A, s('[1, 784, 192]', 'out')],
            col2: [s('[1, 28×28, 192]', 'out')],
          },
        ],
      }],
    },
    {
      title: 'Stage 2 — C=192, n_h=6, 28×28',
      repeat: 2,
      repeatLabel: '<span>2 blocks</span> of W-MSA/SW-MSA. 784 tokens, 6 heads of 32.',
      tables: [{
        col1Header: '▶ W-MSA / SW-MSA (same as Stage 1)',
        col1Class: 'phase-prefill',
        col2Header: '▶ Output',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Window partition', sub: '784 tokens → [16 windows, 49, 192]',
            col1: [s('[1, 784, 192]', 'act'), A, s('[16, 49, 192]', 'out')],
            col2: [note('16 windows')],
          },
          {
            name: 'Q/K/V proj', sub: '6 heads × 32',
            col1: [s('[16, 49, 192]', 'act'), X, s('[576, 192]', 'wt'), A, s('[16, 49, 576]', 'out')],
            col2: [note('3× QKV fused')],
          },
          {
            name: 'Attn + proj + MLP', sub: 'same structure, 4×C FFN',
            col1: [s('[16, 49, 192]', 'act'), A, s('[1, 784, 192]', 'out')],
            col2: [s('[1, 784, 192]', 'out'), note('FFN: 192→768→192')],
          },
        ],
      }],
    },
    {
      title: 'Patch Merging 2→3  (2× downsample)',
      tables: [{
        col1Header: '▶ Forward',
        col1Class: 'phase-prefill',
        col2Header: '▶ Output',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Concat + LayerNorm + Linear', sub: '[784,192]→[196,384]',
            col1: [s('[1, 784, 192]', 'act'), A, s('[1, 784, 384]', 'act'), X, s('[384, 768]', 'wt'), A, s('[1, 196, 384]', 'out')],
            col2: [note('28×28 → 14×14, C=384')],
          },
        ],
      }],
    },
    {
      title: 'Stage 3 — C=384, n_h=12, 14×14',
      repeat: 6,
      repeatLabel: '<span>6 blocks</span> of W-MSA/SW-MSA. 196 tokens, 12 heads × 32.',
      tables: [{
        col1Header: '▶ W-MSA / SW-MSA',
        col1Class: 'phase-prefill',
        col2Header: '▶ Key shapes',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Window partition', sub: '196 tokens → [4 windows, 49, 384]',
            col1: [s('[1, 196, 384]', 'act'), A, s('[4, 49, 384]', 'out')],
            col2: [note('4 windows at 14×14')],
          },
          {
            name: 'Q/K/V + 12-head attn + proj', sub: 'heads×32 dim',
            col1: [s('[4, 49, 384]', 'act'), X, s('[1152, 384]', 'wt'), A, s('[4, 49, 1152]', 'out')],
            col2: [note('FFN: 384→1536→384')],
          },
          {
            name: 'MLP FFN (4×C)', sub: '[384→1536→384]',
            col1: [s('[1, 196, 384]', 'act'), X, s('[1536, 384]', 'wt'), A, s('[1, 196, 1536]', 'out')],
            col2: [s('[1, 196, 384]', 'out')],
          },
        ],
      }],
    },
    {
      title: 'Patch Merging 3→4 + Stage 4 — C=768, 7×7',
      tables: [{
        col1Header: '▶ Forward',
        col1Class: 'phase-prefill',
        col2Header: '▶ Key shapes',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Patch merging', sub: '[196,384]→[49,768]',
            col1: [s('[1, 196, 384]', 'act'), A, s('[1, 196, 768]', 'act'), X, s('[768, 1536]', 'wt'), A, s('[1, 49, 768]', 'out')],
            col2: [note('14×14 → 7×7, C=768')],
          },
          {
            name: 'Stage 4 W/SW-MSA (×2)', sub: '24 heads × 32, single 7×7 window',
            col1: [s('[1, 49, 768]', 'act'), X, s('[2304, 768]', 'wt'), A, s('[1, 49, 768]', 'out')],
            col2: [note('1 window (entire 7×7 feature map)')],
          },
          {
            name: 'FFN (4×C)', sub: '[768→3072→768]',
            col1: [s('[1, 49, 768]', 'act'), X, s('[3072, 768]', 'wt'), A, s('[1, 49, 3072]', 'out')],
            col2: [s('[1, 49, 768]', 'out')],
          },
        ],
      }],
    },
    {
      title: '5 · Classification Head',
      tables: [{
        col1Header: '▶ Forward',
        col1Class: 'phase-prefill',
        col2Header: '▶ Shape',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Adaptive avg pool', sub: '[1, 49, 768] → [1, 768]',
            col1: [s('[1, 49, 768]', 'act'), A, s('[1, 768]', 'out')],
            col2: [note('global average pooling')],
          },
          {
            name: 'LayerNorm', sub: 'norm on pooled features',
            col1: [s('[1, 768]', 'act'), X, s('[768]', 'wt'), A, s('[1, 768]', 'out')],
            col2: [],
          },
          {
            name: 'Linear classifier', sub: '[768→1000]',
            col1: [s('[1, 768]', 'act'), X, s('[1000, 768]', 'wt'), A, s('[1, 1000]', 'out')],
            col2: [note('ImageNet-1K classes')],
          },
        ],
      }],
    },
  ],
};
