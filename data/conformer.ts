import { ModelSpec } from '../types';

const s = (shape: string, badge: 'act'|'wt'|'out'|'cache'|'scalar'|'gemv') =>
  ({ kind: 'shape' as const, shape, badge });
const op = (symbol: string) => ({ kind: 'op' as const, symbol });
const note = (text: string) => ({ kind: 'note' as const, text });

const A = op('→'), X = op('×'), P = op('+'), E = op('⊙');

// Conformer-S (typical ASR config)
// H = 256, n_h = 4, d_h = 64, F = 2048 (8×H)
// Conv kernel k=31, L=16 layers
// Input: mel spectrogram 80-dim → subsampled by 4× (Conv2D stack)
// T_sub ≈ T_audio / 4 (subsampling)

export const conformer: ModelSpec = {
  id: 'conformer',
  name: 'Conformer',
  family: 'Conformer',
  category: 'hybrid',
  paramCount: '~100M',
  subtitle: 'Conv-Transformer hybrid for ASR. Each block: ½ FFN → Multi-Head Self-Attention → Depthwise Conv module → ½ FFN (Macaron layout). Convolutional subsampling frontend.',
  techniques: ['Macaron FFN', 'Depthwise Separable Conv', 'Conv-Transformer Hybrid', 'Convolutional Subsampling', 'Relative Pos Encoding', 'LayerNorm', 'Swish'],
  hyperparameters: [
    { symbol: 'H', value: '256', name: 'encoder hidden size' },
    { symbol: 'n_h', value: '4', name: 'attention heads' },
    { symbol: 'd_h', value: '64', name: 'head dim = H/n_h' },
    { symbol: 'F', value: '2048', name: 'FFN intermediate (8×H)' },
    { symbol: 'k', value: '31', name: 'depthwise conv kernel size' },
    { symbol: 'L', value: '16', name: 'conformer blocks' },
    { symbol: 'R', value: '4', name: 'subsampling reduction factor' },
    { symbol: 'T_sub', value: 'T/4', name: 'seq len after subsampling' },
  ],
  notes: [
    {
      type: 'info',
      html: '<strong>Macaron-Net (½ FFN bookend):</strong> The conformer block follows a "Macaron" structure: Feed-Forward (half-step, ×0.5) → MHSA → Conv → Feed-Forward (half-step, ×0.5). Each FFN contributes half its residual. This regularizes training and improves gradient flow.',
    },
    {
      type: 'info',
      html: '<strong>Conv Module:</strong> After MHSA, a depthwise separable convolution block is applied: LayerNorm → pointwise conv (expand H→2H) → gating (GLU) → <em>depthwise conv</em> (k=31, per-channel) → BatchNorm → Swish → pointwise conv (H→H). This captures local temporal patterns that attention may miss.',
    },
    {
      type: 'info',
      html: '<strong>Convolutional subsampling:</strong> Mel spectrograms are downsampled 4× via two strided Conv2D layers before entering the conformer stack, reducing sequence length from ~1000s of frames to ~250 tokens.',
    },
  ],
  sections: [
    {
      title: '0 · Subsampling Frontend',
      tables: [{
        col1Header: '▶ Forward (one utterance)',
        col1Class: 'phase-prefill',
        col2Header: '▶ Shape',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Input mel', sub: '80-dim filter bank features',
            col1: [s('[1, T, 80]', 'act'), A, note('T frames @ 10ms')],
            col2: [],
          },
          {
            name: 'Sub-sample Conv2D 1', sub: 'Conv2D(1→256, k=3, s=2) + ReLU',
            col1: [s('[1, 1, T, 80]', 'act'), X, s('[256, 1, 3, 3]', 'wt'), A, s('[1, 256, T/2, 39]', 'out')],
            col2: [note('2× time reduction')],
          },
          {
            name: 'Sub-sample Conv2D 2', sub: 'Conv2D(256→256, k=3, s=2) + ReLU',
            col1: [s('[1, 256, T/2, 39]', 'act'), X, s('[256, 256, 3, 3]', 'wt'), A, s('[1, 256, T/4, 19]', 'out')],
            col2: [note('4× total reduction')],
          },
          {
            name: 'Reshape + Linear', sub: '[T/4, 256×19] → [T/4, H=256]',
            col1: [s('[1, T/4, 256×19]', 'act'), X, s('[256, 256×19]', 'wt'), A, s('[1, T/4, 256]', 'out')],
            col2: [s('[1, T_sub, 256]', 'out'), note('T_sub ≈ T/4')],
          },
          {
            name: 'Dropout', sub: 'regularization',
            col1: [s('[1, T_sub, 256]', 'act'), A, s('[1, T_sub, 256]', 'out')],
            col2: [],
          },
        ],
      }],
    },
    {
      title: 'Conformer Block (Macaron)',
      repeat: 16,
      repeatLabel: 'Repeats <span>16 times</span>. Block order: ½FFN → MHSA → Conv → ½FFN → LN.',
      tables: [
        {
          col1Header: '▶ ½ Feed-Forward Module (first)',
          col1Class: 'phase-prefill',
          col2Header: '▶ Shape',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm', sub: 'ff1_norm',
              col1: [s('[1, T_sub, 256]', 'act'), X, s('[256]', 'wt'), A, s('[1, T_sub, 256]', 'out')],
              col2: [],
            },
            {
              name: 'FC up', sub: 'ff1.w_1 [256→2048]',
              col1: [s('[1, T_sub, 256]', 'act'), X, s('[2048, 256]', 'wt'), A, s('[1, T_sub, 2048]', 'out')],
              col2: [],
            },
            {
              name: 'Swish', sub: 'elementwise x·σ(x)',
              col1: [s('[1, T_sub, 2048]', 'act'), A, s('[1, T_sub, 2048]', 'out')],
              col2: [],
            },
            {
              name: 'Dropout + FC down', sub: 'ff1.w_2 [2048→256]',
              col1: [s('[1, T_sub, 2048]', 'act'), X, s('[256, 2048]', 'wt'), A, s('[1, T_sub, 256]', 'out')],
              col2: [],
            },
            {
              name: '½ residual add', sub: 'x = x + 0.5 × ff1(x)',
              col1: [s('[1, T_sub, 256]', 'act'), P, s('0.5', 'scalar'), E, s('[1, T_sub, 256]', 'out'), A, s('[1, T_sub, 256]', 'out')],
              col2: [note('half-step weight')],
            },
          ],
        },
        {
          col1Header: '▶ Multi-Head Self-Attention Module',
          col1Class: 'phase-prefill',
          col2Header: '▶ Shape',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm', sub: 'mhsa_norm',
              col1: [s('[1, T_sub, 256]', 'act'), X, s('[256]', 'wt'), A, s('[1, T_sub, 256]', 'out')],
              col2: [],
            },
            {
              name: 'Q/K/V proj', sub: '[256→256] each, 4 heads',
              col1: [s('[1, T_sub, 256]', 'act'), X, s('[256, 256]', 'wt'), A, s('[1, T_sub, 256]', 'out'), X, s('(×3)', 'scalar')],
              col2: [s('[1, 4, T_sub, 64]', 'out'), X, s('(×3)', 'scalar')],
            },
            {
              name: 'Rel-pos attn scores', sub: 'Q · K\u1d40 + rel pos encoding (Shaw-style)',
              col1: [s('[1, 4, T_sub, 64]', 'act'), X, s('[1, 4, 64, T_sub]', 'act'), A, s('[1, 4, T_sub, T_sub]', 'out')],
              col2: [note('+ rel pos bias R')],
            },
            {
              name: 'Scale ÷ √64 + softmax + attn·V', sub: 'full bidirectional attn (ASR: no mask)',
              col1: [s('[1, 4, T_sub, T_sub]', 'act'), X, s('[1, 4, T_sub, 64]', 'act'), A, s('[1, T_sub, 256]', 'out')],
              col2: [note('no causal mask')],
            },
            {
              name: 'O proj + dropout', sub: '[256→256]',
              col1: [s('[1, T_sub, 256]', 'act'), X, s('[256, 256]', 'wt'), A, s('[1, T_sub, 256]', 'out')],
              col2: [],
            },
            {
              name: 'Residual add', sub: 'x = x + attn_out',
              col1: [s('[1, T_sub, 256]', 'act'), P, s('[1, T_sub, 256]', 'out'), A, s('[1, T_sub, 256]', 'out')],
              col2: [],
            },
          ],
        },
        {
          col1Header: '▶ Convolution Module',
          col1Class: 'phase-prefill',
          col2Header: '▶ Shape',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm', sub: 'conv_norm',
              col1: [s('[1, T_sub, 256]', 'act'), X, s('[256]', 'wt'), A, s('[1, T_sub, 256]', 'out')],
              col2: [],
            },
            {
              name: 'Pointwise conv 1', sub: 'Conv1D(256→512, k=1) → expand',
              col1: [s('[1, T_sub, 256]', 'act'), X, s('[512, 256, 1]', 'wt'), A, s('[1, T_sub, 512]', 'out')],
              col2: [note('expand to 2H')],
            },
            {
              name: 'GLU gate', sub: 'x[:,256:] * sigmoid(x[:,:256])',
              col1: [s('[1, T_sub, 512]', 'act'), A, s('[1, T_sub, 256]', 'out')],
              col2: [note('halves channel dim')],
            },
            {
              name: 'Depthwise Conv', sub: 'Conv1D(256, k=31, groups=256) — per-channel!',
              col1: [s('[1, 256, T_sub]', 'act'), X, s('[256, 1, 31]', 'wt'), A, s('[1, 256, T_sub]', 'out')],
              col2: [note('k=31, groups=C (cheap)')],
            },
            {
              name: 'BatchNorm + Swish', sub: 'BN along T, then Swish',
              col1: [s('[1, 256, T_sub]', 'act'), A, s('[1, T_sub, 256]', 'out')],
              col2: [],
            },
            {
              name: 'Pointwise conv 2', sub: 'Conv1D(256→256, k=1) → project back',
              col1: [s('[1, T_sub, 256]', 'act'), X, s('[256, 256, 1]', 'wt'), A, s('[1, T_sub, 256]', 'out')],
              col2: [],
            },
            {
              name: 'Dropout + residual', sub: 'x = x + conv_out',
              col1: [s('[1, T_sub, 256]', 'act'), P, s('[1, T_sub, 256]', 'out'), A, s('[1, T_sub, 256]', 'out')],
              col2: [],
            },
          ],
        },
        {
          col1Header: '▶ ½ Feed-Forward Module (second)',
          col1Class: 'phase-prefill',
          col2Header: '▶ Shape',
          col2Class: 'phase-decode',
          operations: [
            {
              name: 'LayerNorm + FC up', sub: 'ff2_norm → ff2.w_1 [256→2048]',
              col1: [s('[1, T_sub, 256]', 'act'), X, s('[2048, 256]', 'wt'), A, s('[1, T_sub, 2048]', 'out')],
              col2: [],
            },
            {
              name: 'Swish + FC down', sub: 'Swish → ff2.w_2 [2048→256]',
              col1: [s('[1, T_sub, 2048]', 'act'), X, s('[256, 2048]', 'wt'), A, s('[1, T_sub, 256]', 'out')],
              col2: [],
            },
            {
              name: '½ residual add', sub: 'x = x + 0.5 × ff2(x)',
              col1: [s('[1, T_sub, 256]', 'act'), P, s('0.5', 'scalar'), E, s('[1, T_sub, 256]', 'out'), A, s('[1, T_sub, 256]', 'out')],
              col2: [note('half-step weight')],
            },
            {
              name: 'Block LayerNorm', sub: 'final block norm',
              col1: [s('[1, T_sub, 256]', 'act'), X, s('[256]', 'wt'), A, s('[1, T_sub, 256]', 'out')],
              col2: [],
            },
          ],
        },
      ],
    },
    {
      title: 'Output — CTC Head',
      tables: [{
        col1Header: '▶ Forward',
        col1Class: 'phase-prefill',
        col2Header: '▶ Shape',
        col2Class: 'phase-decode',
        operations: [
          {
            name: 'Linear + softmax', sub: 'CTC projection [256→vocab]',
            col1: [s('[1, T_sub, 256]', 'act'), X, s('[vocab, 256]', 'wt'), A, s('[1, T_sub, vocab]', 'out')],
            col2: [note('CTC: per-frame token probs')],
          },
          {
            name: 'Softmax', sub: 'token probability distribution',
            col1: [s('[1, T_sub, vocab]', 'act'), A, s('[1, T_sub, vocab]', 'out')],
            col2: [note('decoded via Viterbi / beam search')],
          },
        ],
      }],
    },
  ],
};
