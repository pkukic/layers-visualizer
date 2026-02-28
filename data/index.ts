export { gpt2 } from './gpt2';
export { llama2_7b } from './llama2-7b';
export { mistral7b } from './mistral-7b';
export { bloom7b1 } from './bloom-7b1';
export { gemma2_2b } from './gemma2-2b';
export { vitLarge } from './vit-large';
export { vitHuge } from './vit-huge';
export { paligemma3b } from './paligemma-3b';
export { whisperLarge } from './whisper-large';

import { gpt2 } from './gpt2';
import { llama2_7b } from './llama2-7b';
import { mistral7b } from './mistral-7b';
import { bloom7b1 } from './bloom-7b1';
import { gemma2_2b } from './gemma2-2b';
import { vitLarge } from './vit-large';
import { vitHuge } from './vit-huge';
import { paligemma3b } from './paligemma-3b';
import { whisperLarge } from './whisper-large';
import { ModelSpec, ModelCategory } from '../types';

export const ALL_MODELS: ModelSpec[] = [
  gpt2,
  llama2_7b,
  mistral7b,
  bloom7b1,
  gemma2_2b,
  vitLarge,
  vitHuge,
  paligemma3b,
  whisperLarge,
];

export const CATEGORY_LABELS: Record<ModelCategory, string> = {
  llm: 'Large Language Models',
  vision: 'Vision Transformers',
  audio: 'Audio / Speech',
  vlm: 'Vision-Language Models',
  hybrid: 'Conv-Transformer Hybrid',
};

export const CATEGORY_ORDER: ModelCategory[] = ['llm', 'vision', 'vlm', 'audio'];
