export { gpt2 } from './gpt2';
export { llama2_7b } from './llama2-7b';
export { mistral7b } from './mistral-7b';
export { gemma2_9b } from './gemma2-9b';
export { falcon7b } from './falcon-7b';
export { bloom7b1 } from './bloom-7b1';
export { vitB16 } from './vit-b16';
export { swinTiny } from './swin-tiny';
export { whisperSmall } from './whisper-small';

import { gpt2 } from './gpt2';
import { llama2_7b } from './llama2-7b';
import { mistral7b } from './mistral-7b';
import { gemma2_9b } from './gemma2-9b';
import { falcon7b } from './falcon-7b';
import { bloom7b1 } from './bloom-7b1';
import { vitB16 } from './vit-b16';
import { swinTiny } from './swin-tiny';
import { whisperSmall } from './whisper-small';
import { ModelSpec, ModelCategory } from '../types';

export const ALL_MODELS: ModelSpec[] = [
  gpt2,
  llama2_7b,
  mistral7b,
  gemma2_9b,
  falcon7b,
  bloom7b1,
  vitB16,
  swinTiny,
  whisperSmall,
];

export const CATEGORY_LABELS: Record<ModelCategory, string> = {
  llm: 'Large Language Models',
  vision: 'Vision Transformers',
  audio: 'Audio / Speech',
  hybrid: 'Conv-Transformer Hybrid',
};

export const CATEGORY_ORDER: ModelCategory[] = ['llm', 'vision', 'audio'];
