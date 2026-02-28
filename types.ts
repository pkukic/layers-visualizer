export type BadgeType = 'act' | 'wt' | 'out' | 'cache' | 'scalar' | 'gemv';

export type Operand =
  | { kind: 'shape'; shape: string; badge: BadgeType }
  | { kind: 'op'; symbol: string }
  | { kind: 'note'; text: string };

export interface Operation {
  name: string;
  sub: string;
  col1: Operand[];
  col2?: Operand[];
}

export interface OpTableSpec {
  col1Header: string;
  col1Class?: string;
  col2Header?: string;
  col2Class?: string;
  operations: Operation[];
}

export interface Section {
  title: string;
  repeat?: number;
  repeatLabel?: string;
  note?: string;
  tables: OpTableSpec[];
}

export interface Hyperparameter {
  symbol: string;
  value: string;
  name: string;
}

export interface KVLine {
  label: string;
  value: string;
}

export interface KVCacheSpec {
  title: string;
  columns: { title: string; lines: KVLine[] }[];
}

export interface ModelNote {
  type: 'gemv' | 'info' | 'warning';
  html: string;
}

export type ModelCategory = 'llm' | 'vision' | 'audio' | 'vlm' | 'hybrid';

export interface ModelSpec {
  id: string;
  name: string;
  family: string;
  category: ModelCategory;
  paramCount: string;
  subtitle: string;
  techniques: string[];
  hyperparameters: Hyperparameter[];
  notes?: ModelNote[];
  sections: Section[];
  kvCache?: KVCacheSpec;
}
