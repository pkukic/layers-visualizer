import React from 'react';
import { Operand } from '../types';

interface Props {
  operands: Operand[];
}

const BADGE_CLASS: Record<string, string> = {
  act:    'sb sb-act',
  wt:     'sb sb-wt',
  out:    'sb sb-out',
  cache:  'sb sb-cache',
  scalar: 'sb sb-scalar',
  gemv:   'sb sb-gemv',
};

export const OperandCell: React.FC<Props> = ({ operands }) => (
  <div className="col-shapes-cell">
    {operands.map((op, i) => {
      if (op.kind === 'shape') {
        return (
          <span key={i} className={BADGE_CLASS[op.badge] ?? 'sb sb-act'}>
            {op.shape}
          </span>
        );
      }
      if (op.kind === 'op') {
        return <span key={i} className="op-sym">{op.symbol}</span>;
      }
      if (op.kind === 'note') {
        return <span key={i} className="note-pill">{op.text}</span>;
      }
      return null;
    })}
  </div>
);
