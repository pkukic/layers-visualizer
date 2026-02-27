import React from 'react';
import { OpTableSpec } from '../types';
import { OperandCell } from './OperandCell';

interface Props {
  table: OpTableSpec;
  tableIndex: number;
}

export const OpTableView: React.FC<Props> = ({ table }) => {
  const hasCol2 = table.col2Header !== undefined;

  return (
    <div style={{ overflowX: 'auto', marginBottom: '1px' }}>
      <table className="op-table">
        <colgroup>
          <col style={{ width: '190px' }} />
          <col />
          {hasCol2 && <col />}
        </colgroup>
        <thead>
          <tr>
            <th style={{ color: '#9ca3af' }}>Operation</th>
            <th className={table.col1Class ?? ''}>{table.col1Header}</th>
            {hasCol2 && (
              <th className={table.col2Class ?? ''}>{table.col2Header}</th>
            )}
          </tr>
        </thead>
        <tbody>
          {table.operations.map((op, i) => (
            <tr key={i}>
              <td className="col-op-name">
                <div>{op.name}</div>
                {op.sub && <div className="col-op-sub">{op.sub}</div>}
              </td>
              <td className="col-shapes">
                <OperandCell operands={op.col1} />
              </td>
              {hasCol2 && (
                <td className="col-shapes">
                  <OperandCell operands={op.col2 ?? []} />
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
