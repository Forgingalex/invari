import React from 'react';
import './FilterControls.css';

function FilterControls({ alpha, onAlphaChange, onProcess }) {
  return (
    <div className="filter-controls">
      <h3>Filter Parameters</h3>
      <div className="control-group">
        <label>
          Complementary Filter Alpha:
          <input
            type="number"
            min="0"
            max="1"
            step="0.01"
            value={alpha}
            onChange={(e) => onAlphaChange(parseFloat(e.target.value))}
          />
        </label>
        <small>Higher values trust gyroscope more (0-1)</small>
      </div>
    </div>
  );
}

export default FilterControls;

