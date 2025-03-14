import React from 'react';
import { Card, CardContent, CardHeader, Typography } from '@mui/material';
import _ from 'lodash';

const VennDiagram = ({ data }) => {
  if (!data || Object.keys(data).length === 0) {
    return (
      <Card>
        <CardHeader title="Result Overlap" />
        <CardContent>
          <Typography variant="body1">No overlap data available</Typography>
        </CardContent>
      </Card>
    );
  }

  // Calculate SVG dimensions
  const width = 400;
  const height = 300;
  const padding = 40;
  const circleRadius = 80;

  // Function to create a basic circle for the Venn diagram
  const createCircle = (cx, cy, label, size, count) => {
    const opacity = 0.7;
    const colors = {
      'ads': '#4285F4', // Blue
      'scholar': '#EA4335', // Red
      'webOfScience': '#34A853', // Green
      'semanticScholar': '#FBBC05' // Yellow
    };

    // Add this function to format the label
    const formatLabel = (label) => {
      switch(label) {
        case 'ads':
          return 'ADS/SciX';
        case 'scholar':
          return 'Google Scholar';
        case 'webOfScience':
          return 'Web of Science';
        case 'semanticScholar':
          return 'Semantic Scholar';
        default:
          return _.startCase(label);
      }
    };

    return (
      <g key={label}>
        <circle 
          cx={cx} 
          cy={cy} 
          r={size || circleRadius} 
          fill={colors[label] || '#999'} 
          fillOpacity={opacity}
          stroke={colors[label] || '#999'}
          strokeOpacity={0.9}
        />
        <text 
          x={cx} 
          y={cy - 15} 
          textAnchor="middle" 
          fill="#000" 
          fontWeight="bold"
        >
          {formatLabel(label)}
        </text>
        <text 
          x={cx} 
          y={cy + 10} 
          textAnchor="middle" 
          fill="#000" 
          fontWeight="bold"
        >
          {count}
        </text>
      </g>
    );
  };

  // Create SVG elements based on the number of sources
  const renderVennDiagram = () => {
    const keys = Object.keys(data);
    
    if (keys.length === 0) {
      return null;
    }
    
    // Two-circle Venn diagram
    if (keys.length === 1) {
      const key = keys[0];
      const { source1_name, source2_name, overlap, source1_only, source2_only } = data[key];
      
      return (
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
          {/* Left circle */}
          {createCircle(
            width / 2 - circleRadius / 2, 
            height / 2, 
            source1_name, 
            circleRadius, 
            source1_only + overlap
          )}
          
          {/* Right circle */}
          {createCircle(
            width / 2 + circleRadius / 2, 
            height / 2, 
            source2_name, 
            circleRadius, 
            source2_only + overlap
          )}
          
          {/* Overlap text */}
          <text 
            x={width / 2} 
            y={height / 2} 
            textAnchor="middle" 
            fill="#000" 
            fontWeight="bold"
          >
            {overlap}
          </text>
          
          {/* Labels for non-overlapping regions */}
          <text 
            x={width / 2 - circleRadius} 
            y={height / 2} 
            textAnchor="middle" 
            fill="#000"
          >
            {source1_only}
          </text>
          
          <text 
            x={width / 2 + circleRadius} 
            y={height / 2} 
            textAnchor="middle" 
            fill="#000"
          >
            {source2_only}
          </text>
        </svg>
      );
    }
    
    // Multiple sources - create a more complex visualization
    // This is a simplified approach for multiple sources
    const sources = new Set();
    keys.forEach(key => {
      const { source1_name, source2_name } = data[key];
      sources.add(source1_name);
      sources.add(source2_name);
    });
    
    const sourcesArray = Array.from(sources);
    const centerX = width / 2;
    const centerY = height / 2;
    const angleStep = (2 * Math.PI) / sourcesArray.length;
    
    return (
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {sourcesArray.map((source, index) => {
          const angle = index * angleStep;
          const cx = centerX + Math.cos(angle) * circleRadius * 0.8;
          const cy = centerY + Math.sin(angle) * circleRadius * 0.8;
          
          // Count results for this source
          let count = 0;
          keys.forEach(key => {
            const { source1_name, source2_name, source1_only, overlap } = data[key];
            if (source === source1_name) {
              count += source1_only + overlap;
            } else if (source === source2_name) {
              // We don't add overlap again to avoid double counting
              count += data[key].source2_only;
            }
          });
          
          return createCircle(cx, cy, source, circleRadius * 0.7, count);
        })}
        
        {/* Central overlap region */}
        <text 
          x={centerX} 
          y={centerY} 
          textAnchor="middle" 
          fill="#000" 
          fontWeight="bold"
        >
          {/* Calculate total overlap */}
          {keys.reduce((acc, key) => acc + data[key].overlap, 0) / keys.length}
        </text>
      </svg>
    );
  };

  return (
    <Card>
      <CardHeader title="Result Overlap" />
      <CardContent>
        {renderVennDiagram()}
      </CardContent>
    </Card>
  );
};

export default VennDiagram;