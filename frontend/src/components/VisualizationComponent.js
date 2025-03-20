import React from 'react';
import { Box, Typography, Paper } from '@mui/material';

/**
 * Component for visualizing search result data
 * 
 * @param {Object} props - Component props
 * @param {Object} props.data - The search result data
 * @param {string} props.query - The search query
 */
const VisualizationComponent = ({ data, query }) => {
  return (
    <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        Search Results Visualization
      </Typography>
      <Typography variant="body1">
        Visualization for query: "{query}"
      </Typography>
      
      {/* You can expand this with charts or other visualizations later */}
      <Box sx={{ mt: 2 }}>
        <Typography variant="body2" color="text.secondary">
          {data ? `Found ${Object.keys(data.sourceResults || {}).length} sources with results` : 'No data available'}
        </Typography>
      </Box>
    </Paper>
  );
};

export default VisualizationComponent; 