import React from 'react';
import { Box, Typography, Paper, Grid } from '@mui/material';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts';

/**
 * Component for visualizing search result data
 * 
 * @param {Object} props - Component props
 * @param {Object} props.data - The search result data
 * @param {string} props.query - The search query
 */
const VisualizationComponent = ({ data, query }) => {
  // Format source name for display
  const formatSourceName = (source) => {
    switch(source) {
      case 'ads':
        return 'ADS/SciX';
      case 'scholar':
        return 'Google Scholar';
      case 'webOfScience':
        return 'Web of Science';
      case 'semanticScholar':
        return 'Semantic Scholar';
      default:
        return source.charAt(0).toUpperCase() + source.slice(1);
    }
  };

  // Get source color
  const getSourceColor = (source) => {
    switch(source) {
      case 'ads':
        return '#E57373'; // Light red
      case 'scholar':
        return '#64B5F6'; // Light blue
      case 'webOfScience':
        return '#FFD54F'; // Light yellow
      case 'semanticScholar':
        return '#81C784'; // Light green
      default:
        return '#9E9E9E'; // Grey
    }
  };

  // Prepare data for the results count chart
  const prepareChartData = () => {
    if (!data || !data.sourceResults) return [];
    
    const chartData = [];
    
    // Get sources
    const sources = Object.keys(data.sourceResults || {});
    
    // Count items per source
    sources.forEach(source => {
      const results = data.sourceResults[source] || [];
      chartData.push({
        name: formatSourceName(source),
        results: results.length,
        color: getSourceColor(source)
      });
    });
    
    return chartData;
  };

  return (
    <Box>
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Results Overview
        </Typography>
        <Typography variant="body1" gutterBottom>
          Showing results for query: "{query}"
        </Typography>
        
        <Box sx={{ height: 300, mt: 4 }}>
          <Typography variant="h6" gutterBottom>
            Results per Source
          </Typography>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={prepareChartData()}
              margin={{ top: 20, right: 30, left: 20, bottom: 30 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="name" 
                tick={{ fontSize: 11 }} 
                height={80}
                tickMargin={10}
                angle={-45}
                textAnchor="end"
              />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Legend />
              <Bar dataKey="results" name="Number of Results">
                {prepareChartData().map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Box>
      </Paper>
    </Box>
  );
};

export default VisualizationComponent; 